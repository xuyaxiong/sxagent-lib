"""交互式对话：验证用的 CLI 入口，打包打的就是它。你当用户，技能从一个文件夹全量注入。

    uv run python -m app.chat                      # 默认 skills/
    uv run python -m app.chat --skills 别的目录

和跑批分工不同：跑批用预置的 `turns` 和断言，机器判通过率，验"稳不稳"；这里自由对话，验真实
节奏下能不能跑通，也用来手工试新技能。

技能启动时全量注入（**每个子目录是一个技能包**，入口 `SKILL.md`，frontmatter 给
`name` / `description`），但主 agent 只看得见这两个字段，全文按需 `read_skill` 取。跑完落一份
与跑批同格式的记录，在 `runs/<tag>/` 下。"""

# ruff: noqa: T201  # 交互式入口，输出即界面

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.runtime import (  # noqa: E402
    DEFAULT_SKILLS_DIR,
    append_trace,
    RUNS_DIR,
    ctx_warn,
    dump_artifacts,
    dump_run_record,
    ensure_app_dirs,
    install,
    snapshot,
)
from sxagent import aio
from sxagent.config import describe  # noqa: E402
from sxagent import recorder, session, trace  # noqa: E402
from sxagent.graph import build, next_pending  # noqa: E402
from sxagent.handoff.briefing import asked_text  # noqa: E402
from sxagent import artifacts as artifacts_mod  # noqa: E402 —— artifacts 全仓统一叫 artifacts_mod
from sxagent.registry import registry  # noqa: E402

# 路径一律从 `app.runtime` 取（`ROOT` / `RUNS_DIR` / `DEFAULT_SKILLS_DIR`），本文件不自己算：
# 打包后自己算会落到解包目录，和程序旁边那个目录对不上。
BAR = "─" * 74

#: 打这些就退出，大小写不敏感 —— 和 Ctrl-C / Ctrl-D 一样。
#: 空行不退出，见 `_session` 里的读取循环。
_EXIT_WORDS = frozenset({"exit", "quit", "退出", "再见"})

#: 在待答卡片下打这些 = 显式跳过，大小写不敏感。
#: 空行不算跳过 —— 见 `_read_card_answer` 的 docstring。
_SKIP_WORDS = frozenset({"skip", "跳过"})

#: 开场白里的标题，面向使用者的显示名。
APP_TITLE = "SX Agent"


def _cut(s: str, width: int) -> str:
    """按显示宽度截断（CJK 占 2 列），超了补 `…`。"""
    if _dw(s) <= width:
        return s
    out, w = "", 0
    for ch in s:
        cw = _dw(ch)
        if w + cw > width - 1:
            break
        out += ch
        w += cw
    return out + "…"


#: 技能清单的配色：名字亮、说明暗。非 tty / `NO_COLOR` 下 `paint()` 原样返回。
_SKILL_NAME_COLOR = "\033[97m"
_SKILL_DESC_COLOR = "\033[90m"


def _print_skills(skills, indent: str = "   ") -> None:
    """打技能清单：名字一行、说明一行，名字亮、说明暗。"""
    for s in skills:
        head, detail = _skill_label(s)
        print(f"{indent}· {paint(head, _SKILL_NAME_COLOR)}")
        if detail:
            print(f"{indent}  {paint(detail, _SKILL_DESC_COLOR)}")


def _skill_label(skill) -> tuple[str, str]:
    """技能 →（一句话能干什么，细节）。

    按 `description` 的第一个 `：` 切；没有就用 `name`。细节截到 68 列——开场白是给人扫
    一眼的。
    """
    desc = (skill.description or "").strip()
    head, sep, tail = desc.partition("：")
    if sep and head.strip() and tail.strip():
        return head.strip(), _cut(tail.strip(), 68)
    return skill.name, _cut(desc, 68)


# ──────────────────────────── 输出层：一套视觉语言 ────────────────────────────
#
# 每条输出带时间、标签和内容，各角色用不同标签和颜色区分。没有单独的"推理"通道：
# OpenAI 兼容端点不返回 `reasoning_content`，能显示的就是它说的话和它调的工具、参数。

#: 角色 →（标签 + ANSI 颜色）。颜色在非 tty / `NO_COLOR` 下自动关掉。
#: 标签是"图标 ＋ 英文名词"，宽度按显示列算（见 `_dw`），gutter 才对得齐。
_STYLE: dict[str, tuple[str, str]] = {
    "user":   ("👤 USER",     "\033[96m"),
    "agent":  ("🧠 AGENT",    "\033[92m"),
    "say":    ("🤖 EXECUTOR", "\033[96m"),
    "tool":   ("🔧 TOOL",     "\033[93m"),
    "ret":    ("🔙 RETURN",   "\033[90m"),
    "inject": ("📥 INJECT",   "\033[35m"),
    "decide": ("🧭 DECIDE",   "\033[95m"),
    "exec":   ("🚀 RUN",      "\033[94m"),
    "ask":    ("💬 ASK",      "\033[91m"),
    #: 答复通道的提示符，和 `USER` 共用同一条 stdin，标签必须能分辨谁在说话。
    "reply":  ("↩️ ANSWER",   "\033[91m"),
    "plat":   ("📮 PLATFORM", "\033[90m"),
    "done":   ("🏁 DONE",     "\033[92m"),
    "task":   ("📋 TASKS",    "\033[90m"),
    "ctx":    ("📊 CONTEXT",  "\033[90m"),
}

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _dw(s: str) -> int:
    """显示宽度，按终端实际渲染算，不按码点阈值。

    四条规则，缺一条就会歪：
      1. 零宽：变体选择符（U+FE0E/FE0F）与 ZWJ 不占列
      2. emoji 区（U+1F000–1FAFF）与杂项符号区（U+2600–27BF）算 2 列 ——
         `❓` 的 East Asian Width 是 W，`🗣` 是 N 却仍按 emoji 呈现，靠 EAW 判会漏
      3. 带 VS16 的算 2 列 —— `↩️`(U+21A9+FE0F) 的 EAW 是 N，不加这条会算成 1
      4. 其余按 East Asian Width：`W`/`F` 算 2，其他算 1
    """
    w, i, n = 0, 0, len(s)
    while i < n:
        c = s[i]
        if c in "︎️‍":  # 零宽：变体选择符 / ZWJ
            i += 1
            continue
        o = ord(c)
        if (
            (i + 1 < n and s[i + 1] == "️")  # 带 VS16 → emoji 呈现
            or 0x1F000 <= o <= 0x1FAFF
            or 0x2600 <= o <= 0x27BF
            or unicodedata.east_asian_width(c) in ("W", "F")
        ):
            w += 2
        else:
            w += 1
        i += 1
    return w


#: 正文层（亮色）只有"人和主 agent 的对话"，其余都是机制层（暗灰），靠颜色分、左边对齐。
_LABEL_W = 12   # gutter 里标签的宽度（最长的 EXECUTOR/PLATFORM 是 11 列）
#: gutter 里时间那段的列数 —— 按 `trace.fmt_clock` 的真实宽度算，改格式不用回来改这里。
_TS = f"[{trace.fmt_clock(trace.stamp())}] "

#: 读全文的工具：返回一个字都不贴，只报"读了多大一份、原文在哪"。判据是全文另有出处——
#: "另有出处"的正文贴一段，既不是全文也不是摘要，只是噪音。其余一律照原样贴、不截：
#: 平台注入、agent 的话、别的工具返回都是观察窗口。记录层不跟着改，要少看噪音在播报层砍。
_READ_TOOLS: dict[str, str] = {
    "read_skill": "技能包",
    "read_skill_file": "技能包",   # 渐进式披露第三层：技能自带的 references/ assets/
    "read_artifact": "产物落盘文件",
}
#: `read_user_turn` 不在此列：它返回的是用户原话（事实来源），是【用户原话】的披露出口。

_MAIN_ROLES = frozenset({"user", "agent"})
_MECH_COLOR = "\033[90m"     # 暗灰：机制日志统一压暗，靠标签区分类型


#: CLI 输出落盘的 sink（`main()` 里打开）：终端里滚过去的，去色后逐行落进 `_console.log`。
_SINK: "object | None" = None


def _to_file(line: str) -> None:
    if _SINK is not None:
        _SINK.write(line + "\n")
        _SINK.flush()


def paint(s: str, color: str = "") -> str:
    """上色（非 tty / `NO_COLOR` 时原样返回）。"""
    return f"{color}{s}\033[0m" if (_COLOR and color) else s


def _gutter_prompt(role: str) -> str:
    """提示符本身就是那行 gutter：`[时间] 标签 ┃ `，打的字接在后面，天然是一行。

    两个提示符同一个形状，使用说明放在上面一行（`_emit`），提示符里只留形状。
    """
    icon, _color = _STYLE[role]
    pad = " " * max(0, _LABEL_W - _dw(icon))
    return f"[{trace.fmt_clock(trace.stamp())}] {icon}{pad}┃ "


#: 折行的最小单位：英文词、标识符、`` `代码段` `` 各是一整块，其余逐字。不能逐字符折——
#: 标识符被拆成 `read_us` / `er_turn` 就读不懂，也没法拷去搜代码。三个分支别并成一个：
#: 反引号成对，带点的标识符（`SKILL.md`）要连上，其余逐字。
_TOKENS = re.compile(r"`[^`]+`|[0-9A-Za-z_]+(?:\.[0-9A-Za-z_]+)*|[\s\S]")


def _wrap(s: str, width: int) -> list[str]:
    """按显示宽度折行（CJK 占 2 列），优先断在词边界上。

    自己折是为了续行能带上同一道 gutter。续行还要补回行首缩进，否则段体折成"首行缩进、
    后面顶到 0 列"。所以 `"".join(_wrap(s, w))` 对带缩进的行不等于 `s`，缩进是故意重复的。
    """
    if width <= 1:
        return [s]
    lead = s[: len(s) - len(s.lstrip())]
    body = s[len(lead) :]
    if not body:
        return [s]
    inner = max(1, width - sum(_dw(c) for c in lead))
    out: list[str] = []
    cur, w = "", 0
    for tok in _TOKENS.findall(body):
        tw = sum(_dw(c) for c in tok)
        if w + tw > inner and cur:
            # 整块放不下 → 换行。不留半截词：下一行的开头就是这个完整 token。
            out.append(lead + cur)
            cur, w = "", 0
        if tw > inner:
            # 单个词比整行还宽（超长 URL / 标识符）—— 没有边界可用，只能硬切
            for c in tok:
                if w + _dw(c) > inner:
                    out.append(lead + cur)
                    cur, w = "", 0
                cur += c
                w += _dw(c)
            continue
        cur += tok
        w += tw
    out.append(lead + cur)
    return out


def _cap_line(text: str) -> str:
    """机制层摘要行的截断：切了必须标。

    单独成函数是为了只有一个来源 —— `_emit` 用它，测试的捕获器也用它。
    """
    if len(text) <= _LIVE_CAP:
        return text
    return text[:_LIVE_CAP] + f"（**平台截断：还有 {len(text) - _LIVE_CAP} 字未显示**）"


def _emit(role: str, text: str, stamp: datetime | None = None, *, cap: bool = False) -> None:
    """按统一版式打一条：`[时间] 标签 ┃ 内容`，续行对齐到 `┃`。

    两层靠颜色分，不靠缩进：正文层亮色、机制层暗灰，左边一律对齐。
    `cap=True` 的机制层摘要行超过 `_LIVE_CAP` 会带标记截断；决策块和注入块永不截。
    """
    if cap:
        text = _cap_line(text)
    icon, color = _STYLE[role]
    stamp = trace.fmt_clock(stamp or trace.stamp())
    pad = " " * max(0, _LABEL_W - _dw(icon))
    head = f"[{stamp}] {icon}{pad}┃ "
    indent = _dw(_TS) + _LABEL_W
    tint = color if role in _MAIN_ROLES else _MECH_COLOR
    cont = " " * indent + "┃ "
    # 可用宽度 = 终端列数 − gutter。管道运行没 tty 时用 100 而不是默认的 80：
    # 这个宽度也决定 `_console.log` 怎么折，80 会把记录折得过窄。
    avail = shutil.get_terminal_size(fallback=(100, 30)).columns - indent - 2
    lines = [
        ln for raw in ((text or "").splitlines() or [""]) for ln in _wrap(raw, avail)
    ]
    print(paint(head + lines[0], tint))
    _to_file(head + lines[0])
    for ln in lines[1:]:
        print(paint(cont + ln, tint))
        _to_file(cont + ln)


# ──────────────────────────── 事件订阅：展示层只认结构化事件 ────────────────────────────
#
# 订阅 `trace` 的事件总线，拿的是结构化事件，不读日志文件、不按字符串前缀认行。前缀、
# 缩进、gutter 都是本层的排版决定，记录层不知道它们存在。

#: 业务语义标签 → 显示角色。记录层只发"发生了什么"（`kind`），角色归属是本层的决定。
_ROLE_BY_KIND: dict[str, str] = {
    "dispatch": "plat",         # delegate 委派
    "supersede": "plat",        # 取代旧 Task
    "delegate_failed": "plat",  # 委派以 failed 收场
    "run_begin": "exec",        # run_task 开始（section）
    "run_end": "exec",          # run_task 结束
    "feed": "plat",             # 平台把用户原话投给待答任务（section）
    "retry": "plat",            # 这一轮抖动、已重试一次（stream.stream_agent）
    "stream_empty": "plat",     # 流式一步都没输出 → 退回 invoke 兜底（这一趟没有流式日志）
    "run_fault": "exec",        # 重试过仍失败 → 这一轮没跑成（executor.run_task）
    "ask": "exec",              # 执行器申报要信息（调了 `ask_user`）→ 这一轮结束
    "delivered": "exec",        # 执行器交付了产物 → 这一轮结束
    "writing": "exec",          # 正在写工具参数（写产物那段 —— 报"已收到 N 字"）
    "artifact": "exec",         # 产物落盘了（ID / 标题 / 字数 / 路径）
}

#: 有专门渲染分支的 kind（不走上表那条摘要行的路）。
_SPECIAL_KINDS = frozenset({
    "ask_main",            # 试答：开一个挂着的块，PROMPT/RESPONSE 归并进来
    "executor_briefing",   # 平台注入执行器的整份交接书（block）
    "waiting",             # 主 agent 答不上 → 登记待答（note）
    "relay",               # 待答透传（note）
    "llm_io",              # 那次 LLM 调用的 PROMPT/RESPONSE/失败提示
})

#: 故意不上屏的 kind —— 只进日志文件。写在这里 = 一个表态（不是漏了）。
_LOG_ONLY_KINDS = frozenset({
    "nudge",   # 推执行器继续的提示词（正文已由 `handoff("推回去")` 记档）
})

def declared_kinds() -> set[str]:
    """展示层表过态的 kind —— 上面三个集合的并集。

    新加的 kind 必须在这一层表态（上屏 / 有专门分支 / 只进日志），漏表态的后果是那一行
    静默消失。测试拿这个函数断它覆盖 `trace.KINDS`。
    """
    return set(_ROLE_BY_KIND) | set(_SPECIAL_KINDS) | set(_LOG_ONLY_KINDS)

#: 挂着的块：`[role, lines, ts]` —— 只用于把同一次 LLM 调用的 PROMPT/RESPONSE 归并成一块。
_PENDING: list | None = None

#: 播报行的长度上限（超了带标记截断）—— 只对机制层摘要行生效，决策块 / 注入块不截。
_LIVE_CAP = 400


def _close_pending() -> None:
    """先把挂着的块输出出来，再播下一条事件。块没闭合就播下一条，看到的因果就是倒的。"""
    global _PENDING
    if _PENDING is not None:
        _emit(_PENDING[0], "\n".join(_PENDING[1]), stamp=_PENDING[2])
        _PENDING = None


def _render_decision(ev: dict, *, with_who: bool = True) -> str:
    """决策事件 → 显示文本。排版归 `handoff.decision.render_decision`，本层只加载体：
    段头一行，整体缩 2 格。

    `with_who=False` 用于接在挂着的块尾巴上——那个块的标题已经是同一句话，也不再缩 2 格，
    否则看着像两层。
    """
    from sxagent.handoff.decision import render_decision

    body = render_decision(ev)
    if not with_who:
        return body
    return f"{ev['who']}\n" + "\n".join(f"  {ln}" for ln in body.splitlines())


#: 已经吼过的未知角色值 —— 同一个错误只吼一次，不刷屏。
_WARNED_ROLE: set[str] = set()

#: 同上：已经吼过的"没表过态的 kind"。
_WARNED_KIND: set[str] = set()

#: 角色 → 屏幕上的称呼。角色是逻辑（`trace.ROLES` 枚举），这里只是文案。
_ROLE_LABEL = {trace.AGENT: "主 agent", trace.EXECUTOR: "执行器"}


def _role_of(ev: dict) -> str:
    """事件里的角色，只认枚举值，不从措辞认。

    取不到或值不合法就当场吼一声（同一个值只吼一次），按"未知"上屏，不猜。
    """
    role = ev.get("role")
    if role in trace.ROLES:
        return role
    key = repr(role)
    if key not in _WARNED_ROLE:
        _WARNED_ROLE.add(key)
        _emit("plat", f"⚠️ 收到未知角色的事件：role={role!r}（type={ev.get('type')}）"
                      f" —— 按「未知」上屏。合法值：{sorted(trace.ROLES)}",
              stamp=ev.get("ts"))
    return "unknown"


#: 工具配对表：`(role, tool_call_id)` → 工具名，跨事件累积。
#:
#: `dump` 按 super-step 分批发，`tool_call` 和它的 `ToolMessage` 可能落在两次 dump 里，所以
#: 表要跨事件存活。键必须带 `role`：`tool_call_id` 只在一轮响应内唯一，主 agent 和执行器都
#: 从 `call_0` 起，不带角色会互相覆盖。表只增不减，所以设了上限；配对只发生在同一轮内，
#: 到顶整体清空是安全的。
_TOOL_NAMES: dict[tuple[str, str], str] = {}
_TOOL_NAMES_CAP = 2000


def _on_dump(ev: dict) -> None:
    """消息轨迹事件。消息对象就在事件里，所以文本用 `content`，谁说的看 `ev["role"]`，
    工具配对用 `(role, tool_call_id)`。"""
    role = _role_of(ev)   # 配对键的一半，两个 agent 的 id 空间会撞（见 `_TOOL_NAMES`）
    who = _ROLE_LABEL.get(role, "未知角色")
    for m in ev["messages"]:
        if len(_TOOL_NAMES) >= _TOOL_NAMES_CAP:
            _TOOL_NAMES.clear()   # 配对只在同一轮内，清空是安全的
        for c in getattr(m, "tool_calls", None) or []:
            if c.get("id"):
                _TOOL_NAMES[(role, c["id"])] = str(c.get("name"))
            # 不截参数：`delegate(task=...)` 就是交接本身，砍了就看不到交接了什么。
            _emit("tool", f"{who} → {c.get('name')}（{trace.brief_args(c.get('args') or {})}）",
                  stamp=ev["ts"])
        kind = type(m).__name__
        if kind == "AIMessage":
            continue  # 它说的话已由流式实时播过，dump 不重复播 AI 正文
        content = str(m.content)
        if kind == "ToolMessage":
            name = _TOOL_NAMES.get((role, getattr(m, "tool_call_id", "") or ""), "?")
            if name in _READ_TOOLS:
                # 读全文的工具一个字都不贴，见 `_READ_TOOLS`。
                _emit("ret", f"{name} 的返回：（**读了全文，{len(content)} 字** —— "
                             f"原文在{_READ_TOOLS[name]}里，**不在这里复述**）", stamp=ev["ts"])
            else:
                _emit("ret", f"{name} 的返回：\n{content}", stamp=ev["ts"])
        elif kind == "SystemMessage":
            # 不截断：平台注入是观察窗口
            _emit("inject", f"平台 → 主 agent（每轮注入的任务台账）：\n{content}", stamp=ev["ts"])


def _on_event(ev: dict) -> None:
    """事件总线的订阅入口：展示层从这里拿结构化事实，自己排版上屏。"""
    global _PENDING
    t, k = ev.get("type"), ev.get("kind") or ""
    # 同一次 LLM 调用的 PROMPT / RESPONSE / "未逐字记录"那条 note 都归并进挂着的试答块；
    # 其余事件到达先关块，时序不许倒。`decision` 不关块：它是这个块的另一半（结果）。
    # `note` 也算进来，否则块会先被关掉，一个块变三个。
    _merges_into_block = t in ("block", "note") and k == "llm_io" and _PENDING is not None
    if not _merges_into_block and t != "decision":
        _close_pending()

    if t == "stream":
        # 谁说的按 `role` 枚举取，不从 `who` 的措辞里猜（见 `_role_of`）
        _emit({trace.EXECUTOR: "say", trace.AGENT: "agent"}.get(_role_of(ev), "plat"),
              str(ev["text"]), stamp=ev["ts"])
    elif t == "dump":
        _on_dump(ev)
    elif t == "section":
        if k == "ask_main":
            # 试答开一个挂着的块：PROMPT 是这次决策的前提（`llm_io` 归并进来），四段是它的
            # 结果（`decision` 到达时接到同一块尾部）。标题取事件里的 `title`，不自己编。
            # PROMPT 默认不逐字记，那时归并进来的是一条 note，位置不变，块照样只有一个。
            _PENDING = ["decide", [str(ev["title"])], ev["ts"]]
        elif k in _ROLE_BY_KIND:
            _emit(_ROLE_BY_KIND[k], str(ev["title"]), stamp=ev["ts"], cap=True)
    elif t == "block":
        if k == "llm_io" and _PENDING is not None:
            _PENDING[1].append(str(ev["label"]))
            _PENDING[1] += str(ev["text"]).splitlines() or [""]
        elif k == "executor_briefing":
            # 平台注入给执行器的整份交接书（任务 / 上次做到哪 / 历次答复 / 可读产物）
            _emit("inject", "平台 → 执行器：本次交接的全部已知\n" + str(ev["text"]),
                  stamp=ev["ts"])
    elif t == "decision":
        # 接在挂着的块尾巴上：那是这次 LLM 调用的一端（PROMPT），四段是另一端；分开输出的话
        # 同一件事在屏幕上就是两个 DECIDE 块。没有挂着的块就自己输出一块。
        if _PENDING is not None:
            _PENDING[1] += _render_decision(ev, with_who=False).splitlines()
            _close_pending()
        else:
            _emit("decide", _render_decision(ev), stamp=ev["ts"])
    elif t == "note":
        if k == "waiting":
            _emit("decide", "分支：主 agent 答不上 → " + str(ev["text"]), stamp=ev["ts"])
        elif k == "relay":
            # 原文照上屏：这条 note 带的就是平台透传给用户的那段（执行器的原话），归属由 kind
            # 给 —— 不读文本，也不切前缀（`finalize: ` 是记录层的格式，见 `recorder._NOTE_HEAD`）。
            _emit("ask", "平台 → 用户：" + str(ev["text"])
                  + "\n（平台把执行器的话**原样**转达 —— 代码保证不丢）",
                  stamp=ev["ts"])
        elif k == "llm_io":
            # `llm_io` 的 note 必须看得见：它是"这次调用的 PROMPT 没逐字记"那条声明，
            # 静默丢掉就是违反"切了要标"。它落在块里原来该有 PROMPT 的位置。
            if _PENDING is not None:
                _PENDING[1] += [str(ev["text"])]
            else:
                _emit("decide", str(ev["text"]), stamp=ev["ts"])
        elif k in _ROLE_BY_KIND:
            _emit(_ROLE_BY_KIND[k], str(ev["text"]), stamp=ev["ts"], cap=True)
    elif t == "cut":
        # 切一刀就得看得见：标记落在块的头，不必长在切口上。
        _emit("plat", f"✂️ 截断 · {ev['where']}：保留 {ev['kept']} 字 / 原文 {ev['total']} 字"
                      + (f"｜{ev['see']}" if ev.get("see") else ""),
              stamp=ev["ts"], cap=True)
    # `handoff` 与未打 kind 的 note / section 只进日志文件，不上屏。

    # 收尾自检：kind 没归角色、没专门分支、又不在"只进日志"名单里，说明这一层没对它表过
    # 态，它就静默消失了。测试里会先红，这是运行时那道网。
    if t in ("note", "section", "block") and k and k not in _ROLE_BY_KIND \
            and k not in _SPECIAL_KINDS and k not in _LOG_ONLY_KINDS:
        if k not in _WARNED_KIND:
            _WARNED_KIND.add(k)
            _emit("plat", f"⚠️ 收到**没表过态**的 kind={k!r}（type={t}）—— 这一条没上屏。"
                          "去 `chat._ROLE_BY_KIND` / `_SPECIAL_KINDS` / `_LOG_ONLY_KINDS` 里表态。",
                  stamp=ev.get("ts"))


def _turn_summary(snap: dict, elapsed: float) -> None:
    """回合小结：任务台账和主上下文读数。

    这一轮的话、工具调用、试答、投递已由事件订阅按真实时序播过，这里只补事件里没有的。
    主上下文大小是上下文隔离的唯一现场读数：跟着产物一起涨，就是产物正文漏进来了。
    """
    ctx = snap.get("main_ctx_chars") or 0
    sys_chars = snap.get("sys_prompt_chars") or 0
    body = snap.get("artifacts_chars") or 0
    warn = ctx_warn(ctx + sys_chars)
    _emit(
        "ctx",
        f"主上下文 {ctx + sys_chars} 字（消息 {ctx} + system {sys_chars}）"
        f"　｜　产物库正文 {body} 字　｜　本轮耗时 {elapsed:.0f}s"
        + (f"\n{warn}" if warn else ""),
    )

    tasks = snap.get("tasks") or []
    if not tasks:
        return
    rows = []
    for t in tasks:
        mark = {
            "completed": "✅", "waiting_for_input": "❓", "failed": "❌",
            "running": "⏳", "superseded": "🔁",
        }.get(t["stage"], "?")
        rows.append(
            f"{mark} {t['id']}  {t['skill']:<20} 产物={t['artifacts'] or '（无）'}"
            + (f"  引用={t['refs']}" if t.get("refs") else "")
        )
    _emit("task", "\n".join(rows))


def _read_card_answer(color: str) -> str | None:
    """在待答卡片下读一条输入，返回答复原文；`None` = 跳过（显式词 / Ctrl-C / EOF）。

    空行不作数：答复通道和 `USER` 是两个提示符、共用同一条 stdin，缓冲里一个杂换行就足以
    把答复挤到 `USER`，变成新消息，旧任务永远挂着。
    """
    hinted = False
    while True:
        try:
            ans = input(paint(_gutter_prompt("reply"), color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if ans.lower() in _SKIP_WORDS:
            return None
        if ans:
            return ans
        if not hinted:
            print(paint("（空行不作数 —— 答复就直接打字；跳过请打 skip，回 👤 说别的）", color))
            hinted = True


@dataclass
class _Session:
    """一次交互式会话的运行态，只在 chat 内部传。"""

    tag: str
    out_dir: Path
    graph: object
    config: dict
    #: `install()` 的返回（counter / writes / run_tasks / fault…），收尾写进 results.json。
    inst: dict
    skills_dir: str
    t0: float
    #: 跑成了的轮（`results.json` 的 `turns`）。
    turns: list[dict] = field(default_factory=list)
    #: 没跑成的轮（平台故障）。单独记：`turns` 里只放跑成了的轮，但"失败过"这件事也得进
    #: 记录，不然 results.json 看着像一切正常。
    turn_errors: list[dict] = field(default_factory=list)
    #: 主 agent 上下文读到第几条了，每轮比上一轮多出来的就是这一轮的增量。
    seen_len: int = 0
    #: 会话级的异常（按轮兜住的不算，见 `_one_turn`）。
    error: str | None = None


def _parse_args() -> argparse.Namespace:
    """命令行参数：技能目录、会话标识、执行路径。"""
    ap = argparse.ArgumentParser(description="交互式对话（你当用户；技能从文件夹加载）")
    ap.add_argument(
        "--skills",
        default=str(DEFAULT_SKILLS_DIR),
        help=f"技能包目录（默认 {DEFAULT_SKILLS_DIR}）：**每个子目录是一个技能包**，"
        "入口是 SKILL.md（frontmatter 给 name/description，正文即 instructions），"
        "包里的文本文件（references/ assets/ …随作者）按需读（渐进式披露第三层，read_skill_file）",
    )
    ap.add_argument("--thread", default="interactive-1")
    ap.add_argument(
        "--exec-path", choices=["auto", "delegate", "inline"], default="auto",
        help="执行路径：auto 两条都开（默认，产品行为）｜ delegate **强制委派**（主 agent 拿不到"
             "自己干的工具）｜ inline 强制主 agent 自己干（不给 delegate）。"
             "⚠️ 想验「执行器那条链路」时用它，别靠碰运气 —— 见 sxagent/agent.EXEC_PATH",
    )
    return ap.parse_args()


def _banner(s: _Session, seed_note: str) -> None:
    """开场白：标题、模型、技能清单、日志与记录路径，以及首次运行那句"放了 N 个示例技能"。

    记录路径要留着 —— 那个文件夹正是用户打包后要发回来的东西。
    """
    print(BAR)
    # 模型只显示名字：端点 URL 在 .env 里，打在这里是噪音。
    print(f"{APP_TITLE} ｜ 模型 {describe().split(' @ ')[0]}")
    # 标题与技能清单之间也隔一条线，扫一眼能分出"这是谁"和"有什么"。
    print(BAR)
    print("技能清单")
    _print_skills(registry.values(), indent="   ")
    print(BAR)
    print(f"日志   {recorder.log_file()}")
    print(f"记录   {s.out_dir}")
    print(BAR)
    print("直接说你要什么。退出：Ctrl-C 或 exit。\n")
    if seed_note:
        # 只在打包后首次运行有（开发态目录本来就在）—— 用户得知道技能从哪来。
        print(f"  {seed_note} —— 你也可以自己往 skills 文件夹里加。\n")


def _absorb(s: _Session, state: dict, text: str, t: float, *, answer_to: str = "") -> None:
    """把这一轮收进记录：快照、计数、摘要。

    `answer_to`：这一句是在答哪一条待答（主循环那条为空）。
    """
    snap = snapshot(state, text, s.seen_len, len(s.turns) + 1, round(time.time() - t, 1))
    snap["input_source"] = "human"  # 用户当场打的，不是脚本预置
    if answer_to:
        snap["answer_to"] = answer_to   # 记录层如实标这句在答哪一条
    s.turns.append(snap)
    s.seen_len = len(state.get("messages") or [])
    _turn_summary(snap, time.time() - t)
    print()


def _one_turn(s: _Session, text: str, *, answer_to: str = "") -> dict | None:
    """跑一轮，返回新的 `state`；这一轮没跑成就记进 `turn_errors` 并返回 `None`。

    一轮没跑成不等于结束会话：说清原因，提示符接着来。`graph.ainvoke` 是阻塞的（一轮 2–3
    分钟），但过程不黑屏，事件同步到达 `_on_event`。
    """
    import traceback

    payload: dict = {"messages": [HumanMessage(text)], "log": []}
    if answer_to:
        payload["answer_to"] = answer_to
    try:
        return aio.run(s.graph.ainvoke(payload, s.config))   # 同步宿主驱动协程
    except Exception as e:  # noqa: BLE001
        s.turn_errors.append({
            "turn": len(s.turns) + 1, "input": text,
            "error": f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
        })
        print(f"\n⚠️ **这一轮没跑成**（平台故障）：{type(e).__name__}: {e}")
        print("   会话还在 —— 把刚才那句**再说一遍**就行。\n")
        return None


def _ask_loop(s: _Session, state: dict) -> None:
    """待答子循环：一次摆一个。

    这个提示符就是通道：在它下面打的字属于那一条请求。摆哪一条由 `graph.next_pending`
    定，和投递认的是同一条规则。
    """
    while (p := next_pending(state)) is not None:
        icon, color = _STYLE["ask"]
        # "取哪句、取不到怎么办"只有 `asked_text` 一处定义，排版归这里。
        q = asked_text(p)
        # 走 `_emit` 不走裸 print：裸 print 的长问题会被终端折到 0 列，续行和段头分不清。
        # 正文缩进 2 格：段头独占一行、正文成一块。
        print()
        _emit("ask", f"执行器在问 · {p.get('skill')}\n"
                     + "\n".join("  " + ln for ln in (q or "").splitlines() or [""]))
        _emit("task", "打字即答复；跳过打 skip，回 👤 说别的")
        # 空行不作数、跳过要明说，见 `_read_card_answer`。
        ans = _read_card_answer(color)
        if ans is None:  # 显式跳过 / Ctrl-C / EOF → 回到用户通道
            break
        t2 = time.time()
        state = _one_turn(s, ans, answer_to=p["id"])
        if state is None:   # 投递那一轮也没跑成，同主循环
            break
        _absorb(s, state, ans, t2, answer_to=p["id"])


def _session(s: _Session) -> None:
    """交互主循环：读一句 → 跑一轮 → 摆待答。"""
    # 订阅事件总线：`graph.invoke` 跑在主线程，业务代码每发一个事件就同步收到，实时上屏。
    trace.subscribe(_on_event)
    try:
        while True:
            try:
                # 提示符本身就是那行 gutter，打的字接在后面，天然是一行；不要先 input
                # 再播报，那样同一句话会出现两次（终端回显一次、播报一次）。
                color = _STYLE["user"][1]
                prompt = _gutter_prompt("user")
                text = input(paint(prompt, color)).strip()
                _to_file(prompt + text)
                # 管道输入没有终端回显，补上；真终端上补了会重复。空行也要打（只换行），
                # 不然下一个提示符会接在同一行上。
                if not sys.stdout.isatty():
                    print(text)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            # 空行不退出：粘贴的内容带尾随换行，那个换行留在缓冲里，下一个提示符一出现就
            # 被当成空行，一次多余的回车就能结束整场会话。退出是显式动作：Ctrl-C/D 或 exit。
            if not text:
                continue
            if text.lower() in _EXIT_WORDS:
                break
            t1 = time.time()
            state = _one_turn(s, text)
            if state is None:
                continue
            _absorb(s, state, text, t1)
            _ask_loop(s, state)
    except Exception as e:  # noqa: BLE001
        import traceback

        s.error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        cut = len(s.error) - 500
        print(f"\n⚠️ **会话中断**（平台故障）：{s.error[:500]}"
              + (f"\n（**平台截断：还有 {cut} 字未显示**，全文见 {s.out_dir / 'results.json'}）"
                 if cut > 0 else ""))


def _finish(s: _Session) -> None:
    """收尾：写 `results.json`、三份记录，然后说再见。

    收尾才说记录在哪、请对方发回来才自然，开场就说只是噪音。
    """
    result = {
        "case": s.tag,
        "rep": 1,
        "turns": s.turns,
        "turns_done": len(s.turns),
        "turn_errors": s.turn_errors,
        "usage": s.inst["counter"].snapshot(),
        "artifact_writes": s.inst["writes"],
        "run_tasks": s.inst["run_tasks"],
        "artifact_ids": artifacts_mod.ids(),
        "artifact_final": {
            k: {
                "title": v["title"],
                "summary": v.get("summary", ""),
                #: 产出方声明的类型（`md` / `html`），断言按它判"哪些算网页"，不嗅探内容。
                "type": v.get("type", ""),
                "ext": v.get("ext", ""),
                "chars": len(v["content"]),
                "content": v["content"],
            }
            for k, v in artifacts_mod.items()
        },
        "fault": s.inst["fault"],
        "error": s.error,
        "wall_s": round(time.time() - s.t0, 1),
        "trace_log": str(recorder.log_file()),
        "skills_dir": s.skills_dir,
    }
    dump_artifacts(s.out_dir, s.tag, 1, result)
    dump_run_record(s.out_dir, s.tag, 1, result)
    # 把这次跑的原始 trace 作为附录并进 `_run.md`（跑批那条由入口追加，交互式自己做）
    append_trace(result, s.tag)
    if _SINK is not None:
        _SINK.close()
    (s.out_dir / "results.json").write_text(
        json.dumps([result], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recorder.close()
    # 退出要说一声，不然用户分不清是正常结束还是崩了。
    print(f"\n  再见 —— 这次对话已记录，请把这个文件夹发给我：\n    {s.out_dir}\n")


def main() -> None:
    """交互式入口：装配 → 开场白 → 会话 → 收尾。这里只做编排。"""
    args = _parse_args()

    # 首次运行（只在打包后）：建 skills/ 和 runs/，空的话放一遍示例技能。只有"放了技能"
    # 值得告诉用户，建目录是内部事务；数字由 `ensure_app_dirs` 报，别写死。
    created = ensure_app_dirs()
    seed_note = next((c for c in created if "示例技能" in c), "")

    # 技能来自文件夹，启动时全部注入；case 那套（预置 turns + 断言）是跑批用的。
    inst = install(
        {"skills_dir": args.skills, "skills_mode": "replace", "exec_path": args.exec_path}
    )
    graph = build()
    config = {"configurable": {"thread_id": args.thread}}
    session.use(args.thread)      # 进程就是一次会话：thread_id 进事件、日志带标识

    label = Path(args.skills).name or "skills"
    tag = f"interactive-{label}-{time.strftime('%m%d-%H%M%S')}"
    # 记录落 `runtime.RUNS_DIR`（程序旁边 / 仓库顶层的 runs/），用户找得到，也方便整包发回来。
    out_dir = RUNS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # CLI 输出原样落一份：终端里滚过去的，文件里也留得住。
    global _SINK
    _SINK = (out_dir / "_console.log").open("w", encoding="utf-8")

    # 产物当场落盘：先告诉产物库这次 run 的产物放哪，`store_artifact` 写完就落盘，
    # `write_artifact` 的返回里就能给出真实路径。落点必须和最后 `dump_artifacts`
    # 的目录一致，否则前后两份文件对不上。走函数而不是直接赋值，写错模块会当场报错。
    artifacts_mod.use_dir(out_dir / "artifacts" / f"{tag}__r1")

    s = _Session(tag=tag, out_dir=out_dir, graph=graph, config=config, inst=inst,
                 skills_dir=str(args.skills), t0=time.time())
    _banner(s, seed_note)
    _session(s)
    _finish(s)


if __name__ == "__main__":
    main()
