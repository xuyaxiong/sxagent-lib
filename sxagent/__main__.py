"""内置技能入口，装配最简的那种。

    uv run python -m sxagent                # 交互式
    uv run python -m sxagent --say "你好"    # 单轮

验证用的 CLI 是 `app/chat.py`，打包打的是它。这里只装内置技能、显示层最简，用来快速看机制。
显示层订阅事件总线（`_on_event`）自己排版，日志落在 `recorder.LOG_FILE`。

## 不变量

1. 显示层只认事件：谁说的由 `role` / `kind` 带着，不读 `state`，不从渲染文本反推。
2. 不建第二张 kind 到角色的表（那是 `chat._ROLE_BY_KIND` 的表态）。这一层按形状渲染：认得的
   形状各给一行，认不出的一律照原样打出来——排版丑一点没关系，静默丢才要命。
3. 切了必须标：块最多印 `_BLOCK_MAX_LINES` 行，超出写明还剩多少行、全文在哪。
4. 一轮没跑成不等于结束会话：按轮兜住，说清原因让调用方接着问下一句。
5. 待答的标识由通道填：答复提示符就是"通道"，在它下面打的字带 `answer_to` 投回去，一次摆
   一个。`--say` 下没给 `--answer` 就把问题打出来停住，不替用户编答案。"""

from __future__ import annotations

import argparse
import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from sxagent import aio
from sxagent import recorder, session, trace
from sxagent.config import describe
from sxagent.graph import build, next_pending
from sxagent.handoff.briefing import asked_text
from sxagent import artifacts as artifacts_mod

BAR = "─" * 74

#: 块最多印几行（切了必须标；全文永远在日志文件里）
_BLOCK_MAX_LINES = 12


# ──────────────────────────── 显示层：事件 → 屏幕 ────────────────────────────

#: 说话的是谁 —— 按 `role` 枚举取（逻辑），不从 `who` 的措辞里猜。
_ROLE_TAG = {trace.AGENT: "🤖 主 agent", trace.EXECUTOR: "🔧 执行器"}

_last_stream_tag = ""      # 同一角色连着输出时，前缀只打一次
_tool_names: dict[tuple[str, str], str] = {}   # (role, tool_call_id) → 工具名


def _bullets(v: object) -> str:
    """列表 → 逐项 `- x`；字符串原样。空的显式写「（无）」（"没有"也要能审计）。

    这里不截断 —— 四格是一次判断的凭据，看全才有意义。长的是 `given`/`result` 里的自由
    文本，那是它本来该有的样子。
    """
    if isinstance(v, (list, tuple)):
        return "\n".join(f"- {x}" for x in v) if v else "（无）"
    return str(v) or "（无）"


def _capped(text: str) -> str:
    """多行文本截行并显式说明（切了必须标，绝不静默截）。"""
    lines = str(text).splitlines() or [""]
    if len(lines) <= _BLOCK_MAX_LINES:
        return "\n".join(lines)
    keep = lines[:_BLOCK_MAX_LINES]
    return "\n".join([*keep, f"…（还有 {len(lines) - _BLOCK_MAX_LINES} 行，全文见日志文件）"])


def _on_event(ev: dict) -> None:
    """事件总线的订阅入口 —— 展示层从这里拿结构化事实，自己排版上屏。"""
    global _last_stream_tag
    t = ev.get("type")
    if t != "stream":
        _last_stream_tag = ""   # 换个形状就把流式前缀重置（下次输出重新标谁在说）

    if t == "stream":
        tag = _ROLE_TAG.get(ev.get("role") or "", f"· {ev.get('who') or '平台'}")
        if tag != _last_stream_tag:
            print(f"\n{tag}：")
            _last_stream_tag = tag
        print(f"  {ev['text']}")
    elif t == "section":
        print(f"\n{'═' * 3} {ev['title']}")
    elif t == "note":
        if ev.get("kind") == "relay":
            # 平台透传：这句是执行器的原话，平台只是转达 —— 归属由 kind 给，不读文本。
            # 原文照上屏、不切前缀：日志文件那个 `finalize: ` 是记录层的格式，切它会随文案漂。
            print(f"\n📮 平台 → 用户：{ev['text']}")
        else:
            print(f"   │ {ev['text']}")
    elif t == "block":
        print(f"   │ {ev['label']}")
        for ln in _capped(ev["text"]).splitlines():
            print(f"   │   {ln}")
    elif t == "decision":
        # 四格从对象渲染（前提 / 问题 / 结果 / 下一步）—— 不从渲染文本反推
        print(f"   │ 🧭 {ev['who']}｜问题：{ev['q']}")
        print(f"   │   前提：{_bullets(ev['given'])}")
        print(f"   │   结果：{_bullets(ev['result'])}")
        print(f"   │   下一步：{ev['next']}")
    elif t == "cut":
        print(f"   │ ✂️ 截断 · {ev['where']}：保留 {ev['kept']} 字 / 原文 {ev['total']} 字"
              + (f"｜{ev['see']}" if ev.get("see") else ""))
    elif t == "dump":
        _on_dump(ev)
    # `handoff`（交接记录）只进日志文件、不上屏 —— 同 `chat.py` 的契约：它的内容上面那些
    # 形状已经各自播过一遍，这里再播就是同一个事实两副面孔。


def _on_dump(ev: dict) -> None:
    """消息轨迹 —— 消息对象本身就在事件里，归属按 `ev["role"]` 取。"""
    role = ev.get("role") or ""
    who = _ROLE_TAG.get(role, f"· {role or '未知角色'}")
    for m in ev.get("messages") or []:
        for c in getattr(m, "tool_calls", None) or []:
            if c.get("id"):
                _tool_names[(role, str(c["id"]))] = str(c.get("name"))
            print(f"   │ 🧠 {who} → {c.get('name')}（{trace.brief_args(c.get('args') or {})}）")
        kind = type(m).__name__
        if kind == "AIMessage":
            continue        # 它说的话已由流式实时播过（同 `chat._on_dump` 的契约）
        content = str(m.content)
        if kind == "ToolMessage":
            name = _tool_names.get((role, str(getattr(m, "tool_call_id", "") or "")), "?")
            print(f"   │ ↩ {name} 的返回（{len(content)} 字）：\n{_capped(content)}")
        elif kind == "SystemMessage":
            # 平台每轮注入的台账 —— 不截断（它是"平台塞了什么"的观察窗口）
            print(f"   │ 📌 平台 → 主 agent（注入的台账）：\n{content}")


# ──────────────────────────── 快照读数 ────────────────────────────


def _ctx_chars(messages: list) -> int:
    total = 0
    for m in messages:
        total += len(str(m.content))
        for c in getattr(m, "tool_calls", None) or []:
            total += len(str(c.get("args", "")))
    return total


def _dump(result: dict) -> None:
    """回合末的读数（Task 表 ＋ 上下文体积）—— 事件里没有的东西，从快照读。"""
    tasks = result.get("tasks") or []
    print(f"   │ ── Task（只记委派出去的工作，共 {len(tasks)} 个）──")
    for t in tasks:
        print(
            f"   │ {t['id']} skill={t['skill']:<16} stage={t['stage']:<14} "
            f"产物={t['artifacts'] or '（无）'}"
        )
        # "试答过没有"从 `qa` 推：`by=主 agent` 的那些就是它补的
        if any(x.get("by") == "主 agent" for x in (t.get("qa") or [])):
            print(f"   │     主 agent 试答过；执行器说={t['report']!r}")
    ctx = _ctx_chars(result.get("messages") or [])
    body = artifacts_mod.chars()
    print(f"   │ ★ 主 agent 上下文 = {ctx} 字 ｜ 产物库正文 = {body} 字")
    if body and ctx:
        print(f"   │ ★ 隔离效果：产物正文是主上下文的 {body / ctx:.1f} 倍，全在产物库里")


def _parse_args() -> argparse.Namespace:
    """命令行参数 —— 会话标识 / `--say`（非交互跑一句）/ `--answer`（自动应答）。"""
    ap = argparse.ArgumentParser(description="主 agent（完整 agent）＋ 技能 / Task / 执行器")
    ap.add_argument("--thread", default="sxagent-1")
    ap.add_argument("--say", help="非交互：只跑这一句")
    ap.add_argument("--answer", default="",
                    help="--say 模式下自动应答（不给就把问题打出来停住，不替用户编答案）")
    return ap.parse_args()


def _turn(graph, config, text: str, *, answer_to: str = "") -> dict | None:
    """跑一轮 —— 没跑成就按轮兜住、说清原因、返回 `None`（会话不死）。

    `answer_to`：这条消息是在回答哪个待答任务 —— 由通道填（下面那个答复提示符），
    平台不猜。
    """
    init: dict = {"messages": [HumanMessage(text)], "log": []}
    if answer_to:
        init["answer_to"] = answer_to
    try:
        # 异步驱动：LangGraph 把同步节点丢进线程跑，行为不变，但解开了"有 async 节点
        # 就不能同步驱动"那个约束。`session.current()` 在那种线程里认得出会话 ——
        # langchain 的 `run_in_executor` 会 `copy_context()`。
        return aio.run(graph.ainvoke(init, config))
    except Exception as e:  # noqa: BLE001
        print(f"\n⚠️ **这一轮没跑成**（平台故障）：{type(e).__name__}: {e}")
        print("   会话还在 —— 把刚才那句**再说一遍**就行。\n")
        return None


def _say_mode(graph, config, args: argparse.Namespace) -> None:
    """非交互：只跑 `--say` 那一句；有待答就自动补一轮。

    自动补答的上限 3 轮是防死循环：执行器每轮都可能再问一个，答完又问会一直转下去；
    三轮够覆盖「一次答一个」的真实节奏，再多就该人工介入了。
    """
    print(f"\n👤 {args.say}")
    result = _turn(graph, config, args.say)
    if result is None:
        return
    for _ in range(3):
        p = next_pending(result)
        if p is None:
            break
        print(f"\n❓【{p.get('skill')}】问：{asked_text(p)}")
        # 没给 `--answer` 就不自动应答：把问题打出来就停 —— 不替用户编一个没说过的前提。
        if not args.answer:
            print("   （没给 `--answer` —— 不自动应答；要自动答就带上 `--answer '…'`）")
            break
        print(f"   自动答复：{args.answer}")
        result = _turn(graph, config, args.answer, answer_to=p["id"])
        if result is None:
            break
    _dump(result)
    print(BAR)


def _interactive(graph, config) -> None:
    """交互循环 —— 读一句 → 跑一轮 → 摆待答。"""
    print("输入你的请求（空行退出）。试试这些：\n")
    print("  · 你好                    → 主 agent 应该**自己答**（不建 Task）")
    print("  · 帮我做 X                → 需要专业技能 → 委派给隔离执行器")
    print("  · 帮我做 X、Y，再做个 Z\n")
    while True:
        try:
            text = input("👤 ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        result = _turn(graph, config, text)
        if result is None:
            continue          # 这一轮没跑成 → 接着问下一句（会话不死）
        _dump(result)
        print(BAR)

        # 待答：一次摆一个 —— 这个提示符就是"通道"：在它下面打的字属于那一条请求
        # （带 `answer_to` 投回去）。直接回车 = 跳过 → 回到用户提示符，在那里打的字只给主
        # agent（这条待答继续挂着）。
        while (p := next_pending(result)) is not None:
            q = asked_text(p)
            print(f"\n❓【{p.get('skill')}】问：{q}")
            try:
                ans = input("↩ 答它（回车跳过，回 👤 说别的）> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not ans:
                break
            result = _turn(graph, config, ans, answer_to=p["id"])
            if result is None:
                break
            _dump(result)
            print(BAR)


def main() -> None:
    """内置技能 CLI —— 装配 → 跑（非交互 / 交互两条路），收尾关日志。

    它自己只做编排：参数、装配、两种跑法各在一个私有件里。
    """
    args = _parse_args()

    # CLI 进程就是一次会话（`use` 不开作用域 —— 包 `with` 只是白白多一层缩进）。开了它
    # `thread_id` 才进事件、日志才带会话标识。必须在 `subscribe` 之前 —— 订阅挂的是当前会话。
    session.use(args.thread)

    graph = build()
    config = {"configurable": {"thread_id": args.thread}}
    trace.subscribe(_on_event)      # 显示层从这一刻起只认事件

    print(BAR)
    print(f"主 agent = 完整 agent（自己决定：直接答 / 自己干 / 委派） · 模型={describe()}")
    print(f"日志：{recorder.log_file()}")
    print(BAR)

    if args.say:
        _say_mode(graph, config, args)
    else:
        _interactive(graph, config)
    recorder.close()
    print(f"完整日志：{recorder.log_file()}")


if __name__ == "__main__":
    main()
