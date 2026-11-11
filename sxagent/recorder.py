"""记录器：`logs/sxagent-*.log` 的唯一落盘者。它是一个视图，不是基础设施。

把总线上的每条事件渲染成日志文件的一行或一块，顺带管文件生命周期：在哪、何时建、何时关。

## 不变量

1. 它是视图：和展示层同类，都是事件流的渲染，所以依赖 `handoff.decision` 是合适的，视图
   本来就该认识领域词汇表；反过来放进总线，基础能力就传递依赖了领域词表。
2. 记录器不是订阅者：总线先调它、再送订阅者，它出错能走 `_mark_hole` 把洞标出来。
3. 日志是原始记录：不截断、不收回已经流出去的字。
4. 落盘格式不是跨层契约：`_STREAM_MARK` / `_DECISION_HEAD` / `_NOTE_HEAD` 是本模块私有的
   格式，展示层订阅事件、不读日志文件。
5. 文件惰性创建：`LOG_FILE` 只说"文件会在哪儿"，真建出来是第一次写那一刻。"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TextIO

from sxagent import session
from sxagent import trace
from sxagent.config import app_root
from sxagent.handoff.decision import render_decision  # 决策块的唯一那条渲染路

#: 日志落应用根下的 `logs/`（`config.app_root`），不按 `__file__` 算——冻结后那是
#: `_internal/`，用户不该碰。跑批和测试把 `SXAGENT_LOG_DIR` 指到临时目录，噪音就不进 `logs/`。
_log_dir = os.getenv("SXAGENT_LOG_DIR", "").strip()
LOG_DIR = Path(_log_dir) if _log_dir else app_root() / "logs"
# 配成相对路径时也按应用根落，不按 cwd。
if not LOG_DIR.is_absolute():
    LOG_DIR = app_root() / LOG_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

#: 跑批的多个子进程可能同秒启动，撞同一个文件名；用 `SXAGENT_LOG_TAG` 区分。
_TAG = os.getenv("SXAGENT_LOG_TAG", "").strip()
_SUFFIX = f"-{_TAG}" if _TAG else ""

#: 默认会话的日志文件。开了 `session()` 就各写各的，格式一个字不用改。
#: 这里只算出路径，文件在第一次写日志时才建（`_fh()`），import 不在磁盘上留空文件。
LOG_FILE = LOG_DIR / f"sxagent-{time.strftime('%Y%m%d-%H%M%S')}{_SUFFIX}.log"
session.default().log_file = LOG_FILE


def log_size() -> int:
    """当前会话的日志已经写了多少字节 —— 文件还没建 = 0。

    惰性之后"这个进程还没写过日志"是正常状态，直接 `LOG_FILE.stat()` 会
    `FileNotFoundError`。读体积的地方一律走这儿。
    """
    p = log_file()
    return p.stat().st_size if p.exists() else 0


def log_file() -> Path:
    """当前会话的日志文件 —— 第一次写日志时才建（没写过日志的会话不产生文件）。

    文件名里会话那一段要消毒：`thread_id` 由宿主给（web 那边可能就是用户 ID），直接拼进
    路径会被 `/`、`..` 带出去。非 `[A-Za-z0-9_-]` 一律换 `_`，再截断。
    """
    s = session.current()
    if s.log_file is None:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", s.thread_id)[:40] or "session"
        s.log_file = LOG_DIR / f"sxagent-{time.strftime('%Y%m%d-%H%M%S')}-{safe}{_SUFFIX}.log"
    return s.log_file


def _fh() -> TextIO:
    """当前会话打开着的日志句柄 —— 没开就开（跟着 `log_file()` 一起惰性建）。"""
    s = session.current()
    if s.log_fh is None:
        s.log_fh = log_file().open("w", encoding="utf-8")
    return s.log_fh


def write(line: str = "") -> None:
    """写一行到日志文件（不自动换行缩进，原样写）。只属于记录器，不发事件。

    它只能被总线和本模块调：别处直接调就绕过了事件总线 —— 展示层看不见那一行，
    而且是静默的。
    """
    _fh().write(line + "\n")
    _fh().flush()


def close() -> None:
    """关掉当前会话的日志句柄（没开过就什么都不做）。"""
    s = session.current()
    if s.log_fh is not None:
        s.log_fh.close()
        s.log_fh = None


BAR = "═" * 78


# ──────────────────────────── 记录器：唯一的文件落盘者 ────────────────────────────

#: 下面几个前缀是记录器自己的落盘格式，不是跨层契约。名字也私有：公开名本身就是
#: "可以被 import"的邀请。
_STREAM_MARK = "⟨生成·"

_DECISION_HEAD = "🧭 决策"

_NOTE_HEAD = {"relay": "finalize: "}

#: 决策块的段名、顺序、渲染在 `SECTION` / `HOPS["decision"]` /
#: `handoff.decision.render_decision`，记录器只管缩进和落盘。

#: 决策块缩进，日志可读性的选择，与任何别层无关。
_DEC = "    "


def record(ev: dict) -> None:
    """把事件渲染成日志文件的一行（或一块）—— 文件格式的唯一出处。

    日志是权威记录：老日志、`_run.md` 附录、人的翻阅习惯都不能断，所以格式固定。
    由总线调（`trace._publish`），不是订阅者。
    """
    ts = ev["ts"]                      # 结构（`datetime`）：嵌套事件要原样往下传
    head = f"[{trace.fmt_ts(ts)}]"     # 日志行的前缀 —— 渲染成什么样是这一层的事
    t = ev["type"]
    if t == "section":
        write()
        write(f"{BAR}\n{head} {ev['title']}\n{BAR}")
    elif t == "note":
        write(f"{head} {_NOTE_HEAD.get(ev.get('kind') or '', '')}{ev['text']}")
    elif t == "block":
        write(f"{head} {ev['label']}:")
        for line in str(ev["text"]).splitlines() or [""]:
            write(f"{ev['indent']}{line}")
    elif t == "stream":
        write(f"{head} {_STREAM_MARK}{ev['who']}⟩ {ev['text']}")
    elif t == "handoff":
        write(f"{head} 📤 交接 · {ev['hop']}（{ev['frm']} → {ev['to']}）：{len(ev['text'])} 字"
              f"｜结构化：{ev['structured'] or '（纯文本）'}")
        if ev["full"]:
            record({"type": "block", "label": f"交接全文 · {ev['hop']}",
                    "text": ev["text"], "indent": "  ", "ts": ts})
    elif t == "cut":
        write(f"{head} ✂️ 截断 · {ev['where']}：保留 {ev['kept']} 字 / 原文 {ev['total']} 字"
              + (f"｜{ev['see']}" if ev.get("see") else ""))
    elif t == "dump":
        messages = ev["messages"]
        write(f"{head} {ev['tag']} 消息轨迹（{len(messages)} 条）:")
        for i, m in enumerate(messages):
            write(f"  [{i}] {trace.render_message(m)}")
    elif t == "decision":
        # 走唯一那条渲染路：这里只加文件格式的东西（时间戳前缀、段头），
        # 排版归 `render_decision()`。
        body = "\n".join(f"{_DEC}{ln}" for ln in render_decision(ev).splitlines())
        write(f"{head} {_DECISION_HEAD} · {ev['who']}\n{body}")


# 装上 —— 总线留的是槽，谁 import 本模块谁就装上。忘了 import = 日志静默没有，
# 这条接线靠守卫盯入口、不靠记性。记录器不订阅：总线先调它、再送订阅者，
# 这样它出错能走 `_mark_hole` 把洞标出来，不会混进"订阅者出错"里。
trace.install_recorder(record=record, write=write)
