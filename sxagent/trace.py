"""事件总线：业务代码只调这一层，事件是记录层和展示层之间唯一的契约。

    业务代码 ──trace.*()──▶ _publish() ──┬──▶ 记录器：落日志文件
                                        └──▶ 展示层：自己排版上屏

发射器只发结构化事件、不写文件。本模块不认识任何领域词汇。

## 不变量

1. 两种标签都是受控词汇，打错一个字母当场报错：`type` 是事件形状（八种，见 `_EVENT_TYPES`），
   `kind` 是业务语义（见 `KINDS`），展示层靠 `kind` 归角色，不靠子串匹配认行。
2. 总线认识形状，不认识词表。`decision` 是形状，`result_by` 能取哪些值归 `handoff.decision`；
   本模块不许 import `sxagent.handoff.*`。
3. 记录器不是订阅者：`_publish` 先调它、再送订阅者。展示层坏了最多看不见，记录层丢一条就是
   权威记录破洞，所以它单独兜、兜住后把洞标出来。它是槽位，本模块不 import 它。
4. 日志是原始记录：不截断、不收回已经流出去的字。嫌吵就在播报层砍。
5. `dump` 带的是消息对象本身，`tool_call_id` 就在里面。配对与来源标注全走结构。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from sxagent import session

if TYPE_CHECKING:  # 只给标注用 —— 运行时不 import
    from langchain_core.messages import BaseMessage

#: 记录器槽：由装配处装（`recorder.py` 在 import 时自注册）。不装就没有日志文件，那是库的用法。
_sink_record: session.Sink | None = None
_sink_write: Callable[[str], None] | None = None


def install_recorder(*, record: session.Sink, write: Callable[[str], None]) -> None:
    """装上记录器——总线不认识它，只认这两个函数。

    `record(ev)` 把事件渲染成日志行；`write(line)` 写一行原文（标洞和"订阅者出错"用）。
    做成槽位而不是 import，是为了让总线不依赖领域词汇表。调用方是 `recorder.py` 自己。
    """
    global _sink_record, _sink_write
    _sink_record, _sink_write = record, write


def stamp() -> datetime:
    """这一条事件的时间戳 —— 带的是结构（`datetime`），不是渲染好的字符串。

    显示成什么样由各视图自己渲染（`fmt_ts` / `fmt_clock`）：视图要是去切字符串，改一次格式
    就静默切错，而且没人知道它切错了。
    """
    return datetime.now()


def fmt_ts(t: datetime) -> str:
    """日志文件里的时间戳：年月日 + 毫秒。

    一轮里同一秒发生好几件事（工具调用 → 返回 → 判定），秒级看不出先后；跨天也分不清日子。
    """
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def fmt_clock(t: datetime) -> str:
    """终端 gutter 里的时间：只有时分秒 + 毫秒 —— 日期在屏幕上没用。"""
    return t.strftime("%H:%M:%S.%f")[:-3]


# ──────────────────────────── 事件总线 ────────────────────────────

#: 事件的八种形状。每种带什么字段，见各 emitter 与 `recorder.py`。
_EVENT_TYPES = (
    "section",
    "note",
    "block",
    "stream",
    "dump",
    "decision",
    "handoff",
    "cut",
)


class _Subscription:
    """`subscribe()` 的返回，拿它退订。

    不直接返回 `fn`：订阅者常是绑定方法，每次取属性都是新对象，`list.remove` 找不到。
    """

    __slots__ = ("_owner", "_slot")

    def __init__(self, owner: list[list[session.Sink]], slot: list[session.Sink]) -> None:
        self._owner, self._slot = owner, slot

    def cancel(self) -> None:
        """退订。重复调是安全的（已经不在里面就什么都不做）。"""
        try:
            self._owner.remove(self._slot)
        except ValueError:
            pass


def subscribe(fn: session.Sink) -> _Subscription:
    """订阅当前会话的事件总线，事件按发布顺序同步送达。

    订阅者抛异常不会影响业务：兜住并记进日志，展示层坏了最多是看不见。订阅挂在会话上而不是
    模块上，多会话共享进程时 A 的展示层收不到 B 的事件。
    """
    slot = [fn]
    session.current().subs.append(slot)
    return _Subscription(session.current().subs, slot)


def _publish(ev: dict[str, object]) -> None:
    """发一个事件：先记录器（唯一落盘者），再同步送当前会话的订阅者。

    记录器不走订阅者那张网：展示层坏了只是看不见，记录层丢一条却是权威记录破了洞，
    所以它单独兜，兜住之后把洞标出来。`thread_id` 在这一层加，八个形状共用一处。
    """
    if ev.get("type") not in _EVENT_TYPES:
        # 形状只有那八种：打错一个字母，记录器与展示层都会静默跳过它 —— 宁可当场报错。
        raise ValueError(f"未知事件形状 type={ev.get('type')!r}：{list(_EVENT_TYPES)}")
    # 会话标识（多会话共享进程时，这是"这条事件属于谁"的唯一来源）。
    # 默认会话的 thread_id 是空串 —— 单会话宿主（CLI / 跑批）行为不变。
    ev["thread_id"] = session.current().thread_id
    # 记录器是槽位（不是 import、不是订阅者）—— 没装就什么都不写（那是库的用法）。
    if _sink_record is not None:
        try:
            _sink_record(ev)
        except Exception as e:  # noqa: BLE001 —— 不影响业务，但必须把洞标出来，见 _mark_hole
            _mark_hole(ev, e)
    for slot in list(session.current().subs):
        fn = slot[0]
        try:
            fn(ev)
        except Exception as e:  # noqa: BLE001 —— 见 subscribe 的 docstring
            # 没装记录器时这一行没地方写 —— 那也不能影响业务。
            if _sink_write is not None:
                _sink_write(
                    f"[{fmt_ts(stamp())}] ⚠️ 事件订阅者出错"
                    f"（{getattr(fn, '__qualname__', fn)}）：{type(e).__name__}: {e}"
                )


def _mark_hole(ev: dict[str, object], e: Exception) -> None:
    """记录器渲染失败 → 在日志里把洞标出来。标洞本身也兜住，但不许静默。"""
    if _sink_write is None:  # 没装记录器 —— 连"洞"都无处可标（库的用法）
        return
    try:
        _sink_write(
            f"[{fmt_ts(stamp())}] ⚠️ **这一条没有落进日志**（记录器渲染失败）："
            f"type={ev.get('type')!r} kind={ev.get('kind')!r}"
            f" —— {type(e).__name__}: {e}"
        )
    except Exception:  # noqa: BLE001 —— 连标洞都写不进去，只能放弃；仍不影响业务
        pass


# ──────────────────────────── 事件发射器（业务代码只调这一层） ────────────────────────────

#: 角色枚举。展示层按它归版式，不按 `who` 的措辞猜 —— 措辞一改，展示层就静默降级。
AGENT = "agent"
EXECUTOR = "executor"
ROLES = frozenset({AGENT, EXECUTOR})


def check_role(role: object, where: str) -> str:
    """角色只能是枚举值 —— 传错就当场报错，不许静默当成主 agent。"""
    if role not in ROLES:
        raise ValueError(
            f"{where} 的 role 必须是 {sorted(ROLES)} 之一，收到 {role!r}。"
            "（角色是逻辑，不是措辞：见 trace.AGENT / trace.EXECUTOR）"
        )
    return role


#: kind 的受控词汇，展示层按它归角色。加一个 kind 要登记在这里，并去 `app/chat.py` 表态
#: （上屏 / 有专门分支 / 只进日志）；漏表态，那一行就静默消失。
KINDS = frozenset(
    {
        # agent.delegate
        "dispatch",
        "supersede",
        "waiting",
        "delegate_failed",
        # executor.run_task
        "run_begin",
        "run_end",
        "executor_briefing",
        "nudge",
        # executor 的工具回调（`executor_tool` = 执行器调了一次工具：起 / 止各一条）
        "executor_tool",
        # executor 的模型回调（`thinking` = 执行器开始了一次模型调用：第一个字出来前的静默段）
        "thinking",
        # graph.prepare / finalize
        "feed",
        "relay",
        # llm.safe_structured（`llm_io` = 那次调用的 PROMPT / RESPONSE / 失败提示）
        "ask_main",
        "llm_io",
        # stream.stream_agent（`retry` = 抖动已重试一次；`writing` = 工具参数在流，报"已收到 N 字"）
        "retry",
        "writing",
        # artifacts.store_artifact（`artifact` = 产物落盘了：ID、标题、字数、路径）
        "artifact",
        # graph.node_agent（`stream_empty` = 流式一步都没输出，退回 `invoke` 兜底：这一趟没有流式日志）
        "stream_empty",
        # executor.run_task（`run_fault` = 重试仍失败；`ask` / `delivered` = 这一轮结束的结构事实）
        "run_fault",
        "ask",
        "delivered",
    }
)


def _check_kind(kind: str, where: str) -> str:
    """kind 只能是受控词汇里的值（空串 = 纯记录、不上屏）—— 打错就当场报错。"""
    if kind and kind not in KINDS:
        raise ValueError(
            f"{where} 的 kind={kind!r} 不在受控词汇里（{sorted(KINDS)}）。"
            "加新 kind 要同时：① 登记进 trace.KINDS；"
            "② 在 app/chat.py 表态（上屏 / 有专门分支 / 只进日志）。"
        )
    return kind


def section(title: str, kind: str = "") -> None:
    """一个段落头（前后两道 `═══` 包着标题）。

    `kind` 是业务语义标签（受控词汇，见 `KINDS`），展示层按它归角色；不给就是纯记录、不上屏。
    """
    _publish(
        {"type": "section", "kind": _check_kind(kind, "section"), "title": title, "ts": stamp()}
    )


def note(
    text: str, kind: str = "", *, fields: Mapping[str, object] | None = None
) -> None:
    """一条记录（带时间戳）。只发内容，日志文件里的前缀（如 `finalize: `）归记录器。

    `text` 通常是一行；透传那条是例外 —— 它带的就是待答原文，本来就可能多行。
    `fields` 是给订阅者的结构化补充（如故障分类）：记录器只渲染 `text`、不看它；
    订阅者靠它做判断，不用去解析文本。
    """
    ev: dict[str, object] = {
        "type": "note",
        "kind": _check_kind(kind, "note"),
        "text": text,
        "ts": stamp(),
    }
    if fields:
        ev.update(fields)
    _publish(ev)


def block(label: str, text: str, *, kind: str = "", indent: str = "  ") -> None:
    """一个带标签的多行文本块（prompt / response / 交接正文）。"""
    _publish(
        {
            "type": "block",
            "kind": _check_kind(kind, "block"),
            "label": label,
            "text": text,
            "indent": indent,
            "ts": stamp(),
        }
    )


def handoff(
    hop: str,
    frm: str,
    to: str,
    text: str,
    *,
    structured: str = "",
    full: bool = False,
) -> None:
    """一次交接的完整记录：一行摘要（加可选全文）。

    摘要让复盘的人一眼看到：谁给谁、交了多少字、有没有结构化对象、跳名与方向。截断由切的
    那一方自己报，不在这里事后解析文本。`full=True` 只在别处没 dump 过这一跳时用。
    """
    _publish(
        {
            "type": "handoff",
            "kind": "",
            "hop": hop,
            "frm": frm,
            "to": to,
            "text": text,
            "structured": structured,
            "full": full,
            "ts": stamp(),
        }
    )


def emit(ev: dict[str, object]) -> None:
    """发一个已经是最终形状的事件，给领域包自己的发射器用。

    领域包（如 `handoff.decision`）发事件走这一条，总线就不必 import 领域包。校验一样跑：
    `type` / `kind` 都在 `_publish` 里查。现成的发射器能用就用，别拿它绕过。
    """
    _publish(ev)


def cut(where: str, kept: int, total: int, *, see: str = "") -> None:
    """切了一刀就报一刀，由切的那一方自己调，不靠事后从文本里解析。

    标记在文本里（读的人看得见），这一行在日志里。`where` 是切的哪一份，`kept` / `total`
    是实测算数，`see` 补一句全文在哪。
    """
    _publish(
        {
            "type": "cut",
            "kind": "",
            "where": where,
            "kept": kept,
            "total": total,
            "see": see,
            "ts": stamp(),
        }
    )


def dump_trace(tag: str, messages: list[BaseMessage], *, role: str) -> None:
    """把一整条消息轨迹作为事件发出去，事件里带的是消息对象本身。

    展示层靠 `tool_call_id` 配对、靠 `additional_kwargs` 认注入，不反推结构。`role` 必填。
    """
    check_role(role, "dump_trace")
    _publish(
        {
            "type": "dump",
            "kind": "",
            "role": role,
            "tag": tag,
            "messages": list(messages),
            "ts": stamp(),
        }
    )


# ──────── 消息渲染（记录器 / 终端 / 台账 / 跑批记录共用） ────────
#
# 放总线这边是因为它不认识领域词汇；搬去 `recorder.py` 的话，展示层就得反过来依赖记录器。

#: 正文类参数只留字数：`write_artifact(content=...)` 的正文另有出处，灌进日志纯属浪费。
#: 其余参数照原样给全 —— 判据同显示层：全文另有出处才省，没有就全给。
BODY_ARGS = frozenset({"content"})


def brief_args(args: object, limit: int = 120) -> str:
    """工具参数的一行摘要 —— 只有 `BODY_ARGS` 里的正文类参数会被压成字数。"""
    if not isinstance(args, dict):
        return str(args)
    parts = []
    for k, v in args.items():
        text = str(v)
        parts.append(
            f"{k}=<{len(text)}字>" if k in BODY_ARGS and len(text) > limit else f"{k}={text!r}"
        )
    return ", ".join(parts)


def tool_calls_of(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    """从消息里抽出全部工具调用 —— `(名字, 参数摘要)`，按发生顺序。

    主 agent 的工具调用不进 dump 轨迹，日志要记全就得从消息里抽一遍。
    """
    out: list[tuple[str, str]] = []
    for m in messages:
        for c in getattr(m, "tool_calls", None) or []:
            out.append((str(c.get("name")), brief_args(c.get("args") or {})))
    return out


def render_message(m: BaseMessage) -> str:
    """把一条 LangChain 消息渲染成可读文本（记录器落盘用）。这里不截断。

    日志是原始记录，"平台注入了什么、agent 说了什么"是优化的唯一观察窗口。AI 的话只写
    头部，正文已由 `stream_agent` 逐行流过。
    """
    kind = type(m).__name__
    content = str(m.content)
    parts = [f"{kind}({len(content)}字)"]
    # 平台注入的标记（`handoff.INJECTED`）—— 透传注入的是 AIMessage，内容是执行器的
    # 原话；不写出来，读日志的人会以为那是主 agent 说的。
    tag = (getattr(m, "additional_kwargs", None) or {}).get("injected_by")
    if tag:
        parts.append(f"    injected_by={tag}")
    if content.strip() and kind != "AIMessage":
        # `!r` 只为压成一行（日志一行一条），不是为了截断。
        parts.append(f"    content={content!r}")
    for c in getattr(m, "tool_calls", None) or []:
        parts.append(f"    tool_call={c.get('name')} args={brief_args(c.get('args') or {})}")
    return "\n".join(parts)
