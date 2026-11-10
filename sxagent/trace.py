"""**事件总线** —— 业务代码只调这一层；事件是记录层与展示层之间的**唯一契约**。

## 这个文件是什么

**发射器**（`section` / `note` / `block` / `kv` / `handoff` / `emit` / `cut` / `dump_trace`）
只发**结构化事件**、不写文件；`_publish` 把它们送给**记录器**（落日志）与**当前会话的
订阅者**（展示层）。

```
业务代码 ──trace.*()──▶ 事件总线 _publish() ──┬──▶ 记录器（sxagent/recorder.py）：落日志文件
                                            └──▶ 展示层（app.chat 订阅）：自己排版上屏
```

⚠️ **另外两半都不在这个文件里**：**落盘**住 `sxagent/recorder.py`、
**跑图**（`stream_agent` 那 5 个）住 `sxagent/stream.py` —— 本模块**只做总线**，
**不认识任何领域词汇**，只依赖 `session`。

## 设计原则（不变量）

1. **事件的两种标签都是受控词汇** —— 打错一个字母**当场炸**，不许静默跳过
   （`check_role` / `_check_kind` / `_publish`）：
   - `type`：事件的**形状**（八种之一，见 `_EVENT_TYPES`）—— 决定它带哪些字段；
   - `kind`：业务**语义**（`dispatch` / `run_begin` / `relay` …，见 `KINDS`）——
     展示层据此决定归到哪个角色，**不再靠子串匹配认行**（措辞一改就静默失配的根）。
2. **总线认识形状，不认识词表** —— `_EVENT_TYPES` 里列着 `decision` 这个**形状**，
   但"`result_by` 能取哪些值"是**领域词汇**，归 `handoff.decision`（**发射器也在那儿**）。
   ⇒ 本模块**不许 import `sxagent.handoff.*`**。
3. **记录器不是"订阅者"** —— `_publish` 先调它、再送订阅者：展示层坏了最多是"看不见"，
   而记录层丢一条 = **权威记录破了个洞**，所以它单独兜、兜住后**把洞标出来**。
   ⚠️ 记录器是**槽位**（`install_recorder`），由装配处装 —— 本模块**不 import 它**
   （否则又回到"基础能力传递依赖领域词表"）。
4. **日志是原始记录** —— 记录层不截断、不收回已经流出去的字；要少看噪音就在**播报层**砍。
5. **`dump` 事件里带的是消息对象本身** —— `tool_call_id` 就在里面，工具配对、来源标注
   全部走结构，**不从渲染文本里反推结构**。

## 关键取舍

**① 展示层要的粒度：每个 super-step ＋ 每个字**（`stream.py` 的 `stream_agent`）。
`invoke` 跑完才返回、全程黑屏，所以拆两层给：`values` 那一支给调用方做逐步骤 dump，
`stream` 事件给记录器落盘、展示层实时上屏。

**② 重试只一次，且只在"还没走完第一步"时**（`stream.py` 的 `_stream_with_retry`）。
从头重跑 = `state` 干净、**副作用不干净**。抖在那之前 = 什么都还没干，重试是白赚；
抖在那之后 = 已经动过手，**如实抛出去**比静默重做一遍强。

**③ 确定性拒绝不重试**（`stream.py` 的 `upstream_rejected`）。4xx（除 408/429）是上游对
**这份内容**的拒绝 —— 原样重发会再被拒。

## 遗留问题

- 落在"第二步之后"的抖动，这一轮**真的失败**了 —— 更彻底的"从最后一个走完的
  super-step 接着跑"还没做（想法与代价见 `stream.py`）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from sxagent import session

#: **记录器槽** —— 由装配处装（`sxagent/recorder.py` 在 import 时自注册）。
#: ⚠️ **默认空**：库用户 / 不落盘的宿主不装，行为就是"没有日志文件"（原则 3、取舍 ①）。
_sink_record: Callable[[dict], None] | None = None
_sink_write: Callable[[str], None] | None = None


def install_recorder(*, record: Callable[[dict], None], write: Callable[[str], None]) -> None:
    """**装上记录器** —— 总线不认识它，只认这两个函数。

    - `record(ev)`：把一条事件渲染成日志行（**权威记录**，出错会被 `_mark_hole` 标出来）；
    - `write(line)`：写一行原文（**标洞**和"订阅者出错"用 —— 它们不走渲染）。

    🔴 **为什么是槽位而不是 import**：import 它 = 总线认识领域词汇表，于是
    基础能力传递依赖领域词表。
    ⚠️ 调用方是 `sxagent/recorder.py` 自己（模块末尾一行）—— **别处不要调**。
    """
    global _sink_record, _sink_write
    _sink_record, _sink_write = record, write


def stamp() -> str:
    """日志时间戳：**年月日 + 毫秒** —— 一轮里同一秒发生好几件事（工具调用 → 返回 → 判定），
    秒级看不出先后；跨天的日志也分不清是哪天。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ──────────────────────────── 事件总线 ────────────────────────────

#: 事件的八种**形状**。每种带什么字段，见各 emitter 与 `sxagent/recorder.py`。
_EVENT_TYPES = (
    "section", "note", "block", "stream", "dump", "decision", "handoff", "cut",
)

class _Subscription:
    """`subscribe()` 的返回 —— **拿它退订**。

    ⚠️ 为什么不是"返回 `fn` 让调用方自己 `remove`"：订阅者常常是**绑定方法**
    （`chat._on_event`），每次取属性都是**新对象** —— `list.remove` 找不到它。
    所以句柄**记着它在哪个列表、是哪一个槽**。
    """

    __slots__ = ("_owner", "_slot")

    def __init__(self, owner: list, slot: list) -> None:
        self._owner, self._slot = owner, slot

    def cancel(self) -> None:
        """退订。**重复调是安全的**（已经不在里面就什么都不做）。"""
        try:
            self._owner.remove(self._slot)
        except ValueError:
            pass


def subscribe(fn: Callable[[dict], None]) -> _Subscription:
    """订阅**当前会话**的事件总线。事件按发布顺序同步送达每个订阅者。

    🔴 订阅者抛异常**不会炸业务**：兜住并原样记进日志文件 ——
    展示层坏了最多是"看不见"，不能把正在跑的一轮弄死。

    ⚠️ **订阅挂在会话上，不是模块上**：多会话共享进程时，A 的展示层收不到 B 的事件。
    ⚠️ **返回句柄**（`_Subscription`）：会话结束要能摘掉订阅，否则长期运行**只增不减**。
    不接返回值也行（老调用点一个没改）。
    """
    slot = [fn]
    session.current().subs.append(slot)
    return _Subscription(session.current().subs, slot)


def _publish(ev: dict) -> None:
    """发一个事件：**先记录器**（唯一落盘者），再同步送**当前会话**的订阅者（展示层）。

    🔴 记录器**不走订阅者那张网**：展示层坏了最多是"看不见"，记录层丢一条却是
    **权威记录破了个洞** —— 所以它单独兜，且兜住之后 `_mark_hole` **把洞标出来**。

    ⚠️ **`thread_id` 是这一层加的**（不是每个发射器自己带）：八个形状共用一处，
    加漏一个就是"某类事件不知道属于谁"。
    """
    if ev.get("type") not in _EVENT_TYPES:
        # 形状只有那八种：打错一个字母，记录器与展示层都会**静默跳过**它 —— 宁可当场炸。
        raise ValueError(f"未知事件形状 type={ev.get('type')!r}：{list(_EVENT_TYPES)}")
    # 🔴 **会话标识**（多会话共享进程时，这是"这条事件属于谁"的唯一来源）。
    #    ⚠️ 默认会话的 `thread_id` 是空串 —— 单会话宿主（CLI / 跑批）行为不变。
    ev["thread_id"] = session.current().thread_id
    # 🔴 **记录器是槽位**（不是 import、不是订阅者）—— 没装就什么都不写（那是库的用法）。
    #    ⚠️ 装了之后**处置一个字没改**：单独兜，兜住走 `_mark_hole` 把洞标出来。
    if _sink_record is not None:
        try:
            _sink_record(ev)
        except Exception as e:  # noqa: BLE001 —— 不炸业务，但**必须把洞标出来**（见 _mark_hole）
            _mark_hole(ev, e)
    for slot in list(session.current().subs):
        fn = slot[0]
        try:
            fn(ev)
        except Exception as e:  # noqa: BLE001 —— 见 subscribe 的 docstring
            # ⚠️ **没装记录器时这一行没地方写** —— 那也不能炸业务（同记录器那条的取舍）。
            if _sink_write is not None:
                _sink_write(
                    f"[{stamp()}] ⚠️ 事件订阅者出错"
                    f"（{getattr(fn, '__qualname__', fn)}）：{type(e).__name__}: {e}"
                )


def _mark_hole(ev: dict, e: Exception) -> None:
    """记录器渲染失败 → 在日志里**把洞标出来**（交接三问②：切了必须标）。

    ⚠️ 标洞本身也兜住（比如文件已关）—— 但**不许静默**：正常路径下这行会写进日志，
    读的人一眼知道**丢的是哪一条**，而不是"日志里怎么少了这一段"。
    """
    if _sink_write is None:  # 没装记录器 —— 连"洞"都无处可标（库的用法）
        return
    try:
        _sink_write(
            f"[{stamp()}] ⚠️ **这一条没有落进日志**（记录器渲染失败）："
            f"type={ev.get('type')!r} kind={ev.get('kind')!r}"
            f" —— {type(e).__name__}: {e}"
        )
    except Exception:  # noqa: BLE001 —— 连标洞都写不进去，只能放弃；仍不炸业务
        pass


# ──────────────────────────── 事件发射器（业务代码只调这一层） ────────────────────────────

#: **角色枚举** —— 展示层按它归版式（谁说的话、什么颜色），**不从 `who` 的措辞里猜**。
#:
#: 🔴 靠 `"执行器" in ev["who"]` / `"run_task" in ev["tag"]` 认角色是不行的 —— 它们是
#: **给人看的自由文本**，改个措辞展示层就**静默降级**成主 agent，没有任何报错
#: （正是"禁止反结构化"那条铁律，只是挪进了字段里）。
#:
#: 所以拆成两样：**`role` 是逻辑（枚举，逻辑依赖它）**、**`who` 是显示名（随便改）**。
#: 加新角色就加在这里 —— 加完 `ROLES` 会自动收紧，漏传的地方当场报错。
AGENT = "agent"
EXECUTOR = "executor"
ROLES = frozenset({AGENT, EXECUTOR})


def check_role(role: Any, where: str) -> str:
    """角色只能是枚举值 —— 传错就**当场炸**，不许静默当成主 agent。"""
    if role not in ROLES:
        raise ValueError(
            f"{where} 的 role 必须是 {sorted(ROLES)} 之一，收到 {role!r}。"
            "（角色是逻辑，不是措辞：见 trace.AGENT / trace.EXECUTOR）"
        )
    return role


#: **kind 的受控词汇** —— 业务语义标签，展示层按它归角色 / 归并。
#:
#: 🔴 原来是**自由字符串** —— 把 `dispatch` 打成 `delegate`、或改个名，终端上那一行就
#: **静默消失**，没有任何测试会红。所以按 `role` 的办法收口：加一个 kind 就登记在这里，
#: 漏登记的地方**当场炸**。
#: ⚠️ 加完还要去 `app/chat.py` **表态**（上屏 / 有专门分支 / 只进日志）——
#: 漏表态要有检查能发现。
KINDS = frozenset({
    # agent.delegate
    "dispatch", "supersede", "waiting", "delegate_failed",
    # executor.run_task
    "run_begin", "run_end", "executor_briefing", "nudge",
    # graph.prepare / finalize
    "feed", "relay",
    # llm.safe_structured（`llm_io` = 那次调用的 PROMPT / RESPONSE / 失败提示）
    "ask_main", "llm_io",
    # stream.stream_agent（`retry` = 这一轮抖动、已重试一次；
    # `writing` = **工具参数正在流**（写产物那段最长，实测 5–9 分钟）—— 报"已收到 N 字"）
    "retry", "writing",
    # artifacts.store_artifact（`artifact` = 产物**落盘**了：ID / 标题 / 字数 / 路径）
    "artifact",
    # graph.node_agent（`stream_empty` = 流式**一步都没吐** → 退回 `invoke` 兜底，
    # 这一趟**全程没有流式日志** —— 正是那个函数要治的黑屏，所以必须让用户看见）
    "stream_empty",
    # executor.run_task（`run_fault` = 重试过仍失败，这一轮没跑成；
    # `ask` = 申报要信息 / `delivered` = 交付产物 → 这一轮结束，两者都是**结构事实**）
    "run_fault", "ask", "delivered",
})


def _check_kind(kind: str, where: str) -> str:
    """kind 只能是受控词汇里的值（空串 = 纯记录、不上屏）—— 打错就**当场炸**。"""
    if kind and kind not in KINDS:
        raise ValueError(
            f"{where} 的 kind={kind!r} 不在受控词汇里（{sorted(KINDS)}）。"
            "加新 kind 要同时：① 登记进 trace.KINDS；"
            "② 在 app/chat.py 表态（上屏 / 有专门分支 / 只进日志）。"
        )
    return kind


def section(title: str, kind: str = "") -> None:
    """一个**段落头**（前后两道 `═══` 包着标题）。

    `kind`：业务语义标签（受控词汇，见 `KINDS`）—— 展示层按它归角色；
    不给就是"纯记录，不上屏"。
    """
    _publish({"type": "section", "kind": _check_kind(kind, "section"),
              "title": title, "ts": stamp()})


def note(text: str, kind: str = "") -> None:
    """一行记录（带时间戳）。`kind` 的语义同 `section`。

    ⚠️ **只发内容**：这一行在日志文件里的前缀（如 `relay` 的 `finalize: `）归**记录器**
    —— 业务侧带上它，展示层就得按位置切回来（措辞一改就静默吃掉几个字）。
    """
    _publish({"type": "note", "kind": _check_kind(kind, "note"), "text": text, "ts": stamp()})


def block(label: str, text: str, *, kind: str = "", indent: str = "  ") -> None:
    """一个带标签的多行文本块（prompt / response / 交接正文）。"""
    _publish({"type": "block", "kind": _check_kind(kind, "block"), "label": label,
              "text": text, "indent": indent, "ts": stamp()})


def kv(label: str, **fields: Any) -> None:
    """写一行键值对（形状上是一行 note）。"""
    inner = " | ".join(f"{k}={v}" for k, v in fields.items())
    _publish({"type": "note", "kind": "", "text": f"{label} | {inner}", "ts": stamp()})


def handoff(
    hop: str,
    frm: str,
    to: str,
    text: str,
    *,
    structured: str = "",
    full: bool = False,
) -> None:
    """**一次交接的完整记录** —— 一行摘要（＋可选全文）。

    摘要行让复盘的人**一眼看到**：
      - **谁给谁、交了多少字** → 判据①（内容）
      - **有没有结构化对象** → 判据④（结构 ＋ 模板两层分离）
      - **跳名与方向** → 判据⑤（这一跳是平台硬拼还是 LLM 转述）

    截断（判据②）由**切的那一方**自己报（`cut()`）—— 不在这里事后解析文本。
    膨胀（判据③）看每轮的上下文读数（CLI 的 `📊 CONTEXT` 行）。

    ⚠️ `full=True` 只在**别处没有 dump 过**这一跳时用（否则日志里两份）。
    """
    _publish({"type": "handoff", "kind": "", "hop": hop, "frm": frm, "to": to,
              "text": text, "structured": structured, "full": full, "ts": stamp()})


def emit(ev: dict) -> None:
    """发一个**已经是最终形状**的事件 —— 给**领域包自己的发射器**用。

    🔴 **为什么要有它**：`decision` 这个形状的发射器住在 `handoff.decision`
    （它要校验 `result_by` ∈ `RESULT_BY` —— 那是**领域词汇**，不是总线该认识的东西）。
    领域包发事件走这一条，总线就不必 import 领域包。

    ⚠️ **校验一样跑**：`type` / `kind` 都在 `_publish` 里查，走这条**不会**绕过它。
    ⚠️ 别处不要用它"绕过"现成的发射器 —— 那些发射器存在的理由是**形状只有一处定义**。
    """
    _publish(ev)


def cut(where: str, kept: int, total: int, *, see: str = "") -> None:
    """**切了一刀就报一刀** —— 由切的那一方自己调，不靠事后从文本里解析。

    标记在**文本里**（读的人看得见），这一行在**日志里**（复盘的人看得见），
    两者都要有（交接三问②）。

    - `where`：**切的是哪一份**，要说人话（谁看到的东西被切了、原件动没动）。
    - `kept` / `total`：**实测**字数（不是预算）—— 报意图不报事实，改天就静默不符。
    - `see`：**补一句给读日志的人**（"全文在哪"／"为什么没有全文"）—— 原样接在 `｜` 后面，
      所以措辞由调用点负责（要写"全文：…"就自己写）。
    """
    _publish({"type": "cut", "kind": "", "where": where, "kept": kept,
              "total": total, "see": see, "ts": stamp()})


def dump_trace(tag: str, messages: list, *, role: str) -> None:
    """把一整条消息轨迹**作为事件**发出去（执行器内部发生了什么，全在这）。

    🔴 事件里带的是**消息对象本身**（不是渲染好的行）—— 展示层靠 `tool_call_id`
    做工具配对、靠 `additional_kwargs` 认平台注入，**不从渲染文本里反推结构**。

    `role`：说话的是谁（`trace.AGENT` / `trace.EXECUTOR`），**必填** ——
    展示层按它归版式。`tag` 只是给人看的描述（`run_task 第 1 轮`），
    **逻辑不许从 tag 里认东西**（`"run_task" in tag` 那种写法实测踩过）。
    """
    check_role(role, "dump_trace")
    _publish({"type": "dump", "kind": "", "role": role, "tag": tag,
              "messages": list(messages), "ts": stamp()})


# ──────────────── 消息渲染（**各视图共用**：记录器 / 终端 / 台账 / 跑批记录） ────────────────
#
# ⚠️ 这里**不是"记录器专用"**：`app/chat.py`（终端）、`sxagent/graph.py`（待答透传）、
#    `app/runtime.py`（`_run.md`）都在用。它们是**通用的**"LangChain 消息对象 → 一行文本"，
#    **不认识任何领域词汇** —— 所以留在总线这边，而不是搬去 `sxagent/recorder.py`：
#    那样**展示层就得反过来依赖记录器**（方向反了）。

#: **正文类参数**：只留字数。
#:
#: 🔴 **只有它们该省**：`write_artifact(content=...)` 的正文
#: **另有出处**（产物文件），灌进日志纯属浪费。
#:
#: ⚠️ **其余参数一律照原样给全** —— 判据跟显示层那条一样：
#: **全文另有出处才省，没有就全给**。
BODY_ARGS = frozenset({"content"})


def brief_args(args: Any, limit: int = 120) -> str:
    """工具参数的一行摘要 —— **只有 `BODY_ARGS` 里的正文类参数**会被压成字数。"""
    if not isinstance(args, dict):
        return str(args)
    parts = []
    for k, v in args.items():
        text = str(v)
        parts.append(f"{k}=<{len(text)}字>" if k in BODY_ARGS and len(text) > limit else f"{k}={text!r}")
    return ", ".join(parts)


def tool_calls_of(messages: list) -> list[tuple[str, str]]:
    """从消息里抽出**全部**工具调用 —— `(名字, 参数摘要)`，按发生顺序。

    主 agent 的 `read_artifact` 这类调用，原来在日志里查不到（只有执行器的轨迹被 dump），
    只能靠"消息类型对不上"反推。这个函数是给日志补上那一段的。
    """
    out: list[tuple[str, str]] = []
    for m in messages:
        for c in getattr(m, "tool_calls", None) or []:
            out.append((str(c.get("name")), brief_args(c.get("args") or {})))
    return out


def render_message(m) -> str:
    """把一条 LangChain 消息渲染成可读文本（**记录器落盘用**）。

    🔴 **这里不截断**：日志是**原始记录**，而"平台注入了什么、agent 说了什么"正是
    **优化 agent 表现的唯一观察窗口** —— 要少看噪音就在**播报层**砍，别在记录层砍。

    ⚠️ **AI 的话只写头部，不重复正文** —— 正文已经由 `stream_agent` 逐行流过日志了。
    """
    kind = type(m).__name__
    content = str(m.content)
    parts = [f"{kind}({len(content)}字)"]
    # 🔴 **平台注入的标记**（`sxagent.handoff.INJECTED`）—— 透传注入的是 AIMessage，
    # 内容是**执行器的原话**；不写出来，读日志的人会以为那是主 agent 说的。
    tag = (getattr(m, "additional_kwargs", None) or {}).get("injected_by")
    if tag:
        parts.append(f"    injected_by={tag}")
    if content.strip() and kind != "AIMessage":
        # `!r` 只为**压成一行**（日志一行一条），不是为了截断。
        parts.append(f"    content={content!r}")
    for c in getattr(m, "tool_calls", None) or []:
        parts.append(f"    tool_call={c.get('name')} args={brief_args(c.get('args') or {})}")
    return "\n".join(parts)
