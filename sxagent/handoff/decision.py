"""决策（一次判断 → 一条记录）：判断的两端，加平台的两句描述。

一次判断发生时，把它拿到的、被问的、吐出来的、平台据此怎么处置的记下来，渲染成【前提】
【问题】【结果】【下一步】四格。两种判断共用它：LLM 的试答（是交接），和平台的结构判定
（`executor._verdict`，没有 LLM 参与）。

## 不变量

1. 它为什么算交接：关键在 LLM 参与——外部给 LLM 什么、LLM 吐出什么，两端各是一次交接。
2. 四格各有各的类：【前提】【问题】【下一步】是平台的框架话，不标；只有【结果】要标来源。
3. 两种作者不共用一个段名（`RESULT_BY`）：【结果】标来源，【平台判定】不标，共用必有一边
   是假话。
4. 渲染只有一处：记录层和终端都走 `render_decision()`，载体各自处理（时间前缀、逐行上色）。
5. 不认识的 `result_by` 当场炸，不静默退回 `llm`。"""

from __future__ import annotations

from typing import Any

from sxagent import trace
from sxagent.handoff import bullet_lines, render_hop

#: 事件字段 → 段名 key。顺序不在这张表，在 `HOPS[hop]`。
_DECISION_KEYS: dict[str, str] = {
    "given": "dec_given",
    "q": "dec_q",
    "result": "dec_result",
    "next": "next",
}

#: 【结果】那格谁写的 → 哪个 hop、哪个段名。两种作者不能共用一个段名，否则必有一边是假话：
#: 模型吐的要标来源，平台自己的结论不标（自证没意义）。
RESULT_BY: dict[str, tuple[str, str]] = {
    "llm": ("decision", "dec_result"),
    "platform": ("verdict", "verdict_result"),
}


def emit_decision(
    who: str,
    *,
    q: str = "",
    given: Any = (),
    result: Any = (),
    nxt: str = "",
    result_by: str = "llm",
) -> None:
    """记一次判断：前提 → 问题 → 结果 → 下一步（段名与顺序见本模块上面）。

    两种判断共用这一条记录形状 —— 它们问的是同一件事（谁基于什么、得出了什么、接下来
    干什么），只是"填它的人"不同：试答（LLM，是交接）／结构判定（平台，不是交接，但四格
    一样成立）。
    发的是结构化事件（字段原样在 `ev` 里）—— 渲染成什么样是记录器与展示层各自的事，
    展示层不再从渲染文本里反解析。
    `given` / `result` 传列表（逐项一行）→ 渲染成 `- ` 列表；传字符串 → 一行。列表化不是
    装饰：原来直接把字段打出来，人看到的是 Python repr。
    `result_by` 决定那格用哪个段名（见 `RESULT_BY`）。发射器住在这儿、不住总线：`result_by`
    的合法值是领域词汇，而总线只认识形状、不认识词表。
    """
    if result_by not in RESULT_BY:
        # 打错一个字母就静默把平台的结论标成"它说的" —— 宁可当场炸。
        raise ValueError(
            f"decision 的 result_by={result_by!r} 不在 {sorted(RESULT_BY)} 里（见 handoff.decision.RESULT_BY）"
        )
    trace.emit({"type": "decision", "kind": "", "who": who, "q": q,
                "given": given, "result": result, "next": nxt,
                "result_by": result_by, "ts": trace.stamp()})


def render_decision(ev: dict) -> str:
    """决策事件 → 渲染出来的文本。记录层和终端共用这一处。

    列表 → `- ` 列表项（续行对齐，`bullet_lines`）；字符串 → 一段；空的不渲染。
    `result_by` 决定第三个格子用哪个段名 / 哪个 hop —— 不认识的取值当场炸
    （不静默退回 `llm`：那会把平台的结论标成"它说的"）。
    """
    by = str(ev.get("result_by") or "llm")
    if by not in RESULT_BY:
        raise ValueError(f"未知的 result_by={by!r}：合法值 {sorted(RESULT_BY)}（见 handoff.decision）")
    hop, result_key = RESULT_BY[by]
    keys = {**_DECISION_KEYS, "result": result_key}
    sections: dict[str, str] = {}
    for field, key in keys.items():
        val: Any = ev.get(field)
        items = [str(x) for x in val] if isinstance(val, (list, tuple)) else [str(val or "")]
        items = [x for x in items if x.strip()]
        if not items:
            continue
        if len(items) == 1 and not isinstance(val, (list, tuple)):
            sections[key] = items[0]
        else:
            lines: list[str] = []
            for it in items:
                lines += bullet_lines(it, "")
            sections[key] = "\n".join(lines)
    return render_hop(hop, sections)
