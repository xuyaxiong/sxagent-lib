"""决策记录：两种作者不共用段名，不认识的 `result_by` 当场报错。

兑现 `sxagent/handoff/decision.py` 的不变量 3、5。断言比的是词汇表里的段名与来源标注
（我们自己的常量）到没到 —— 文案改一句不该让测试红。
"""

from __future__ import annotations

import pytest

from sxagent import trace
from sxagent.handoff import SECTION, SOURCE_LABEL, SOURCE_NOTE
from sxagent.handoff.decision import RESULT_BY, emit_decision, render_decision


def _emitted(result_by: str) -> dict[str, object]:
    """走 `emit_decision` 真发一条，收总线上那条事件 —— 测的是 emit → render 这条路。"""
    evs: list[dict[str, object]] = []
    sub = trace.subscribe(evs.append)
    try:
        emit_decision(
            "SENTINEL-WHO",
            q="SENTINEL-Q",
            given=["SENTINEL-GIVEN"],
            result=["SENTINEL-RESULT"],
            nxt="SENTINEL-NEXT",
            result_by=result_by,
        )
    finally:
        sub.cancel()
    assert len(evs) == 1
    return evs[0]


def test_llm_result_is_sourced_and_platform_verdict_is_not() -> None:
    """两种作者不共用一个段名：共用必有一边是假话（标了不是它写的，或没标是它写的）。"""
    llm = render_decision(_emitted("llm"))
    platform = render_decision(_emitted("platform"))
    assert SECTION[RESULT_BY["llm"][1]] in llm
    assert SECTION[RESULT_BY["platform"][1]] not in llm
    assert SECTION[RESULT_BY["platform"][1]] in platform
    assert SECTION[RESULT_BY["llm"][1]] not in platform
    # 四格里只有【结果】有来源行；平台判定不标（自证没意义）
    assert llm.count(SOURCE_LABEL) == 1
    assert SOURCE_NOTE[RESULT_BY["llm"][1]] in llm
    assert SOURCE_LABEL not in platform
    assert "SENTINEL-RESULT" in llm
    assert "SENTINEL-RESULT" in platform


def test_unknown_result_by_is_rejected_at_emit() -> None:
    """打错一个字母就静默把平台的结论标成"它说的" —— 宁可当场报错。"""
    with pytest.raises(ValueError) as e:
        emit_decision("SENTINEL-WHO", result="SENTINEL-RESULT", result_by="platfrom")
    msg = str(e.value)
    assert "platfrom" in msg
    assert [by for by in RESULT_BY if by not in msg] == []


def test_unknown_result_by_is_rejected_at_render() -> None:
    ev: dict[str, object] = {
        "type": "decision",
        "who": "SENTINEL-WHO",
        "result": "SENTINEL-RESULT",
        "result_by": "platfrom",
    }
    with pytest.raises(ValueError):
        render_decision(ev)
