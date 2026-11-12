"""台账的标记与截断：故障压过 stage、待答不重印、切了说清全文在哪。

兑现 `sxagent/handoff/ledger.py` 的不变量 3、4。断言比的是标记表与来源表里的值（我们自己的
常量）、哨兵值（我们自己的输入）到没到 —— 文案改一句不该让测试红。

来源表 `ledger._SRC` 是私有的，这里照 `test_receipt.py` 的干法取来比：它是"全文在哪"的
唯一定义，往测试里抄一份就多出第二份。
"""

from __future__ import annotations

from typing import cast

import pytest

from sxagent import trace
from sxagent.handoff import BODY, DELIVERED_DESPITE_FAULT, ledger
from sxagent.handoff.briefing import Task
from sxagent.handoff.ledger import render_ledger

#: 任务下那一层细节行的缩进：段体 2 格 + 列表项 2 格。
_DETAIL = BODY * 2 + "- "


def _task(**overrides: object) -> Task:
    """一份够渲染的 Task，个别字段按用例覆盖。"""
    base: dict[str, object] = {
        "id": "t1",
        "skill": "some-skill",
        "stage": "failed",
        "report": "SENTINEL-REPORT",
        "run_by": "delegate",
    }
    return cast(Task, {**base, **overrides})


def _events() -> list[dict[str, object]]:
    """订阅当前会话的事件总线 —— 台账报的那一刀（`trace.cut`）从这儿收。"""
    evs: list[dict[str, object]] = []
    trace.subscribe(evs.append)
    return evs


def _cuts(evs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [e for e in evs if e["type"] == "cut"]


@pytest.mark.parametrize("kind", ["transient", "rejected", "timeout"])
def test_fault_mark_is_chosen_by_kind_and_overrides_stage(kind: str) -> None:
    """出故障的 Task `stage` 也是 `failed` —— 照 stage 说就是假话。"""
    text = render_ledger([_task(stage="failed", fault="SENTINEL-FAULT", fault_kind=kind)])
    assert ledger.FAULT_MARK[kind] in text
    assert [m for k, m in ledger.FAULT_MARK.items() if k != kind and m in text] == []
    assert ledger.STAGE_MARK["failed"] not in text


@pytest.mark.parametrize("kind", ["", "not-a-kind"])
def test_missing_or_unknown_fault_kind_falls_back_to_transient(kind: str) -> None:
    text = render_ledger([_task(stage="failed", fault="SENTINEL-FAULT", fault_kind=kind)])
    assert ledger.FAULT_MARK["transient"] in text
    assert ledger.STAGE_MARK["failed"] not in text


def test_faulted_task_shows_the_fault_not_what_the_executor_said() -> None:
    """故障时打故障原因、不打「执行器说」—— 它一个字都没说，照打就是冒号后面空白。"""
    text = render_ledger(
        [_task(stage="failed", fault="SENTINEL-FAULT", fault_kind="transient")]
    )
    assert "SENTINEL-FAULT" in text
    assert "SENTINEL-REPORT" not in text


def test_a_faulted_task_that_delivered_says_so() -> None:
    """故障那条默认"什么都没交"——交过的要说清（有没有交 / 交得完不完整）。"""
    said = DELIVERED_DESPITE_FAULT.format(2)
    text = render_ledger(
        [
            _task(
                stage="failed",
                fault="SENTINEL-FAULT",
                fault_kind="timeout",
                artifacts=["art_001", "art_002"],
            )
        ]
    )
    assert said in text

    plain = render_ledger([_task(stage="failed", fault="SENTINEL-FAULT", fault_kind="timeout")])
    assert DELIVERED_DESPITE_FAULT.format(2) not in plain


def test_waiting_task_reports_the_count_only() -> None:
    """不重印问题（重印的那份是截断的，会和【待答】的全文成两个来源），也不拿 report 顶。"""
    asked = "SENTINEL-ASKED-HEAD" + "x" * 120
    text = render_ledger([_task(stage="waiting_for_input", asked=asked)])
    assert "SENTINEL-ASKED-HEAD" not in text
    assert str(len(asked)) in text
    assert "SENTINEL-REPORT" not in text


def test_missing_asked_is_stated_explicitly() -> None:
    """`asked` 空（fixture / 手写 state）：显式写"没有"，不留空冒号、不静默退回 report。"""
    text = render_ledger([_task(stage="waiting_for_input", asked="")])
    detail = [ln for ln in text.splitlines() if ln.startswith(_DETAIL)]
    assert detail != []
    assert [ln for ln in detail if ln.rstrip().endswith("：")] == []
    assert "SENTINEL-REPORT" not in text


@pytest.mark.parametrize("run_by", ["delegate", "feed"])
def test_truncated_report_points_at_the_right_source(run_by: str) -> None:
    """`run_by` 决定"全文在哪"—— 指错了比指不到更坏（`feed` 那条路没有回执）。"""
    report = "SENTINEL-REPORT-HEAD" + "x" * 300
    evs = _events()
    text = render_ledger([_task(stage="failed", report=report, run_by=run_by)])
    assert ledger._SRC[run_by] in text
    assert [m for k, m in ledger._SRC.items() if k != run_by and m in text] == []
    (ev,) = _cuts(evs)
    assert ev["total"] == len(report)
    assert ev["see"] == ledger._SRC[run_by]
    assert isinstance(ev["kept"], int) and 0 < ev["kept"] < len(report)
    assert str(len(report) - ev["kept"]) in text


def test_short_report_is_not_marked_as_truncated() -> None:
    """没切就别标：没报一刀，也没说"全文在哪"。"""
    evs = _events()
    text = render_ledger([_task(stage="failed", report="SENTINEL-REPORT")])
    assert "SENTINEL-REPORT" in text
    assert _cuts(evs) == []
    assert [m for m in ledger._SRC.values() if m in text] == []
