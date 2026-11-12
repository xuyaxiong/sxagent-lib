"""交接书字段覆盖（不调模型）：`Task` 的每个字段要么渲染进交接书，要么在排除名单里写明理由。

兑现 `sxagent/handoff/briefing.py` 不变量 3。加字段忘了渲染，这里红。

断言用的是夹具里的哨兵值（我们自己的输入），不是渲染出来的文案 —— 文案改一句不该让测试红，
"字段的内容有没有到达交接书"才是这条不变量管的事。
"""

from __future__ import annotations

from typing import cast, get_type_hints

from sxagent.artifacts import store_artifact
from sxagent.executor import ExecOutcome
from sxagent.handoff.briefing import Task, absorb_outcome, render_briefing

#: 刻意不渲染进交接书的字段 —— 每个都要写清为什么。
#: 放在测试这边：产品代码里不需要一个只有测试读的常量。
EXCLUDED: dict[str, str] = {
    "id": "Task 编号是平台内部的账，交接书里不出现",
    "skill": "技能名由 `run_task` 单独传，交接书里不重述",
    "stage": "任务状态是平台的账，执行器不需要知道",
    "asked": "它要的那句走回执；交接书里走 `qa` 的「问」",
    "fault": "平台故障是平台侧的事，执行器重入时用不上",
    "fault_kind": "同上",
    "run_by": "台账据此说清全文在哪（`ledger._SRC`），交接书里不渲染",
}

#: 每个字段一个哨兵值：能唯一地证明"这个字段的内容到了交接书里"。
SENTINEL: dict[str, object] = {
    "id": "SENTINEL-ID",
    "skill": "SENTINEL-SKILL",
    "task": "SENTINEL-TASK",
    "stage": "running",
    "artifacts": ["art_001"],
    "refs": ["art_002"],
    "report": "SENTINEL-REPORT",
    "asked": "SENTINEL-ASKED",
    "fault": "SENTINEL-FAULT",
    "fault_kind": "transient",
    "qa": [{"by": "用户", "q": "SENTINEL-QA-Q", "a": "SENTINEL-QA-A"}],
    "user_turns": [{"n": 1, "text": "SENTINEL-TURN", "after": "SENTINEL-AFTER"}],
    "run_by": "delegate",
}


def _declared() -> set[str]:
    """`Task` 声明的字段（`total=False`，键就是这些）。"""
    return set(get_type_hints(Task))


def _sentinels(key: str) -> list[str]:
    """这个字段里所有该到达交接书的值。"""
    value = SENTINEL[key]
    if key == "qa":
        return [str(x["q"]) for x in cast(list[dict], value)] + [
            str(x["a"]) for x in cast(list[dict], value)
        ]
    if key == "user_turns":
        return [str(x["text"]) for x in cast(list[dict], value)] + [
            str(x["after"]) for x in cast(list[dict], value)
        ]
    return [str(x) for x in value] if isinstance(value, list) else [str(value)]


def _fixture(**overrides: object) -> Task:
    """所有字段都填满的 Task，可覆盖个别字段。"""
    return cast(Task, {**SENTINEL, **overrides})


def _render(task: Task | None = None) -> str:
    """渲染一份交接书。

    产物先真的存进当前会话（`conftest` 每个用例给一个干净会话），`artifacts` / `refs`
    才渲染得出清单行。
    """
    store_artifact("SENTINEL-TITLE-1", "哨兵梗概", "md", "哨兵正文")
    store_artifact("SENTINEL-TITLE-2", "哨兵梗概", "md", "哨兵正文")
    return render_briefing(_fixture() if task is None else task)


def test_fixture_covers_every_declared_field() -> None:
    """夹具漏一个字段，下面两条都看不见它 —— 这条不变量就成了空的。"""
    assert _declared() - set(SENTINEL) == set()


def test_excluded_list_is_within_task_and_gives_reasons() -> None:
    assert set(EXCLUDED) <= _declared()
    assert set(EXCLUDED) <= set(SENTINEL)
    assert all(reason.strip() for reason in EXCLUDED.values())


def test_included_fields_reach_the_briefing() -> None:
    text = _render()
    missing = {
        key: [v for v in _sentinels(key) if v not in text] for key in set(SENTINEL) - set(EXCLUDED)
    }
    assert {key: v for key, v in missing.items() if v} == {}


def test_excluded_fields_do_not_leak() -> None:
    text = _render()
    leaked = {key: [v for v in _sentinels(key) if v in text] for key in EXCLUDED}
    assert {key: v for key, v in leaked.items() if v} == {}


def test_missing_artifact_ids_are_not_dropped() -> None:
    """坏数据要显式露出来：少一行和多一行"这条有问题"是两种不同的决定。"""
    text = _render(_fixture(artifacts=["art_999"], refs=["art_998"]))
    assert "art_999" in text
    assert "art_998" in text


def test_artifacts_accumulate_across_runs() -> None:
    """一个任务可以交很多份产物（技能每阶段一份很常见，不区分中间与最终）——
    这一趟交的不许把上一趟的顶掉；顶掉了，下一趟的交接书就"忘"了前面的。"""
    store_artifact("哨兵一号", "梗概", "md", "正文")
    store_artifact("哨兵二号", "梗概", "md", "正文")
    rec: Task = {"id": "t1", "skill": "some-skill"}

    absorb_outcome(rec, ExecOutcome(artifacts=["art_001"], report="第一趟"))
    absorb_outcome(rec, ExecOutcome(artifacts=["art_002"], report="第二趟"))

    assert rec["artifacts"] == ["art_001", "art_002"]
    text = render_briefing(rec)
    assert "art_001" in text
    assert "art_002" in text
