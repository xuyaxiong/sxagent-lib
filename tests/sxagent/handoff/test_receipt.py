"""故障分档：按 `FaultKind` 分的表键集合要一致，而且每一档的说法必须互不相同。

兑现 `sxagent/handoff/receipt.py` 不变量 6 —— 漏一档就是静默走兜底，读的人会看到
"原样再派"这种不该说的话。

断言比的是"各档说法彼此不同"，不是"某一档说了哪句话"：文案随便改，漏档照样红。
"""

from __future__ import annotations

from typing import get_args

from sxagent import executor
from sxagent.artifacts import store_artifact
from sxagent.executor import ExecOutcome
from sxagent.handoff import DELIVERED_DESPITE_FAULT, NUDGED_WITHOUT_DELIVERY, FaultKind, ledger
from sxagent.handoff import receipt as receipt_mod

#: 按 `FaultKind` 分档的全部表 —— 不变量 6 说"四处"，加 `ledger` 与 `executor` 两处共七张。
TABLES: dict[str, dict] = {
    "receipt._DELIVER_BY_FAULT": receipt_mod._DELIVER_BY_FAULT,
    "receipt._NEXT_BY_FAULT": receipt_mod._NEXT_BY_FAULT,
    "receipt._FAULT_SECTION": receipt_mod._FAULT_SECTION,
    "receipt.NOTE_BY_FAULT": receipt_mod.NOTE_BY_FAULT,
    "ledger.FAULT_MARK": ledger.FAULT_MARK,
    "executor.FAULT_VERDICT": executor.FAULT_VERDICT,
    "executor.FAULT_LABEL": executor.FAULT_LABEL,
}

#: 三类故障（去掉 `""` —— 那是"没有故障"）。
KINDS = [kind for kind in get_args(FaultKind) if kind]


def test_fault_tables_share_the_same_keys() -> None:
    assert {name: set(t) for name, t in TABLES.items() if set(t) != set(KINDS)} == {}


def _cells(value: object) -> tuple:
    """表里一格的值：字符串就一格，元组（判定记录那种）拆成几格。"""
    return value if isinstance(value, tuple) else (value,)


def test_each_table_reads_differently_per_kind() -> None:
    """每张表里三档的说法两两不同 —— 漏一档走了兜底，两档就撞成同一句话。

    元组的值逐格比：只抄了其中一格（比如 why 抄了、nxt 没抄）也算漏档。
    """
    colliding = []
    for name, table in TABLES.items():
        columns = [_cells(table[kind]) for kind in KINDS]
        if any(len({col[i] for col in columns}) != len(KINDS) for i in range(len(columns[0]))):
            colliding.append(name)
    assert colliding == []


def _sections(kind: str) -> tuple[str, str, str]:
    """这一档故障渲染出来的三段：交付行 / 平台故障 / 下一步。"""
    outcome = ExecOutcome(status="incomplete", fault="哨兵原因", fault_kind=kind, rounds=1)
    sections = receipt_mod.build_receipt(
        outcome, note=receipt_mod.next_note("fault", fault_kind=kind)
    )["sections"]
    return sections["deliver"], sections["fault"], sections["next"]


def test_each_fault_kind_reads_differently() -> None:
    """逐处比：三档在**每一处**渲染（交付行 / 平台故障 / 下一步）都得有自己的说法。

    比"三段合起来不同"不够 —— 漏档时往往只抄了其中一处，另外两处还是不同的，就漏过去了。
    """
    rendered = [_sections(kind) for kind in KINDS]
    for index, place in enumerate(("deliver", "fault", "next")):
        texts = {row[index] for row in rendered}
        assert len(texts) == len(KINDS), f"{place} 这一处有两档说法相同：{sorted(texts)}"


def test_no_fault_means_no_fault_section() -> None:
    artifact_id = store_artifact("哨兵产物", "哨兵梗概", "md", "哨兵正文")
    outcome = ExecOutcome(status="completed", artifacts=[artifact_id], report="哨兵报告")
    receipt = receipt_mod.build_receipt(
        outcome, note=receipt_mod.next_note("completed", skill="some-skill")
    )
    assert receipt["kind"] == "completed"
    assert "fault" not in receipt["sections"]


def test_a_faulted_run_that_delivered_says_so() -> None:
    """故障话术默认"什么都没交"——交过的必须说清（有没有交 / 交得完不完整）。"""
    artifact_id = store_artifact("哨兵产物", "哨兵梗概", "md", "哨兵正文")
    outcome = ExecOutcome(
        status="completed",
        artifacts=[artifact_id],
        fault="整趟超过墙钟上限（600s）",
        fault_kind="timeout",
    )
    sections = receipt_mod.build_receipt(
        outcome, note=receipt_mod.next_note("fault", fault_kind="timeout", delivered=1)
    )["sections"]
    said = DELIVERED_DESPITE_FAULT.format(1)

    assert said in sections["deliver"]
    assert said in sections["next"]
    assert said in sections["fault"]

    # 没交的照旧不提这句：不许给"什么都没交"的那趟编一个"交过"
    plain = ExecOutcome(status="incomplete", fault="哨兵抖动", fault_kind="transient")
    plain_sections = receipt_mod.build_receipt(
        plain, note=receipt_mod.next_note("fault", fault_kind="transient")
    )["sections"]
    assert DELIVERED_DESPITE_FAULT.format(1) not in plain_sections["deliver"]
    assert DELIVERED_DESPITE_FAULT.format(1) not in plain_sections["next"]
    assert DELIVERED_DESPITE_FAULT.format(1) not in plain_sections["fault"]


def test_a_run_that_never_delivered_says_it_was_nudged() -> None:
    """没交东西、却推了好几轮：交接处要如实说"推了 N 轮都没出口"，不许说成"一直在干活"。"""
    outcome = ExecOutcome(
        status="incomplete", fault="整趟超过墙钟上限（600s）", fault_kind="timeout", rounds=3
    )
    sections = receipt_mod.build_receipt(
        outcome, note=receipt_mod.next_note("fault", fault_kind="timeout")
    )["sections"]
    assert NUDGED_WITHOUT_DELIVERY.format(3) in sections["fault"]

    # 只跑了一轮（墙钟切在第一轮）不算"推了又推"—— 不许编
    once = ExecOutcome(
        status="incomplete", fault="整趟超过墙钟上限（600s）", fault_kind="timeout", rounds=1
    )
    once_sections = receipt_mod.build_receipt(
        once, note=receipt_mod.next_note("fault", fault_kind="timeout")
    )["sections"]
    assert NUDGED_WITHOUT_DELIVERY.format(3) not in once_sections["fault"]
