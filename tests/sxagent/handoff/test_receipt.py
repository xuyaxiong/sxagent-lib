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
from sxagent.handoff import FaultKind, ledger
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
