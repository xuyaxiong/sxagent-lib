"""审计层的三张对照表要覆盖实际会出现的名字 —— 漏一个就是静默回落。

三张表各管一处标注，兜底值都不会报错，只是标错：

- `TOOL_SOURCE`：主 agent 每个工具在记录里怎么标来源（兜底"平台"）
- `RECEIPT_KINDS`：每个回执出口怎么标（兜底裸 key）
- `RECEIPT_LABEL`：回执每一段怎么标（兜底裸 key）

`read_skill_file` 就漏过一次（审计里抓到的）：它的返回被标成"平台"，正文也没走"原文另有
出处"那条路。所以覆盖性要断，而且要双向断。
"""

from __future__ import annotations

from app import runtime
from sxagent import agent
from sxagent.handoff import HOPS
from sxagent.handoff.receipt import DELIVERY


def test_tool_source_covers_every_tool() -> None:
    assert sorted(agent.tool_names() - set(runtime.TOOL_SOURCE)) == []


def test_tool_source_has_no_stale_tool() -> None:
    assert sorted(set(runtime.TOOL_SOURCE) - agent.tool_names()) == []


def test_receipt_kinds_covers_every_exit() -> None:
    """出口 = `DELIVERY` 那四档（completed / waiting / failed / fault）+ 两个拒绝口。"""
    exits = set(DELIVERY) | {"rejected_skill", "rejected_artifact"}
    assert sorted(exits - set(runtime.RECEIPT_KINDS)) == []


def test_receipt_labels_covers_every_section() -> None:
    sections = set(HOPS["receipt"]) | set(HOPS["receipt_rejected"])
    assert sorted(sections - set(runtime.RECEIPT_LABEL)) == []
