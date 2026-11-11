"""交接词汇表自查：`SECTION` 与 `HOPS` 双向对齐。

兑现 `sxagent/handoff/__init__.py` 的"校验用 ⊆（渲染出来的都在声明里）"：

- 每一跳声明的段名都在 `SECTION` 里 —— 否则 `render_hop` 到跑起来才 KeyError
- 登记的段名至少被一跳用到 —— 没有死段名（加了段名却没接进任何一跳）
"""

from __future__ import annotations

from sxagent.handoff import HOPS, SECTION


def test_every_hop_section_is_declared() -> None:
    undeclared = {
        hop: [key for key in keys if key not in SECTION] for hop, keys in HOPS.items()
    }
    assert {hop: keys for hop, keys in undeclared.items() if keys} == {}


def test_every_declared_section_is_used_by_some_hop() -> None:
    used = {key for keys in HOPS.values() for key in keys}
    assert set(SECTION) - used == set()
