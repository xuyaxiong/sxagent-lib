"""展示层的 kind 表态：三个集合的并集必须覆盖 `trace.KINDS`。

兑现 `app/chat.py` 那句"有断言盯着"：新加一个 kind 而没在展示层表态，那一行静默消失 ——
跑真会话也未必看得见（它本来就该出现，不出现没人会觉得少）。
"""

from __future__ import annotations

from app import chat
from sxagent import trace


def test_display_layer_declares_every_kind() -> None:
    assert set(trace.KINDS) - chat.declared_kinds() == set()


def test_display_layer_has_no_stale_kind() -> None:
    """表态过的 kind 必须真的存在 —— 删了 kind 忘了删表态，同样是死代码。"""
    assert chat.declared_kinds() - set(trace.KINDS) == set()
