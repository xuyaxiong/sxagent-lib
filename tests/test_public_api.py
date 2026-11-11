"""公共面：`sxagent/__init__.py` 里 import 进来的每个名字，都要在 `__all__` 里表过态。

导出面是宿主唯一该依赖的东西。顺手 import 一个进来而没表态，它就悄悄成了公共 API —— 之后
改它、删它都算破坏兼容，而没人记得当初有没有这个意思。反向也要断：`__all__` 里写着、实际
没 import 的名字是死名字，`from sxagent import *` 会当场炸。

判据落在结构上（AST 里的 import 名 vs `__all__`），所以能做成机制；"该不该导出它"是取舍，
归人读。
"""

from __future__ import annotations

import ast
from pathlib import Path

INIT = Path("sxagent/__init__.py")

#: 导进来但不算公共面的：`名字 → 为什么`。目前一个都没有 —— 导进来的都是要给宿主的。
EXEMPT: dict[str, str] = {}


def _init_tree() -> ast.Module:
    return ast.parse(INIT.read_text(encoding="utf-8"))


def _imported() -> set[str]:
    """`__init__.py` 里 import 进来的名字（`as` 过的取别名）。"""
    out: set[str] = set()
    for node in ast.walk(_init_tree()):
        if isinstance(node, ast.ImportFrom):
            out |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            out |= {(a.asname or a.name.split(".")[0]) for a in node.names}
    return out


def _declared() -> set[str]:
    """`__all__` 列了哪些名字。"""
    for node in _init_tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return set()


def test_every_import_is_declared() -> None:
    undeclared = sorted(_imported() - _declared() - set(EXEMPT))
    assert undeclared == []


def test_no_stale_name_in_all() -> None:
    stale = sorted(_declared() - _imported())
    assert stale == []
