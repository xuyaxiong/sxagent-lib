"""注解里不许出现 `Any` 和裸 `list` / `dict` —— 除了写明理由的例外。

注解是给人读的（不跑 mypy 强校验），糊一个 `Any` 等于没写；裸 `list` / `dict` 连"里面是什么"
都不说。类型能写出来的就写出来，写不出来的（框架签名、跨模块取不到类型）在 `EXEMPT` 里写清
为什么 —— 沉默地放一个 `Any` 不行，那看起来跟随手写的一样。

判据落在结构上（AST 里的注解节点），所以能做成机制。
"""

from __future__ import annotations

import ast
from pathlib import Path

#: 扫描范围：交付面（库 + 验证 CLI）。
ROOTS = ("sxagent", "app")

#: 允许的例外：`(文件, 所在函数, 注解片段)` → 为什么这里标不出更细的类型。
EXEMPT: dict[tuple[str, str, str], str] = {
    ("sxagent/agent.py", "before_model", "Any"): (
        "跟着 langchain 的中间件签名；状态类型住在 `graph.py`，import 会成环"
    ),
}

_BARE = ("Any", "list", "dict")


def _annotations(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """每一处注解：`(所在函数名, 注解节点)`。"""
    out: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
            for arg in args:
                if arg.annotation is not None:
                    out.append((node.name, arg.annotation))
            if node.returns is not None:
                out.append((node.name, node.returns))
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            out.append(("<模块或类级>", node.annotation))
    return out


def _bare_names(annotation: ast.AST) -> list[str]:
    """注解里的裸 `Any` / `list` / `dict`。

    `list[Task]` 里的 `list` 是下标的名字，不算裸；`list[dict]` 里的 `dict` 算 —— 它什么都没说。
    """
    subscripted = {id(n.value) for n in ast.walk(annotation) if isinstance(n, ast.Subscript)}
    return [
        n.id
        for n in ast.walk(annotation)
        if isinstance(n, ast.Name) and n.id in _BARE and id(n) not in subscripted
    ]


def _offenders() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for root in ROOTS:
        for path in sorted(Path(root).rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            for func, annotation in _annotations(ast.parse(src)):
                found += [(path.as_posix(), func, name) for name in _bare_names(annotation)]
    return found


def test_no_bare_any_or_container_in_annotations() -> None:
    unaccounted = [f"{path} {func} 的 {name}" for path, func, name in _offenders()
                   if (path, func, name) not in EXEMPT]
    assert unaccounted == []


def test_exempt_list_has_no_stale_entry() -> None:
    live = set(_offenders())
    assert sorted(set(EXEMPT) - live) == []


def test_every_exempt_entry_gives_a_reason() -> None:
    assert [key for key, reason in EXEMPT.items() if not reason.strip()] == []
