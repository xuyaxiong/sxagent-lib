"""审计：一次文本改动有没有碰到代码。

    uv run python scripts/text_audit.py            # 工作区 vs HEAD
    uv run python scripts/text_audit.py --rev dev  # 工作区 vs 别的提交

只回答三个**能确定**的问题，不做风格判断：

1. 剥掉全部 docstring 后，两版 `ast.dump` 是否相同 —— 逻辑没动
2. 剥掉 docstring 后，全部字符串常量的多重集是否相同 —— 产品文案没动
3. 前后各有多少 docstring 行、注释行 —— 只报数，不设阈值

体例（该不该写、压没压成一句）不进这里：那是判断，正则只能穷举它想到的反面，
穷举不到的一律静默漏过，通过得越多越像在骗自己。体例本身在 CLAUDE.md 第四节。
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

DOC_HOLDERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _strip_docs(tree: ast.AST) -> ast.AST:
    """删掉所有节点体首位置的字符串表达式，即 docstring。"""
    for node in ast.walk(tree):
        if not isinstance(node, DOC_HOLDERS):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                node.body = body[1:]
    return tree


def _strings(tree: ast.AST) -> Counter[str]:
    """剥掉 docstring 之后的全部字符串常量。"""
    counter: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            counter[node.value] += 1
    return counter


def _comment_lines(src: str) -> int:
    count = 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            count += 1
    return count


def _doc_lines(tree: ast.AST) -> int:
    total = 0
    for node in ast.walk(tree):
        doc = ast.get_docstring(node, clean=False) if isinstance(node, DOC_HOLDERS) else None
        if doc:
            total += len(doc.splitlines())
    return total


def _shape(src: str) -> tuple[str, Counter[str], int, int]:
    tree = ast.parse(src)
    docs = _doc_lines(tree)
    strings = _strings(_strip_docs(tree))
    return ast.dump(tree, include_attributes=False), strings, docs, _comment_lines(src)


def _show(rev: str, path: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None


def main(argv: list[str]) -> int:
    rev = "HEAD"
    if "--rev" in argv:
        i = argv.index("--rev")
        if i + 1 >= len(argv):
            print("--rev 后面要跟一个提交，比如 --rev dev")
            return 2
        rev = argv[i + 1]

    changed = subprocess.run(
        ["git", "diff", "--name-only", rev, "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if not changed:
        print(f"与 {rev} 相比没有 .py 改动")
        return 0

    broken = 0
    skipped = 0
    for path in changed:
        if not Path(path).is_file():
            print(f"{path}: 已删除，跳过")
            skipped += 1
            continue
        new_src = Path(path).read_text(encoding="utf-8")
        old_src = _show(rev, path)
        if old_src is None:
            print(f"{path}: 新增文件，无对照")
            continue

        new_tree, new_strings, new_docs, new_comments = _shape(new_src)
        old_tree, old_strings, old_docs, old_comments = _shape(old_src)

        marks = []
        if new_tree != old_tree:
            marks.append("代码不同")
            broken += 1
        if new_strings != old_strings:
            diff = new_strings - old_strings
            lost = old_strings - new_strings
            marks.append(f"字符串不同（+{sum(diff.values())} / -{sum(lost.values())}）")
            broken += 1
        if not marks:
            marks.append("代码、字符串均未变")

        print(
            f"{path}: {'；'.join(marks)}"
            f"｜docstring {old_docs}→{new_docs} 行，注释 {old_comments}→{new_comments} 条"
        )

    tail = f"，{skipped} 个已删除（没对照）" if skipped else ""
    print(f"\n{len(changed)} 个文件，{broken} 处代码/文案变动{tail}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
