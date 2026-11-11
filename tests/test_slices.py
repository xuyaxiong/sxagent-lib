"""源码里的切片：每一处都要么在豁免名单里写明理由，要么不许有。

兑现交接三问的第二问——"有没有被截"。切了一刀而没标，读的人分不清"它就说这么多"还是
"被切了"，而这两种情况该做的决定完全不同。所以每处 `[:N]` / `[-N:]` 都得有人认领。

判据落在结构上（AST 里有没有切片节点），所以能做成机制；至于"切得对不对"是语义，归人读。

加一处切片而没进名单，这里红 —— 要么标出来（`trace.cut` + 文本里的标记），要么写清为什么
它不算内容被切。
"""

from __future__ import annotations

import ast
from pathlib import Path

#: 扫描范围：交付面（库 + 验证 CLI）和随仓库走的脚本。
ROOTS = ("sxagent", "app", "scripts")

#: 允许存在的切片。键是 `(文件, 切片表达式的源码片段)`，值是"为什么它不算内容被切"。
#: 用源码片段而不是行号：挪几行不该让名单失效，而挪了地方该被看见。
EXEMPT: dict[tuple[str, str], str] = {
    # 标了：切了会报 `trace.cut`，文本里也带「平台截断」标记
    ("sxagent/handoff/ledger.py", "text[:limit]"): "台账里的长文本：报了 `trace.cut`，文本里带「平台截断」标记",
    ("app/chat.py", "text[:_LIVE_CAP]"): "终端摘要行：文本里带「平台截断：还有 N 字未显示」",
    ("app/chat.py", "s.error[:500]"): "终端那句报错：带「平台截断」标记并指到 results.json",
    ("app/runtime.py", "content[:1200]"): "审计记录里的工具返回：标了「共 N 字」，原文在原始 trace 附录",
    ("app/runtime.py", "content[:800]"): "同上，标了「平台截断：还有 N 字未显示」",
    ("app/runtime.py", "err[:3000]"): "运行记录里的整轮报错：带「平台截断」标记并指到 results.json",
    # 索引，不是截断：装得下的就是全文，装不下的走按需取
    ("sxagent/handoff/briefing.py", "flat[:limit]"): "用户原话的索引行：全文走 `read_user_turn(n)` 取",
    # 以下都不是内容被切：取尾部、取新增部分、拆结构、消毒文件名
    ("sxagent/__main__.py", "lines[:_BLOCK_MAX_LINES]"): "块最多印 12 行，超出写明还剩多少行、全文在哪",
    ("sxagent/agent.py", "(runtime.tool_call_id or 'x')[-8:]"): "拿 tool_call_id 的尾段当 Task 编号，不是内容",
    ("sxagent/artifacts.py", '(re.sub(r"[^\\w一-鿿]+", "-", s).strip("-") or "untitled")[:n]'): "产物文件名：消毒 + 限长，正文原样落盘",
    ("sxagent/graph.py", "msgs[prev_n:]"): "取这一轮新增的消息，不丢内容",
    ("sxagent/handoff/__init__.py", "ls[1:]"): "列表项的续行，不丢内容",
    ("sxagent/handoff/briefing.py", "done_turns[: len(t_turns)]"): "比对原话快照的前缀，不是切",
    ("sxagent/recorder.py", 're.sub(r"[^A-Za-z0-9_-]", "_", s.thread_id)[:40]'): "日志文件名：消毒 + 限长",
    ("sxagent/skill_pack.py", "lines[1:i]"): "拆 frontmatter，两半都留着",
    ("sxagent/skill_pack.py", "lines[i + 1 :]"): "同上",
    ("sxagent/trace.py", 't.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]'): "毫秒后三位，不是内容",
    ("sxagent/trace.py", 't.strftime("%H:%M:%S.%f")[:-3]'): "同上",
    ("app/chat.py", "s[: len(s) - len(s.lstrip())]"): "折行时切出缩进前缀，不丢内容",
    ("app/chat.py", "s[len(lead) :]"): "同上",
    ("app/chat.py", "lines[1:]"): "续行，不丢内容",
    ("app/runtime.py", "msgs[prev_len:]"): "取本轮新增的消息，不丢内容",
    ("app/runtime.py", '(t.get("actions") or [])[len(prev.get("actions") or []) :]'): "取本轮新增的动作，不丢内容",
    ("scripts/text_audit.py", "body[1:]"): "AST 里取函数体的其余部分，不丢内容",
}


def _slices() -> list[tuple[str, int, str]]:
    """交付面里每一处切片：`(文件, 行号, 切片表达式的源码片段)`。"""
    found: list[tuple[str, int, str]] = []
    for root in ROOTS:
        for path in sorted(Path(root).rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
                    found.append((path.as_posix(), node.lineno, ast.get_source_segment(src, node) or ""))
    return found


def test_every_slice_is_accounted_for() -> None:
    unaccounted = [
        f"{path}:{line} {segment}"
        for path, line, segment in _slices()
        if (path, segment) not in EXEMPT
    ]
    assert unaccounted == []


def test_exempt_list_has_no_stale_entry() -> None:
    """名单里的每一条都要还认得出对应的切片 —— 删了切片忘了删名单，也是死数据。"""
    live = {(path, segment) for path, _, segment in _slices()}
    assert sorted(set(EXEMPT) - live) == []


def test_every_exempt_entry_gives_a_reason() -> None:
    assert [key for key, reason in EXEMPT.items() if not reason.strip()] == []
