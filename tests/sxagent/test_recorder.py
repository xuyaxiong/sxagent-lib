"""日志落哪：默认在应用根下的 `logs/`，宿主可以用 `use_dir()` 换，目录到第一次写才建。

兑现 `recorder.py` 不变量 6。宿主把库嵌进自己的进程时，`app_root()` 指的是它装包的那个目录
（容器里可能只读），日志得能挪走；而"只是 import 这个库"不该在任何地方留东西。
"""

from __future__ import annotations

from pathlib import Path

from sxagent import recorder


def test_use_dir_puts_the_log_where_told_and_builds_it_late(tmp_path: Path) -> None:
    where = tmp_path / "深" / "一层"
    recorder.use_dir(where)

    # 目录不在 `use_dir` 那一刻建 —— 一次日志都不写就什么都不该留
    assert not where.exists()

    recorder.write("哨兵一行")

    p = recorder.log_file()
    assert p.parent == where
    assert p.read_text(encoding="utf-8") == "哨兵一行\n"


def test_dump_keeps_keys_not_full_text(monkeypatch, tmp_path: Path) -> None:
    """日志只放钥匙：长正文（技能指令 / 参考文件 / 产物这类，持久化数据里查得回）默认
    只记开头，并标出全文多少字、开关在哪；短内容整段记（读的人用得上）。"""
    from langchain_core.messages import ToolMessage

    monkeypatch.delenv("SXAGENT_LOG_PROMPT", raising=False)
    recorder.use_dir(tmp_path / "log")
    long_text = "技能指令" * 500 + "尾巴"
    recorder.write(  # 先触发建目录
        ""
    )
    from sxagent import trace

    trace.dump_trace(
        "哨兵一步", [ToolMessage(content=long_text, tool_call_id="c1")], role=trace.AGENT
    )
    body = recorder.log_file().read_text(encoding="utf-8")
    assert "尾巴" not in body, "长正文的结尾不该进日志（全文都进了说明没摘要）"
    assert f"正文共 {len(long_text)} 字" in body
    assert "SXAGENT_LOG_PROMPT=1" in body

    trace.dump_trace(
        "哨兵一步", [ToolMessage(content="短回执", tool_call_id="c2")], role=trace.AGENT
    )
    body = recorder.log_file().read_text(encoding="utf-8")
    assert "短回执" in body, "短内容整段记 —— 只放钥匙不是不放钥匙"
