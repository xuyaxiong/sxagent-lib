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
