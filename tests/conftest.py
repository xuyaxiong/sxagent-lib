"""测试环境：日志指到临时目录、每个用例一个干净会话。

## 不变量

1. `SXAGENT_LOG_DIR` 必须在 `sxagent` 被 import 之前设好 —— `recorder` 在 import 那一刻
   就读它定文件名。所以这段代码在本文件顶层跑（pytest 先 import conftest，再 import 用例）。
2. 每个用例一个新会话：产物库、发号器、日志文件都跟着会话走，用例之间不串。

`no_default_model` 是注入那几条测试的哨兵：三处"造默认模型"的入口都换成"一碰就抛"。注入
某一跳忘了把 `model` 往下传时，那条路会静默回落去造默认模型（真会话里就是连网），这个哨兵
把它当场变成红的。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 测试日志是噪音，别混进给人看的 `logs/`（`tmp/` 已在 .gitignore 里）
os.environ.setdefault("SXAGENT_LOG_DIR", str(Path("tmp") / "logs"))

import pytest  # noqa: E402

from sxagent import session  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_session():
    """每个用例一个会话 —— 产物 ID 从 `art_001` 重新发，用例之间不串。"""
    with session.session("test") as s:
        yield s


@pytest.fixture
def no_default_model(monkeypatch):
    """把三处"造默认模型"的入口换成"一碰就抛" —— 断注入的模型真被用上了。"""

    def boom(*_args, **_kwargs):
        raise AssertionError("这一处该用注入的模型，不该去造默认那个（它会读 .env、连网）")

    for mod in ("sxagent.agent", "sxagent.executor", "sxagent.llm"):
        monkeypatch.setattr(f"{mod}.get_model", boom)
