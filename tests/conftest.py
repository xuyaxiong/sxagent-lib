"""测试环境：日志指到临时目录、每个用例一个干净会话。

## 不变量

1. `SXAGENT_LOG_DIR` 必须在 `sxagent` 被 import 之前设好 —— `recorder` 在 import 那一刻
   就读它定文件名。所以这段代码在本文件顶层跑（pytest 先 import conftest，再 import 用例）。
2. 每个用例一个新会话：产物库、发号器、日志文件都跟着会话走，用例之间不串。
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
