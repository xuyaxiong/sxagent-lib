"""工具表的公共规矩：一张表里的名字不许撞。

主 agent 与执行器各有一张工具表，宿主的领域工具进的是同一份（`build(tools=…)`）——
撞名了模型叫哪个都说不清，而且提示词里写着"用 `write_artifact` 交付"这类话，两个同名
工具会让那句话变成假话。`executor` 与 `agent` 互相不 import，所以这条规矩住这儿，
两边各查各的表。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from langchain_core.tools import BaseTool


def check_unique_tool_names(tools: Sequence[BaseTool]) -> None:
    """一张工具表里不许重名；撞了当场报错。

    Raises:
        ValueError: 表里有重名（报文带着名字，说清改哪儿）。
    """
    dupes = sorted(name for name, n in Counter(tool.name for tool in tools).items() if n > 1)
    if dupes:
        raise ValueError(
            f"工具名撞了：{'、'.join(dupes)} —— 同一个 agent 的工具表里不许重名，"
            "改宿主那个工具的名字（或者拿掉它，用库的同名工具）"
        )
