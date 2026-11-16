"""执行路径（`build(exec_path=…)`）：三档工具集、提示词补句、非法值当场报错。

判据全落在结构上：工具名集合、补句的键集合、枚举校验 —— 不看话语。业务层（宿主）按会话
传值，不传就是 `auto`，行为与参数化之前一致。
"""

from __future__ import annotations

import pytest

from sxagent import graph
from sxagent.agent import _MODE_NOTE, EXEC_PATHS, _tools_for, validate_exec_path
from sxagent.artifacts import DEFAULT_STORE
from sxagent.skill_pack import DEFAULT_SOURCE


def _names(exec_path: str) -> set[str]:
    return {t.name for t in _tools_for(exec_path, DEFAULT_STORE, DEFAULT_SOURCE, None)}


def test_delegate_path_keeps_only_delegation_tools() -> None:
    """`delegate` 档拿掉原料：技能原文与产物写入工具都不在，只剩委派那条路。"""
    assert _names("delegate") == {"delegate", "list_artifacts"}


def test_inline_path_has_no_delegate_tool() -> None:
    """`inline` 档是全硬的：`delegate` 工具根本不存在；亲自执行要用的都在。"""
    names = _names("inline")
    assert "delegate" not in names
    need = {"read_skill", "read_skill_file", "write_artifact", "read_artifact", "list_artifacts"}
    assert need <= names, f"inline 档缺工具：{need - names}"


def test_auto_path_is_the_union_of_the_other_two() -> None:
    """`auto` 两条路都开 = 另外两档的并集，一个不少 —— 默认行为不许比参数化前更窄。"""
    assert _names("auto") == _names("delegate") | _names("inline")


def test_every_non_default_mode_has_a_prompt_note() -> None:
    """工具集变了就得让它知道：非 auto 的每一档都有补句，auto 一个字不加。"""
    assert set(_MODE_NOTE) == set(EXEC_PATHS) - {"auto"}
    assert all(_MODE_NOTE[mode] for mode in _MODE_NOTE)


def test_unknown_exec_path_is_rejected_with_the_legal_values() -> None:
    """非法值当场报错、报错里带合法值 —— 不再炸成看不出哪错了的 KeyError。"""
    with pytest.raises(ValueError) as err:
        validate_exec_path("handoff")
    assert "handoff" in str(err.value)
    for legal in EXEC_PATHS:
        assert legal in str(err.value)


def test_build_rejects_unknown_exec_path_before_touching_the_model(no_default_model) -> None:
    """校验发生在造模型之前：夹具让"碰默认模型就抛"，这里必须抛的是 ValueError。"""
    with pytest.raises(ValueError):
        graph.build(exec_path="bogus")


def test_exec_path_reaches_build_agent(monkeypatch) -> None:
    """注入的档位要真的到装配那一层；不传就是 auto —— 装配的缝上收下参数，一比就知道。"""
    seen: list[object] = []

    def fake_build_agent(schema, **kw):
        seen.append(kw.get("exec_path"))
        return _FakeAgent()

    monkeypatch.setattr("sxagent.graph.build_agent", fake_build_agent)

    graph.build()
    graph.build(exec_path="inline")

    assert seen == ["auto", "inline"]


def test_inline_note_is_wired_into_the_system_prompt(monkeypatch) -> None:
    """补句要真的进主 agent 的系统提示：non-default 档带上自己那句，auto 不带任何补句。"""
    prompts: list[str] = []

    def fake_create_agent(model, **kw):
        prompts.append(kw["system_prompt"])
        return _FakeAgent()

    monkeypatch.setattr("sxagent.agent.create_agent", fake_create_agent)

    graph.build(model=object(), exec_path="inline")
    graph.build(model=object(), exec_path="auto")

    assert _MODE_NOTE["inline"] in prompts[0]
    assert _MODE_NOTE["delegate"] not in prompts[1]


class _FakeAgent:
    """装配缝上的占位（与 `test_graph.py` 同形）：这几条测试不 invoke，只要装得起来。"""

    async def astream(self, state, config, stream_mode=None):
        yield ("values", {"messages": list(state["messages"])})
