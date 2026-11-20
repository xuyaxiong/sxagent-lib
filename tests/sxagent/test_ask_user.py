"""主 agent 的 `ask_user`：提问标记、收话、提示词随工具走。

与执行器的待答同构：工具返回上挂结构标记（`additional_kwargs.ask`），`_StopWhenPending`
在同一个收话点把它挡住；宿主按同一个标记出卡片事件（在宿主侧测）。
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from sxagent import graph
from sxagent.agent import _ASK_NOTE, _StopWhenPending
from tests.fakes import ToolCallingFakeModel


class _AskingFake(ToolCallingFakeModel):
    """第一拍调 `ask_user`（带一个空选项，验过滤），之后若还有拍就收尾 —— 收话没生效才会走到。"""

    def _respond(self, messages):
        self.calls.append(("ask", time.monotonic()))
        if any(isinstance(m, ToolMessage) for m in messages):
            return self._result(AIMessage("收尾"))
        return self._result(
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "ask_user",
                        "args": {
                            "question": "这次要几课时？",
                            "options": ["1 课时", "2 课时", " "],
                        },
                        "id": "call_ask_1",
                    }
                ],
            )
        )


class _EmptyAskFake(ToolCallingFakeModel):
    """第一拍拿空问题调 `ask_user`：当场拒绝、不产生标记，这一轮照常继续。"""

    def _respond(self, messages):
        self.calls.append(("empty-ask", time.monotonic()))
        if any(isinstance(m, ToolMessage) for m in messages):
            return self._result(AIMessage("收尾"))
        return self._result(
            AIMessage(
                "",
                tool_calls=[
                    {"name": "ask_user", "args": {"question": "  "}, "id": "call_ask_empty"}
                ],
            )
        )


async def test_ask_user_ends_the_turn_with_the_marker(monkeypatch) -> None:
    """调了 `ask_user`：这一轮到此结束，末尾那条工具返回带着提问对象（问题 + 过滤后的选项）。"""
    fake = _AskingFake()
    monkeypatch.setattr("sxagent.agent.get_model", lambda *a, **k: fake)
    app = graph.build()
    config = {"configurable": {"thread_id": "test-ask"}}

    await app.ainvoke({"messages": [HumanMessage("帮我设计个活动")]}, config)

    msgs = app.get_state(config).values["messages"]
    last = msgs[-1]
    assert isinstance(last, ToolMessage), "收话没生效：ask 之后模型又说了话"
    assert last.additional_kwargs.get("ask") == {
        "question": "这次要几课时？",
        "options": ["1 课时", "2 课时"],
    }


async def test_empty_question_is_rejected_without_the_marker(monkeypatch) -> None:
    """空问题当场拒绝、不打标记 —— 这一轮不会被收话，模型看得到错误自己纠正。"""
    fake = _EmptyAskFake()
    monkeypatch.setattr("sxagent.agent.get_model", lambda *a, **k: fake)
    app = graph.build()
    config = {"configurable": {"thread_id": "test-ask-empty"}}

    await app.ainvoke({"messages": [HumanMessage("帮我设计个活动")]}, config)

    msgs = app.get_state(config).values["messages"]
    assert isinstance(msgs[-1], AIMessage), "空问题不该被当成有效提问收话"
    rejected = [
        m for m in msgs if isinstance(m, ToolMessage) and "question 不能为空" in str(m.content)
    ]
    assert rejected, "空问题没有当场拒绝"
    assert not any(
        (m.additional_kwargs or {}).get("ask") for m in msgs if isinstance(m, ToolMessage)
    )


def test_an_ask_marker_in_the_batch_stops_the_turn() -> None:
    """本批工具返回里有 ask 标记 → 这一拍不生成；标记独占一条也要认（与 handoff 同形）。"""
    state = {
        "messages": [
            AIMessage(""),
            ToolMessage("返回", tool_call_id="c1", additional_kwargs={"ask": {"question": "Q?"}}),
        ]
    }
    assert _StopWhenPending().before_model(state, None) == {"jump_to": "end"}


def test_a_plain_tool_result_is_not_stopped() -> None:
    """普通工具返回照常发言 —— 拦过头等于主 agent 不说话。"""
    state = {"messages": [AIMessage(""), ToolMessage("返回", tool_call_id="c1")]}
    assert _StopWhenPending().before_model(state, None) is None


def test_ask_note_follows_the_tool_table(monkeypatch) -> None:
    """「请求输入」那段话跟着工具走：auto / inline 有 `ask_user` 就补，delegate 没有就不提。"""
    prompts: list[str] = []

    def fake_create_agent(model, **kw):
        prompts.append(kw["system_prompt"])
        return _FakeAgent()

    monkeypatch.setattr("sxagent.agent.create_agent", fake_create_agent)

    graph.build(model=object(), exec_path="inline")
    graph.build(model=object(), exec_path="auto")
    graph.build(model=object(), exec_path="delegate")

    assert _ASK_NOTE in prompts[0]
    assert _ASK_NOTE in prompts[1]
    assert _ASK_NOTE not in prompts[2]


class _FakeAgent:
    """装配缝上的占位（与 `test_exec_path.py` 同形）：这几条测试不 invoke，只要装得起来。"""

    async def astream(self, state, config, stream_mode=None):
        yield ("values", {"messages": list(state["messages"])})
