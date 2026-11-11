"""主 agent 那个节点把子图的完整 state 当更新返回 —— 断消息不重不漏。

`graph.node_agent` 返回的是子图累积后的**整份** state（含全部消息），而父图 `messages` 通道
的 `add_messages` 是按 id 合并的：**没有 id 的消息会被再追加一遍**。真那样的话，上下文里每条
消息都会重复（或者互相顶掉），而且不报错 —— 模型看到的对话变了，没人知道。

现在能对，靠的是 langchain-core 会给没有 id 的消息补一个 id（1.7.0 实测）。这是**上游的行为**，
不是我们代码保证的：哪天它不补了，这条会红。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from sxagent import graph


class _FakeAgent:
    """替掉主 agent 子图：每走一步就往消息表尾追一条。

    消息**不设 id**，跟真模型回来的消息一样 —— 这正是要探的那件事。
    """

    async def astream(self, state, config, stream_mode=None):
        msgs = list(state["messages"])
        yield ("values", {"messages": msgs})  # 第一个 chunk = 输入 state 本身
        yield ("values", {"messages": [*msgs, AIMessage("哨兵回复")]})  # 走了一步


async def test_full_state_update_keeps_messages_whole(monkeypatch) -> None:
    monkeypatch.setattr("sxagent.graph.build_agent", lambda schema: _FakeAgent())
    app = graph.build()
    config = {"configurable": {"thread_id": "test-graph"}}

    state = await app.ainvoke({"messages": [HumanMessage("第一句")]}, config)
    state = await app.ainvoke({"messages": [HumanMessage("第二句")]}, config)

    kinds = [type(m).__name__ for m in state["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    ids = [m.id for m in state["messages"]]
    assert len(set(ids)) == len(ids)
