"""图装得起来、跑得完：一轮走通，消息不重不漏。

两条各守一件事：

- **装配没断**：`graph.build()` 装得起来、一轮走得完（假模型，不联网）。"CLI 必须一直能跑"
  如果只靠人偶尔跑一次，那就是个愿望；这条把它变成每次 `pytest` 都过一遍的东西——改注入面、
  改签名、动工具表时，最先红的就是它。
- **子图返回全量 state 是安全的**：`graph.node_agent` 返回的是子图累积后的**整份** state
  （含全部消息），而父图 `messages` 通道的 `add_messages` 是按 id 合并的：**没有 id 的消息会被
  再追加一遍**。真那样的话，上下文里每条消息都会重复（或者互相顶掉），而且不报错。现在能对，
  靠的是 langchain-core 会给没有 id 的消息补一个 id（1.7.0 实测）——这是**上游的行为**，不是
  我们代码保证的：哪天它不补了，那条会红。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from sxagent import artifacts, graph
from tests.fakes import ToolCallingFakeModel


class _MarkerSaver(InMemorySaver):
    """自己的 saver —— 能认出"图用的是不是这个实例"。"""


def test_injected_checkpointer_and_store_reach_the_graph() -> None:
    """注入的东西要真的进到图里：传默认值进去等于没验（`is` 一比就知道）。"""
    saver, store = _MarkerSaver(), InMemoryStore()
    app = graph.build(checkpointer=saver, store=store)
    assert app.checkpointer is saver
    assert app.store is store


class _FakeAgent:
    """替掉主 agent 子图：每走一步就往消息表尾追一条。

    消息**不设 id**，跟真模型回来的消息一样 —— 这正是要探的那件事。
    """

    async def astream(self, state, config, stream_mode=None):
        msgs = list(state["messages"])
        yield ("values", {"messages": msgs})  # 第一个 chunk = 输入 state 本身
        yield ("values", {"messages": [*msgs, AIMessage("哨兵回复")]})  # 走了一步


async def test_full_state_update_keeps_messages_whole(monkeypatch) -> None:
    monkeypatch.setattr("sxagent.graph.build_agent", lambda schema, tools=(): _FakeAgent())
    app = graph.build()
    config = {"configurable": {"thread_id": "test-graph"}}

    state = await app.ainvoke({"messages": [HumanMessage("第一句")]}, config)
    state = await app.ainvoke({"messages": [HumanMessage("第二句")]}, config)

    kinds = [type(m).__name__ for m in state["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    ids = [m.id for m in state["messages"]]
    assert len(set(ids)) == len(ids)


async def test_the_graph_assembles_and_finishes_a_turn(monkeypatch) -> None:
    """CLI 那条路：装配 → 一轮 → 收工（假模型；真会话验的是模型行为，这条验装配没断）。"""
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.agent.get_model", lambda *a, **k: fake)
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)

    app = graph.build()
    state = await app.ainvoke(
        {"messages": [HumanMessage("哨兵-A 的问题")]},
        {"configurable": {"thread_id": "test-assembly"}},
    )

    replies = [m for m in state["messages"] if isinstance(m, AIMessage) and str(m.content).strip()]
    assert replies, "主 agent 一个字都没说"
    assert artifacts.count() == 1, "它调了 write_artifact，产物该落进库里"
