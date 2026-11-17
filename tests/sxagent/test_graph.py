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
from sxagent.artifacts import MemoryArtifacts
from sxagent.executor import ExecOutcome
from sxagent.graph import next_pending, render_relay
from sxagent.handoff.briefing import Task
from sxagent.skill_pack import LocalSkills
from sxagent.skills import Skill
from tests.fakes import ToolCallingFakeModel


class _MarkerSaver(InMemorySaver):
    """自己的 saver —— 能认出"图用的是不是这个实例"。"""


class _SpyStore(MemoryArtifacts):
    """记下谁往它里写了 —— 断"注入的产物库真被用上"（传默认实现等于没验）。"""

    def __init__(self) -> None:
        self.puts: list[str] = []

    async def put(
        self, aid: str, title: str, summary: str, file_type: str, content: str
    ) -> None:
        self.puts.append(aid)
        await super().put(aid, title, summary, file_type, content)


async def test_injected_artifact_store_is_actually_used(monkeypatch) -> None:
    """主 agent 写的产物要落进注入的那个库 —— 落回默认库就等于没注入。"""
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.agent.get_model", lambda *a, **k: fake)
    spy = _SpyStore()

    app = graph.build(artifacts=spy)
    await app.ainvoke(
        {"messages": [HumanMessage("哨兵-A 的问题")]},
        {"configurable": {"thread_id": "test-inject-store"}},
    )

    assert len(spy.puts) == 1, "它写了产物，就该落进注入的那个库"


def test_injected_checkpointer_and_store_reach_the_graph() -> None:
    """注入的东西要真的进到图里：传默认值进去等于没验（`is` 一比就知道）。"""
    saver, store = _MarkerSaver(), InMemoryStore()
    app = graph.build(checkpointer=saver, store=store)
    assert app.checkpointer is saver
    assert app.store is store


def test_the_injected_skill_source_reaches_the_agent(monkeypatch) -> None:
    """技能来源要真的到装配那一层 —— 装配的缝上收下参数、`is` 一比就知道有没有被丢掉。"""
    seen: dict[str, object] = {}

    def fake_build_agent(schema, **kw):
        seen.update(kw)
        return _FakeAgent()

    monkeypatch.setattr("sxagent.graph.build_agent", fake_build_agent)
    source = LocalSkills({"哨兵技能": Skill("哨兵技能", "一句话", "哨兵指令")})

    graph.build(skills=source)

    assert seen["skills"] is source


def test_the_business_prompt_is_appended_after_the_mechanism_block(monkeypatch) -> None:
    """业务段只追加在末尾：机制段（含技能目录）逐字不动；不传就一个字不加。

    宿主的 system_prompt 按会话传入，这是它的落点；机制段是库的，不许被替换。
    """
    prompts: list[str] = []

    def fake_create_agent(model, **kw):
        prompts.append(kw["system_prompt"])
        return _FakeAgent()

    monkeypatch.setattr("sxagent.agent.create_agent", fake_create_agent)

    graph.build(model=object())
    graph.build(model=object(), system_prompt="业务段：面向中学教师的备课助手")

    assert prompts[1] == prompts[0] + "\n\n业务段：面向中学教师的备课助手"
    assert not prompts[0].endswith("\n\n")  # 不传时连分隔符都不加


async def test_the_feed_path_uses_the_injected_store_and_source(monkeypatch) -> None:
    """投递那条路也要走注入：查技能按注入的来源，产物落进注入的库，模型用注入的那个。

    三个都得断 —— 少穿一个参数，用户答复后的那次重跑就会写进库自带的产物库、按库自带的
    技能表查技能、用库自带的模型，而这条路上没有任何报错。
    """
    seen: dict[str, object] = {}

    async def fake_run_task(skill, briefing="", user_turns=None, **kw):
        seen["skill"] = skill.name
        seen.update(kw)
        return ExecOutcome(status="completed")

    monkeypatch.setattr("sxagent.executor.run_task", fake_run_task)
    store = _SpyStore()
    source = LocalSkills({"哨兵技能": Skill("哨兵技能", "一句话", "哨兵指令")})
    model = object()  # 这一跳只验"带过去了没有"，所以随便一个哨兵就够
    node = graph._make_node_feed(store, source, model)

    await node(
        {
            "messages": [HumanMessage("这是答复")],
            "answer_to": "t1",
            "tasks": [{"id": "t1", "skill": "哨兵技能", "stage": "waiting_for_input"}],
        }
    )

    assert seen["skill"] == "哨兵技能"  # 技能是照注入的来源查出来的
    assert seen["artifacts"] is store
    assert seen["skills"] is source
    assert seen["model"] is model


async def test_the_feed_path_injects_the_result_back(monkeypatch) -> None:
    """投回跑完的结果要补一条回执进主 agent 的流。

    不补的话，它手里那条 delegate 回执停在"它在要信息"，只能按过期信息决策 ——
    实测：照着旧回执把用户的答复再转一遍、重开一趟委派（新任务没有旧问答，执行器按
    问询纪律又问下一条，用户侧就成了"反复请求参数"）。
    """
    deliveries: list[ExecOutcome] = []

    async def fake_run_task(skill, briefing="", user_turns=None, **kw):
        outcome = deliveries.pop(0)
        return outcome

    monkeypatch.setattr("sxagent.executor.run_task", fake_run_task)
    store = _SpyStore()
    source = LocalSkills({"哨兵技能": Skill("哨兵技能", "一句话", "哨兵指令")})
    node = graph._make_node_feed(store, source, None)
    state = {
        "messages": [HumanMessage("这是答复")],
        "answer_to": "t1",
        "tasks": [{"id": "t1", "skill": "哨兵技能", "stage": "waiting_for_input"}],
    }

    # ① 跑完了：回执要带【投回】+ 任务号 + 产物清单
    deliveries.append(ExecOutcome(status="completed", artifacts=["art_x"], report="干完了"))
    out = await node(dict(state))
    (msg,) = out["messages"]
    assert msg.additional_kwargs["injected_by"] == "platform.feed"
    assert "【投回】" in msg.content
    assert "`t1`" in msg.content and "哨兵技能" in msg.content
    assert "art_x" in msg.content

    # ② 还是缺信息：同一形态，问题也一并交回去（台账那边不重印问题，靠这份）
    deliveries.append(
        ExecOutcome(status="waiting_for_input", asked="请给出这节课的上法")
    )
    out = await node(dict(state))
    (msg,) = out["messages"]
    assert "【投回】" in msg.content
    assert "【它要的信息】" in msg.content and "请给出这节课的上法" in msg.content


def test_the_relay_carries_the_question_not_the_monologue() -> None:
    """贴给用户的是它申报的那句原话（`asked`），不是它最后说的那段（`report`）。

    用户要答的是问题，不是自述；退回 `report` 会把"数据缺失"藏成"看起来正常"。
    """
    t: Task = {
        "id": "t1",
        "skill": "哨兵技能",
        "stage": "waiting_for_input",
        "asked": "哨兵问题",
        "report": "哨兵独白",
    }

    text = render_relay([t])

    assert "哨兵问题" in text
    assert "哨兵独白" not in text


def test_next_pending_picks_the_waiting_one() -> None:
    """通道一次摆一个：摆的是待答那条，没有待答就什么都不摆。"""
    done: Task = {"id": "t1", "skill": "a", "stage": "completed"}
    waiting: Task = {"id": "t2", "skill": "b", "stage": "waiting_for_input"}

    assert next_pending({"tasks": [done, waiting]}) == waiting
    assert next_pending({"tasks": [done]}) is None


async def test_the_injected_model_reaches_the_main_agent(no_default_model) -> None:
    """注入的模型要真的用在主 agent 上 —— 默认模型那三处入口都换成了"一碰就抛"。"""
    fake = ToolCallingFakeModel()

    app = graph.build(model=fake)
    await app.ainvoke(
        {"messages": [HumanMessage("哨兵-A 的问题")]},
        {"configurable": {"thread_id": "test-inject-model"}},
    )

    assert fake.calls, "这一步不是它跑的"


class _FakeAgent:
    """替掉主 agent 子图：每走一步就往消息表尾追一条。

    消息**不设 id**，跟真模型回来的消息一样 —— 这正是要探的那件事。
    """

    async def astream(self, state, config, stream_mode=None):
        msgs = list(state["messages"])
        yield ("values", {"messages": msgs})  # 第一个 chunk = 输入 state 本身
        yield ("values", {"messages": [*msgs, AIMessage("哨兵回复")]})  # 走了一步


async def test_full_state_update_keeps_messages_whole(monkeypatch) -> None:
    monkeypatch.setattr(
        "sxagent.graph.build_agent", lambda schema, **kw: _FakeAgent()
    )  # 装配的缝变了，假工厂跟着改
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
