"""两个执行器并行跑：产物、状态、事件各归各的（假模型，不联网）。

兑现 `executor.py` 不变量 6——运行态按 run 现造（`created` / `asked` / 读原话的工具都是闭包），
不用模块级全局：两个 `delegate` 并行时会互相覆盖。

顺带把"两个 run 确实重叠过"也断掉：不重叠的话，这条测试只是顺序跑了两遍，什么都没验到。
最后一条是墙钟：内层 agent 循环转不完的一趟，必须在预算内被就地叫停。
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from sxagent import artifacts, trace
from sxagent.artifacts import MemoryArtifacts
from sxagent.executor import ExecOutcome, _absorb_delivered, _Run, _structural_status, run_task
from sxagent.skills import Skill
from tests.fakes import ToolCallingFakeModel

MARKER_A, MARKER_B = "哨兵-A", "哨兵-B"


class _SpyStore(MemoryArtifacts):
    """记下谁往它里写了 —— 断"注入的产物库真被用上"。"""

    def __init__(self) -> None:
        self.puts: list[str] = []

    async def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
        self.puts.append(aid)
        await super().put(aid, title, summary, file_type, content)


def test_asking_wins_over_delivering() -> None:
    """三态由结构定，顺序不能反：交了一部分之后卡住也算在要信息 —— 反了用户就没机会答。

    穷举三条：申报优先，其次交了产物，都没有才是"没出口"。
    """
    assert (
        _structural_status(ExecOutcome(asked="缺课时", artifacts=["art_001"]))
        == "waiting_for_input"
    )
    assert _structural_status(ExecOutcome(artifacts=["art_001"])) == "completed"
    assert _structural_status(ExecOutcome()) == "incomplete"


async def test_executor_writes_into_the_injected_store(monkeypatch) -> None:
    """执行器交付的产物也要落进注入的那个库（这条走的是 `run_task` 的注入）。"""
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)
    spy = _SpyStore()

    outcome = await run_task(_skill("skill-a", MARKER_A), "交接书", artifacts=spy)

    assert len(spy.puts) == 1, "执行器交的产物该落进注入的那个库"
    assert outcome.artifacts == spy.puts, "回执里报的产物 ID 就是落库的那一个"


def _skill(name: str, marker: str) -> Skill:
    """技能指令里带标记 —— 假模型从消息里认出它，拿它当产物名。"""
    return Skill(name=name, description=f"{marker} 的技能", instructions=f"任务标记：{marker}")


class _RunawayModel(ToolCallingFakeModel):
    """永远在写产物的假模型 —— 复刻实测里那个退化循环：每步都调 `write_artifact`，从不收尾。

    每步睡 0.2 秒，让墙钟先于内层的步数上限触发 —— 验的才是这道闸。
    """

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(0.2)
        self.calls.append(("runaway", time.monotonic()))
        return self._result(
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "write_artifact",
                        "args": {
                            "title": "循环产物",
                            "summary": "退化循环",
                            "file_type": "md",
                            "content": "正文",
                        },
                        "id": f"call-{len(self.calls)}",
                    }
                ],
            )
        )


async def test_the_injected_model_reaches_the_executor(no_default_model) -> None:
    """注入的模型要真的用在执行器上 —— 默认模型那三处入口都换成了"一碰就抛"。"""
    fake = ToolCallingFakeModel()

    outcome = await run_task(_skill("skill-a", MARKER_A), "交接书", model=fake)

    assert outcome.status == "completed"
    assert fake.calls, "这一趟不是它跑的"


async def test_two_runs_in_parallel_do_not_cross(monkeypatch) -> None:
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)

    outcome_a, outcome_b = await asyncio.gather(
        run_task(_skill("skill-a", MARKER_A), "交接书 A"),
        run_task(_skill("skill-b", MARKER_B), "交接书 B"),
    )

    # ① 两个都交付了，而且产物 ID 各是各的
    assert (outcome_a.status, outcome_b.status) == ("completed", "completed")
    assert len(outcome_a.artifacts) == 1
    assert len(outcome_b.artifacts) == 1
    assert set(outcome_a.artifacts).isdisjoint(outcome_b.artifacts)

    # ② 产物归属没串：A 的产物里写着 A 的标记，B 同理
    assert {artifacts.get(aid)["title"] for aid in outcome_a.artifacts} == {MARKER_A}
    assert {artifacts.get(aid)["title"] for aid in outcome_b.artifacts} == {MARKER_B}


async def test_the_two_runs_really_overlap(monkeypatch) -> None:
    """不重叠的"并发测试"什么也没验到 —— 这条断它们确实同时在跑。"""
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)

    await asyncio.gather(
        run_task(_skill("skill-a", MARKER_A), "交接书 A"),
        run_task(_skill("skill-b", MARKER_B), "交接书 B"),
    )

    spans = {m: [t for mm, t in fake.calls if mm == m] for m in (MARKER_A, MARKER_B)}
    assert spans[MARKER_A] and spans[MARKER_B]
    overlap = max(spans[MARKER_A][0], spans[MARKER_B][0]) < min(
        spans[MARKER_A][-1], spans[MARKER_B][-1]
    )
    assert overlap


class _NeverDeliversModel(ToolCallingFakeModel):
    """只说话、不交产物、也不申报 —— 复刻"推了又推、始终没出口"的那类。

    每步睡 0.05 秒：不睡的话 1 秒墙钟里能转几千轮，测试没意义还吵。
    """

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(0.05)
        self.calls.append(("chat", time.monotonic()))
        return self._result(AIMessage("我先看看……"))


async def test_a_run_that_never_delivers_is_nudged_until_the_wall_clock(monkeypatch) -> None:
    """没有轮次上限：它自己不交东西，平台就一路推，最后由墙钟切停 —— 推了几轮要如实进账。"""
    monkeypatch.setattr("sxagent.executor.MAX_WALL_S", 1.0)
    fake = _NeverDeliversModel()
    seen: list[dict[str, object]] = []
    subscription = trace.subscribe(seen.append)

    outcome = await run_task(_skill("skill-a", MARKER_A), "交接书", model=fake)
    subscription.cancel()

    assert outcome.fault_kind == "timeout"
    assert outcome.artifacts == []
    assert outcome.rounds > 1, "推了不止一轮才被切 —— 没有轮次上限在提前判死"
    faults = [ev for ev in seen if ev.get("kind") == "run_fault"]
    assert faults and faults[0]["fault_kind"] == "timeout", "分类要结构进事件，展示层不解析文本"


async def test_tool_calls_report_start_and_end(monkeypatch) -> None:
    """执行器每次调工具要报两条机制事件（起 / 止）—— 宿主靠它把执行器的工具行显示出来。

    换图后执行器的图事件整个被过滤，不报就等于全黑。断的是"报得出、带结构"：名字、
    起止、成败、耗时都在字段里，展示层不用去解析那句话。
    """
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)
    seen: list[dict[str, object]] = []
    subscription = trace.subscribe(seen.append)

    outcome = await run_task(_skill("skill-a", MARKER_A), "交接书")
    subscription.cancel()

    assert outcome.status == "completed"
    tool_events = [e for e in seen if e.get("kind") == "executor_tool"]
    starts = [e for e in tool_events if e.get("state") == "start"]
    ends = [e for e in tool_events if e.get("state") == "end"]
    assert [e["tool"] for e in starts] == ["write_artifact"]
    assert [e["tool"] for e in ends] == ["write_artifact"], "起止要成对，止只报一次"
    assert ends[0]["ok"] is True
    assert isinstance(ends[0]["elapsed_s"], float)
    # 起止带同一个调用号（模型给的 tool_call_id，拿不到则 run_id）：并行同名调用分得清
    assert starts[0]["call_id"] == ends[0]["call_id"]


async def test_model_calls_report_a_thinking_event(monkeypatch) -> None:
    """执行器每次调模型报一条 `thinking`——宿主据此在第一个字出来前显示"思考中"。

    `thinking` 只报"开始了"，不报结束：展示层收到执行器任何后续动静就把那行收掉。
    """
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)
    seen: list[dict[str, object]] = []
    subscription = trace.subscribe(seen.append)

    outcome = await run_task(_skill("skill-a", MARKER_A), "交接书")
    subscription.cancel()

    assert outcome.status == "completed"
    thinking = [e for e in seen if e.get("kind") == "thinking"]
    assert thinking, "一次模型调用都没报"
    assert all(e.get("type") == "note" for e in thinking)


async def test_a_runaway_round_is_cut_by_the_wall_clock(monkeypatch) -> None:
    """内层循环转不完的一趟，要在墙钟内被就地叫停 —— 实测里它跑过 20 分钟、写了 14 份产物。

    断的是结构事实：故障档是 `timeout`、它真被拦住了（内层没转很多步、产物没攒一堆），
    而且**被切掉的那趟写过的产物要进账**（账上"产物 0 个"而库里有，视图就说假话）。
    内层没有任何步数上限（实测：无闸门时 60 秒能转 3400 次模型调用），所以测试自带一个
    硬上限——闸门坏了就判红，不把 CI 挂死。
    """
    monkeypatch.setattr("sxagent.executor.MAX_WALL_S", 1.0)
    fake = _RunawayModel()

    try:
        outcome = await asyncio.wait_for(
            run_task(_skill("skill-a", MARKER_A), "交接书", model=fake), 20
        )
    except TimeoutError:
        pytest.fail("墙钟没拦住：这一趟 20 秒都没结束")

    assert outcome.fault_kind == "timeout"
    # 交了产物 → 三态就是 `completed`（它只说"交没交"这个事实）；"被切停"由 `fault_kind` 另说
    assert outcome.status == "completed"
    assert len(fake.calls) < 10, "墙钟没拦住：内层转了很多步"
    assert 0 < len(outcome.artifacts) < 10, "被切掉那趟的产物要进账，也不该攒一堆"


async def test_host_tools_reach_the_executor() -> None:
    """宿主的领域工具要进执行器的工具表 —— 技能指令里写着用什么，它手边就得有什么。

    用哨兵工具的反向断言：执行器调不到它的话，`called` 就是空的。
    """
    from langchain_core.tools import tool

    called: list[str] = []

    @tool
    async def query_knowledge_sentinel(query: str) -> str:
        """哨兵工具：执行器该能调到它。"""
        called.append(query)
        return "查到的材料"

    class _SentinelFirstModel(ToolCallingFakeModel):
        """执行器：先调哨兵工具，再交产物 —— 两步都当场证。"""

        stage: str = "sentinel"

        def _respond(self, messages: list[BaseMessage]) -> ChatResult:
            if self.stage == "sentinel":
                self.stage = "write"
                self.calls.append(("sentinel", time.monotonic()))
                return self._result(
                    AIMessage(
                        "",
                        tool_calls=[
                            {
                                "name": "query_knowledge_sentinel",
                                "args": {"query": "材料关键词"},
                                "id": "call-sentinel",
                            }
                        ],
                    )
                )
            if self.stage == "write":
                self.stage = "done"
                return self._result(
                    AIMessage(
                        "",
                        tool_calls=[
                            {
                                "name": "write_artifact",
                                "args": {
                                    "title": "哨兵产物",
                                    "summary": "哨兵梗概",
                                    "file_type": "md",
                                    "content": f"正文：{MARKER_A}",
                                },
                                "id": "call-write",
                            }
                        ],
                    )
                )
            return self._result(AIMessage("执行器收尾"))

    outcome = await run_task(
        _skill("skill-a", MARKER_A),
        "交接书",
        model=_SentinelFirstModel(),
        extra_tools=[query_knowledge_sentinel],
    )

    assert called == ["材料关键词"], "执行器没拿到宿主的工具"
    assert outcome.status == "completed"


async def test_host_tool_named_like_a_mechanism_tool_is_rejected() -> None:
    """宿主的工具撞了机制工具名：当场报错。

    挑 `ask_user` 试：它不在主 agent 的工具表里，主 agent 那道守卫盖不到 —— 执行器
    自己得查一遍，否则一张表里两个同名工具，模型叫哪个都说不清。
    """
    from langchain_core.tools import tool

    @tool
    async def ask_user(question: str) -> str:
        """撞名哨兵。"""
        return ""

    with pytest.raises(ValueError, match="工具名撞了"):
        await run_task(
            _skill("skill-a", MARKER_A),
            "交接书",
            model=ToolCallingFakeModel(),
            extra_tools=[ask_user],
        )


async def test_ask_user_options_ride_to_the_outcome() -> None:
    """`ask_user` 给了备选就一路带进结果：结构走结构的路，谁都不用去解析那句话。"""
    from langchain_core.messages import AIMessage

    class _AskingModel(ToolCallingFakeModel):
        """问一次就收尾 —— 无状态：已经有工具结果就说明问过了。"""

        def _respond(self, messages: list[BaseMessage]) -> ChatResult:
            if any(isinstance(m, ToolMessage) for m in messages):
                return self._result(AIMessage("等答复"))
            return self._result(
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {
                                "question": "这节课侧重哪种上法？",
                                "options": ["探究型", "讲解型"],
                            },
                            "id": "call-ask",
                        }
                    ],
                )
            )

    outcome = await run_task(_skill("skill-a", MARKER_A), "交接书", model=_AskingModel())

    assert outcome.status == "waiting_for_input"
    assert outcome.asked == "这节课侧重哪种上法？"
    assert outcome.asked_options == ["探究型", "讲解型"]


class _ProgramDeliveringModel(ToolCallingFakeModel):
    """按脚本调一次程序侧交付工具（工具自己落库、结果消息带结构化字段），然后收尾。"""

    def _respond(self, messages: list[BaseMessage]) -> ChatResult:
        self.calls.append(("程序侧", time.monotonic()))
        if any(isinstance(m, ToolMessage) for m in messages):
            return self._result(AIMessage("收尾"))
        return self._result(
            AIMessage(
                "",
                tool_calls=[{"name": _deliver_program_side.name, "args": {}, "id": "call-prog"}],
            )
        )


@tool
async def _deliver_program_side(
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """假工具：模拟组卷那样的程序侧交付 —— 自己落库，结果消息带 `delivered_artifacts`。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="已交付 art_prog0001",
                    tool_call_id=tool_call_id,
                    additional_kwargs={
                        "delivered_artifacts": ["art_prog0001"],
                        "working_data": ["m-test360"],
                    },
                )
            ]
        }
    )


async def test_program_side_delivery_counts_as_delivered(monkeypatch) -> None:
    """程序侧交付要认进交付账：不进账的话它明明交了却被当成"没交"，推一轮接着干。"""
    fake = _ProgramDeliveringModel()
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)
    spy = _SpyStore()

    outcome = await run_task(
        _skill("skill-a", MARKER_A), "交接书", artifacts=spy, extra_tools=[_deliver_program_side]
    )

    assert outcome.artifacts == ["art_prog0001"]
    assert outcome.working_data == ["m-test360"], "中间数据另立一本账，与交付物分开"
    assert outcome.status == "completed", "进了账就该收工，而不是被推回去"
    assert outcome.rounds == 1, "交付那一轮就收工（模型自己的收尾话照常说完）——没被推着再跑一轮"


def test_working_data_never_counts_as_delivery() -> None:
    """只报中间数据不算交付：产出中间料不等于"这一趟干完了"，出口判定不许认它。"""
    run = _Run(
        skill=_skill("skill-a", MARKER_A),
        run_id="R99",
        t0=0.0,
        messages=[],
        outcome=ExecOutcome(),
        created=[],
        working=[],
        asked={},
    )
    _absorb_delivered(
        run,
        [
            ToolMessage(
                content="",
                tool_call_id="call-1",
                additional_kwargs={"working_data": ["m-1", "m-1", "m-2"]},
            )
        ],
    )
    assert run.working == ["m-1", "m-2"]
    assert run.created == []
