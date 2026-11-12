"""两个执行器并行跑：产物、状态、事件各归各的（假模型，不联网）。

兑现 `executor.py` 不变量 6——运行态按 run 现造（`created` / `asked` / 读原话的工具都是闭包），
不用模块级全局：两个 `delegate` 并行时会互相覆盖。

顺带把"两个 run 确实重叠过"也断掉：不重叠的话，这条测试只是顺序跑了两遍，什么都没验到。
最后一条是墙钟：内层 agent 循环转不完的一趟，必须在预算内被就地叫停。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from sxagent import artifacts, trace
from sxagent.artifacts import MemoryArtifacts
from sxagent.executor import ExecOutcome, _structural_status, run_task
from sxagent.skills import Skill
from tests.fakes import ToolCallingFakeModel

MARKER_A, MARKER_B = "哨兵-A", "哨兵-B"


class _SpyStore(MemoryArtifacts):
    """记下谁往它里写了 —— 断"注入的产物库真被用上"。"""

    def __init__(self) -> None:
        self.puts: list[str] = []

    async def put(
        self, aid: str, title: str, summary: str, file_type: str, content: str
    ) -> None:
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
