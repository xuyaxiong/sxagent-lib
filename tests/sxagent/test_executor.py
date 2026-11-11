"""两个执行器并行跑：产物、状态、事件各归各的（假模型，不联网）。

兑现 `executor.py` 不变量 6——运行态按 run 现造（`created` / `asked` / 读原话的工具都是闭包），
不用模块级全局：两个 `delegate` 并行时会互相覆盖。

顺带把"两个 run 确实重叠过"也断掉：不重叠的话，这条测试只是顺序跑了两遍，什么都没验到。
"""

from __future__ import annotations

import asyncio

from sxagent import artifacts
from sxagent.artifacts import MemoryArtifacts
from sxagent.executor import run_task
from sxagent.skills import Skill
from tests.fakes import ToolCallingFakeModel

MARKER_A, MARKER_B = "哨兵-A", "哨兵-B"


class _SpyStore(MemoryArtifacts):
    """记下谁往它里写了 —— 断"注入的产物库真被用上"。"""

    def __init__(self) -> None:
        self.puts: list[str] = []

    def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
        self.puts.append(aid)
        super().put(aid, title, summary, file_type, content)


async def test_executor_writes_into_the_injected_store(monkeypatch) -> None:
    """执行器交付的产物也要落进注入的那个库（这条走的是 `run_task` 的注入）。"""
    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.executor.get_model", lambda *a, **k: fake)
    spy = _SpyStore()

    outcome = await run_task(_skill("skill-a", MARKER_A), "交接书", artifacts=spy)

    assert spy.puts == ["art_001"]
    assert outcome.artifacts == ["art_001"]


def _skill(name: str, marker: str) -> Skill:
    """技能指令里带标记 —— 假模型从消息里认出它，拿它当产物名。"""
    return Skill(name=name, description=f"{marker} 的技能", instructions=f"任务标记：{marker}")


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
