"""用户原话快照：三个字段、"原话硬搬"、一个上文只配一轮。

`build_user_turns` 是"用户原话由平台原样搬，不许模型转述"这条判据的落点。这里钉住三件事：
文字是逐字搬的、每轮挂上它回答的那一句、连着两条用户消息时第二条不挂同一条 AI 消息。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sxagent.agent import _AgentAnswer, build_user_turns, try_answer_then_rerun
from sxagent.executor import ExecOutcome
from sxagent.handoff.briefing import Task
from sxagent.skills import Skill
from tests.fakes import ToolCallingFakeModel


def test_text_is_carried_verbatim() -> None:
    out = build_user_turns([HumanMessage("  原文，带空格和换行\n第二行  ")])
    assert out[0]["text"] == "原文，带空格和换行\n第二行"  # 只去首尾空白，不改字
    assert out[0]["n"] == 1


def test_each_turn_carries_the_line_it_answers() -> None:
    out = build_user_turns(
        [
            AIMessage("上一句：你选哪个？"),
            HumanMessage("选 A"),
            AIMessage("好，那我继续"),
            HumanMessage("接着做"),
        ]
    )
    assert [(t["n"], t["text"], t["after"]) for t in out] == [
        (1, "选 A", "上一句：你选哪个？"),
        (2, "接着做", "好，那我继续"),
    ]


def test_two_user_messages_in_a_row_do_not_share_one_context() -> None:
    """连着两条用户消息时，第二条的语境是第一条 —— 挂同一条 AI 消息就是错的上下文。"""
    out = build_user_turns(
        [AIMessage("更早的一句"), HumanMessage("第一句"), HumanMessage("第二句")]
    )
    assert [t["after"] for t in out] == ["更早的一句", ""]


async def test_the_injected_model_reaches_the_structured_call(no_default_model) -> None:
    """试答那一步是结构化调用，也要走注入的模型 —— 没传下去就落回默认那个（会读 `.env`、连网）。"""
    fake = ToolCallingFakeModel(structured={"_AgentAnswer": _AgentAnswer(can_answer=False)})
    rec: Task = {"id": "t1", "skill": "哨兵技能"}
    outcome = ExecOutcome(status="waiting_for_input", asked="缺课时")

    got, answer = await try_answer_then_rerun(
        rec,
        Skill(name="哨兵技能", description="", instructions=""),
        outcome,
        messages=[],
        model=fake,
    )

    assert (got, answer) == (outcome, "")  # 答不上：原样返回，由调用方登记待答
    # 断"它被调过"而不是"没炸"：`safe_structured` 把异常全吞了，落回默认模型炸了也长这样
    assert fake.structured_calls == 1


def test_only_human_messages_count_as_user_words() -> None:
    """主图里台账是 `SystemMessage`、透传是 `AIMessage`，所以 Human 才是用户本人说的。"""
    out = build_user_turns(
        [SystemMessage("台账"), HumanMessage("用户说的"), AIMessage("主 agent 说的")]
    )
    assert [t["text"] for t in out] == ["用户说的"]
