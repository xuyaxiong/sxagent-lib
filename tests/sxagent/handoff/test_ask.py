"""试答的输入要带全：主 agent 自己写的工具调用也在里面。

兑现"信息给全，决策结构化"——主 agent 常把要求写进 `delegate(task=…)`，试答时看不见它，
就只能报"答不上"，白打扰用户一次。正文类参数（`write_artifact(content=…)`）只留字数。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from sxagent.handoff.ask import build_prompt


class _Outcome:
    """够 `Outcome` 契约的桩：试答只读 `asked` 和 `report`。"""

    asked = "哨兵问题"
    report = "哨兵报告"


def _prompt(*messages: object) -> str:
    return build_prompt("some-skill", "SENTINEL-JOB", _Outcome(), list(messages))


def test_tool_call_args_reach_the_prompt() -> None:
    msgs = [
        HumanMessage("SENTINEL-USER"),
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "delegate",
                    "args": {"skill_name": "SENTINEL-SKILL", "task": "SENTINEL-TASK"},
                    "id": "call_1",
                }
            ],
        ),
    ]
    prompt = _prompt(*msgs)
    assert "delegate" in prompt
    assert "SENTINEL-SKILL" in prompt
    assert "SENTINEL-TASK" in prompt


def test_body_args_are_not_poured_into_the_prompt() -> None:
    body = "x" * 500
    msgs = [
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "write_artifact",
                    "args": {"title": "哨兵", "content": body},
                    "id": "call_1",
                }
            ],
        )
    ]
    prompt = _prompt(*msgs)
    assert body not in prompt  # 正文不灌进来
    assert "500" in prompt  # 但字数要报出来
