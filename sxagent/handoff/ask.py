"""试答（平台 → 主 agent）：执行器卡住时，先逼主 agent 用自己的上下文补一次。

执行器调 `ask_user` 申报"要信息"之后走这一跳：把它要什么（`asked`）、派了什么（`task`）、
它最后说了什么（`report`）、主 agent 的上下文（`ask_context`）摆到面前，请它判断。能不打扰
用户就不打扰，是这一步的全部意义。

## 不变量

1. 三处都不截断：这是独立的结构化调用，它只看得见这个提示词里的东西，信息落在长消息后半段
   它就答不上来，只能去打扰用户。
2. 别退回"最近 N 条"：那是无声的范围限制，比截断更难发现。
3. `injected_by` 是结构，`_who` 读它（挂在 `additional_kwargs` 上，不漏给模型），不去看
   "内容像不像平台的话"。
4. 一张单子要么整块在下一层，要么别塞进来（`_item`）。
5. 渲染走唯一那条渲染路（`render_hop`）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sxagent.handoff import BODY, INJECTED, Outcome, body_lines, render_hop

if TYPE_CHECKING:  # 只给标注用 —— 运行时不 import
    from langchain_core.messages import BaseMessage

#: 消息类型 → 人话（读的人是主 agent，所以 `AIMessage` 是"你"）。
#: 不把框架的 Python 类名漏给模型，交接书里一律说"用户说的"、"它说的"。
_WHO: dict[str, str] = {
    "HumanMessage": "用户",
    "AIMessage": "你",
    "ToolMessage": "工具返回",
    "SystemMessage": "平台",
}


def _who(m: BaseMessage) -> str:
    """这条消息是谁说的 —— 打过注入标记的（平台透传）不叫"你说的"。

    `injected_by` 是结构（挂在 `additional_kwargs` 上），这里读它，不是"看内容像不像平台的
    话"。
    """
    if (getattr(m, "additional_kwargs", None) or {}).get("injected_by") in INJECTED:
        return "**平台转达**"
    return f"**{_WHO.get(type(m).__name__, type(m).__name__)}**"


def _item(m: BaseMessage) -> str:
    """一条消息 → 一个列表项（多行的整块缩在下一层）。

    一张单子要么整块在下一层，要么别塞进来 —— 单行拼法会把工具返回 / 平台注入那张单子的
    第一行粘在列表项后面、段名落回和 `- ` 同一列，读的人分不清哪个是哪个。
    """
    body = str(m.content)
    lines = body.splitlines() or [""]
    if len(lines) == 1:
        return f"- {_who(m)}：{lines[0]}"
    # 段名那一行不跟内容 —— 整张单子缩在下一层（`render_hop` 还会再 +2）
    return "\n".join([f"- {_who(m)}：", *body_lines(body, BODY)])


def build_prompt(skill_name: str, task: str, outcome: Outcome, messages: list[BaseMessage]) -> str:
    """试答那一步喂给主 agent 的全部输入 —— 抽成纯函数是为了可单测。

    走唯一那条渲染路（`render_hop`）：段名 0 列 · 来源 2 列 · 段体 2 列。
    `skill_name` 跟着【任务】走 —— 它属于"这次要什么"，不单起一段，但不能丢：这一步是
    独立调用，模型手里没有 `delegate` 的工具上下文，不写它就不知道派给了谁。
    `messages` 渲染的是 `str(m.content)` —— **工具调用的参数不在里面**（`AIMessage.tool_calls`
    的 `args`：主 agent 写的 `delegate(task=…)` 全文、`read_artifact` 的 id 都看不见）。已知口子。
    """
    msgs = list(messages)
    history = "\n".join(_item(m) for m in msgs)
    return render_hop(
        "ask",
        {
            # 要什么 = 工具参数里的原话（结构），不是从它的话里挑出来的
            "asked": str(getattr(outcome, "asked", "") or "").strip(),
            "task": f"（委派给「{skill_name}」）\n{str(task or '').strip()}",
            "report": str(getattr(outcome, "report", "") or "").strip(),
            # "全部 N 条、全文、不截断"写在段体里 —— 它是内容的一部分，不是"来源"
            # （来源那一行回答的是"这句是谁说的"）。
            "ask_context": f"（全部 {len(msgs)} 条，全文，不截断）\n{history}",
            "ask_decide": (
                "你的上下文里**有没有它需要的信息**？\n"
                "- 有（用户之前说过）→ `can_answer=true`，`answer` 里给出补充内容，**原话照搬**\n"
                "- 没有 → `can_answer=false`，**不要编造**"
            ),
        },
    )
