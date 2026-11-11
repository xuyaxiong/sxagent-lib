"""Skill：静态的专业技能包，以及内置技能表。

`Skill` 只有三个字段（`name` / `description` / `instructions`）。内置表在 import 时登记进
`sxagent.registry`，本模块不持有表。没有磁盘 IO，也没有工具，那是 `skill_pack.py` 和
`artifacts.py` 的事。

## 不变量

1. 技能包里的参考文件不是这里的字段：它本来就在磁盘上读得到，装进对象是多余的一份。
2. 平台对 skill 作者零强制约束，不加任何字段。不要求声明参数、交付形态、完成条件：要求了
   就得靠作者配合，而靠配合的机制是脆的。"""

from __future__ import annotations

from dataclasses import dataclass

from sxagent.registry import registry


# ──────────────────────────── Skill 定义 ────────────────────────────


@dataclass(frozen=True)
class Skill:
    """专业技能包（静态、可版本化）。

    三个字段：叫什么（`name`）、干什么（`description`）、怎么干（`instructions`）。
    参考文件不在对象里 —— 它们留在磁盘上，列与读都在 `skill_pack.py`。
    两个被删掉的字段记在这里免得再走回头路：`params`（参数声明）、
    `completion_criteria`（完成条件）—— 为什么不加，见模块头不变量 2。
    """

    name: str
    #: 主 agent 靠它判断"什么时候该用这个技能"
    description: str
    #: 完整工作流程 —— 派发时原样交给执行器
    instructions: str


#: 内置技能表 —— 登记进注册表，不再是一个模块级变量。
#: 唯一持有者是 `sxagent.registry`；这里只是"内置的那一份"的声明。
#: 不要写 `from sxagent.skills import SKILLS` —— 那会重新制造一个绑定点。
registry.register_defaults({
    "lesson-prep": Skill(
        name="lesson-prep",
        description="备课：产出教学设计要点（目标 / 重难点 / 教学过程提纲）",
        instructions=(
            "产出教学设计的要点：教学目标、重难点、教学过程提纲。精简，150 字以内。\n"
            "**你需要知道这节课的「课题」和「课时」（第几课时）**"
            "——不知道就先问，不要自己编，也不要用默认值代替。"
        ),
    ),
    "guided-learning": Skill(
        name="guided-learning",
        description="导学单：产出学生用导学单要点（学习目标 / 预习任务 / 当堂检测）",
        instructions=(
            "产出导学单要点：学习目标、预习任务、当堂检测。精简，150 字以内。\n"
            "**你需要知道「课时」（第几课时）**——不知道就先问。"
        ),
    ),
    "quiz": Skill(
        name="quiz",
        description="随堂小测：产出 3 道随堂小题（含答案）",
        instructions=(
            "产出 3 道随堂小题（含答案）。精简，150 字以内。\n"
            "**你需要知道「难度」**（基础 / 提升 / 挑战 三档）——不知道就先问。"
        ),
    ),
})
