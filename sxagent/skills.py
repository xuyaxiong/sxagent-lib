"""Skill：静态的专业技能包，库内置的技能表，以及"技能从哪来"的接口。

`Skill` 只有三个字段（`name` / `description` / `instructions`）。内置表就是本模块里的一份
字面量，读走 `builtin_skills()`。技能从哪来是接口 `SkillSource`（定义在这儿），磁盘上那个
实现在 `skill_pack.py`。本模块没有磁盘 IO，也没有工具。

## 不变量

1. 技能包里的参考文件不是这里的字段：它本来就在磁盘上读得到，装进对象是多余的一份。
2. 平台对 skill 作者零强制约束，不加任何字段。不要求声明参数、交付形态、完成条件：要求了
   就得靠作者配合，而靠配合的机制是脆的。
3. 内置表只走 `builtin_skills()` 读，不把 dict 传出去：谁拿到谁就能改，那是一个绑定点。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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


# ──────────────────────────── 库内置的技能表 ────────────────────────────

#: 库内置的技能表。私有：读走 `builtin_skills()`，不把这个 dict 传出去。
_BUILTIN: dict[str, Skill] = {
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
}


def builtin_skills() -> dict[str, Skill]:
    """库内置那张技能表的副本 —— 宿主不注入技能来源时，用的就是这一份。"""
    return dict(_BUILTIN)


# ──────────────────────────── 技能从哪来 ────────────────────────────


@runtime_checkable
class SkillSource(Protocol):
    """技能从哪来 —— 库只认这几个方法；宿主可以从数据库、OSS 或别的地方给。

    ## 不变量

    1. `read_file` 只认 `files()` 报过的路径 —— 路径穿越、二进制、超大文件全靠这一条挡。
       库自带的实现这么干，别的实现也得这么干。
    2. 排版不归实现：实现给数据，清单怎么拼是库里的事（`render_catalog` /
       `skill_pack.skill_file_prompt`）。
    """

    def get(self, name: str) -> Skill | None:
        """按名取技能；没有就 None（"没有"是业务事实，由调用方决定怎么回话）。"""

    def values(self) -> list[Skill]:
        """全部技能（顺序 = 清单顺序）。"""

    def files(self, name: str) -> list[tuple[str, int]]:
        """这个技能自带的文件：`(相对包根的路径, 字数)`；没有就空列表。"""

    def read_file(self, name: str, path: str) -> str | None:
        """读自带文件的全文；不在 `files()` 里就 None。"""


def render_catalog(skills: Sequence[Skill]) -> str:
    """技能清单的排版（渐进式披露第一层）—— 进主 agent 的提示词，只有这一处拼。"""
    return "\n".join(f"- {s.name}：{s.description}" for s in skills)
