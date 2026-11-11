"""内置技能表：读走 `builtin_skills()`，拿到的是副本。

兑现 `skills.py` 不变量 3。这张表全进程一份，谁把返回的 dict 当成它本身去改，改的就是
所有人看得见的那一份 —— 那正是当初把表收进单独模块要治的病，所以这里钉住"发出来的是副本"。
"""

from __future__ import annotations

from sxagent.skills import Skill, builtin_skills


def test_builtin_skills_hands_out_a_copy() -> None:
    builtin_skills()["哨兵技能"] = Skill(name="哨兵技能", description="", instructions="")
    assert "哨兵技能" not in builtin_skills()
