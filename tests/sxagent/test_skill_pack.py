"""技能来源：库自带的实现满足 `SkillSource` 接口，注入之后读的确实是注入的那一份。

接口是给宿主用的（它可以实现成 OSS、数据库），所以"库自带的这个实现也满足它"要钉住 ——
不然宿主照接口写完了，库自己却对不上。另一半是"注入真被用上"：拿默认实现去验等于没验，
所以这里现造两份来源，比对它们的产出不一样。
"""

from __future__ import annotations

from pathlib import Path

from sxagent.skill_pack import (
    DEFAULT_SOURCE,
    ENTRY_NAME,
    LocalSkills,
    load_skills_dir,
    make_read_skill,
)
from sxagent.skills import Skill, SkillSource

SENTINEL = "哨兵技能"


def _one(instructions: str) -> LocalSkills:
    """只认识 `SENTINEL` 这一个技能、没有自带文件的来源。"""
    return LocalSkills({SENTINEL: Skill(SENTINEL, "一句话", instructions)})


def _pack(root: Path, name: str, files: dict[str, str]) -> Path:
    """在 `root` 下铺一个技能包目录：`SKILL.md` ＋ 给的几份自带文件。"""
    d = root / name
    (d / "references").mkdir(parents=True)
    (d / ENTRY_NAME).write_text(
        f"---\nname: {name}\ndescription: 一句话\n---\n哨兵指令", encoding="utf-8"
    )
    for rel, text in files.items():
        (d / rel).write_text(text, encoding="utf-8")
    return d


def test_default_source_satisfies_the_interface() -> None:
    assert isinstance(DEFAULT_SOURCE, SkillSource)
    assert isinstance(LocalSkills({}), SkillSource)


def test_read_file_only_accepts_what_files_reports(tmp_path: Path) -> None:
    """`files()` 就是白名单：没报过的路径一律 `None`（接口不变量 5 靠它挡路径穿越）。"""
    _pack(tmp_path, SENTINEL, {"references/规范.md": "哨兵正文"})
    source, warn = load_skills_dir(tmp_path)

    assert warn == []
    assert source.get(SENTINEL) is not None
    assert [p for p, _ in source.files(SENTINEL)] == ["references/规范.md"]
    assert source.read_file(SENTINEL, "references/规范.md") == "哨兵正文"
    # 入口不在清单里，穿越更不在 —— 两条都走同一条判据
    assert source.read_file(SENTINEL, ENTRY_NAME) is None
    assert source.read_file(SENTINEL, "../../../etc/passwd") is None


async def test_read_skill_reads_the_injected_source() -> None:
    """两份来源同一个技能名，读出来必须各是各的 —— 读的是全局表的话，两条会一模一样。"""
    got_a = await make_read_skill(_one("来源A的指令")).ainvoke({"skill_name": SENTINEL})
    got_b = await make_read_skill(_one("来源B的指令")).ainvoke({"skill_name": SENTINEL})

    assert "来源A的指令" in got_a
    assert "来源B的指令" in got_b
    assert "来源B的指令" not in got_a
