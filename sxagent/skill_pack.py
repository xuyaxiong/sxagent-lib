"""技能包：文件夹形态的技能和它带的参考文件，管磁盘 IO 与格式。

    my-skill/
    ├── SKILL.md      ← 入口，只有这一个名字；frontmatter 给 name/description，正文即 instructions
    ├── references/   ← 可选，没有也行
    └── assets/       ← 目录名随作者，整个包都能读

`LocalSkills` 是库自带的技能来源（`SkillSource` 的磁盘实现）：`load_skills_dir()` 扫一个目录，
每个子目录一个包。两个给 LLM 的读工具由 `make_read_skill` / `make_read_skill_file` 造，落在
注入的来源上；`skill_file_prompt` 是任务书和 `read_skill` 共用的提示块。

## 不变量

1. 包里的文件不进 `Skill` 对象：磁盘上读得到，装进对象是多余的一份。来源只记包目录，
   清单与正文每次现场扫盘。
2. 读不读、读哪份是 LLM 的决定，这里只提供"有哪些"和"读一份"。
3. 只认 `SKILL.md`，不做"跟目录同名"的兼容：两种写法并存就得靠猜。子目录名不限定。
4. 收白名单扩展名的文本文件（`TEXT_EXTS`），不收二进制、超大文件、隐藏目录，在"列"这一步
   就过滤掉。
5. `read_file` 只认 `files()` 报得出来的东西：路径穿越、二进制、超大文件全被这一条挡住，
   不需要第二套路径检查。
6. 清单必须带 `skill_name`：`read_skill_file` 的第一个参数是技能名。
7. 提示块只有一份（`skill_file_prompt`）：两个调用方都调它，别各拼一遍。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain.tools import tool

from sxagent.skills import Skill, SkillSource, builtin_skills

#: 技能包的入口文件名 —— 只有这一个。
ENTRY_NAME = "SKILL.md"

#: 收哪些扩展名当参考文件（小写、含点）。要加就加这里。
TEXT_EXTS: frozenset[str] = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".tsv",
        ".html",
        ".xml",
        ".toml",
        ".ini",
        ".sql",
        ".log",
        ".py",
        ".js",
        ".ts",
    }
)

#: 单个参考文件的大小上限（字节）。超了不列 —— 参考文件是给模型读的，几 MB 的多半放错了地方。
MAX_REF_BYTES = 256 * 1024


def split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """拆 `---` 包裹的 frontmatter 与正文；没有就元数据为空。

    结束标记必须是独占一行的 `---`，不能 `text.split("---")`：值里可以带 `---`（真实
    技能就有），naive split 会在引号中间断开，YAML 直接解析失败。
    """
    s = text.lstrip("﻿").lstrip()  # BOM 也算 —— 有的编辑器会写进去
    lines = s.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            try:
                meta = yaml.safe_load("\n".join(lines[1:i])) or {}
            except yaml.YAMLError:
                return {}, text
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text


# ──────────────────────────── 技能从磁盘来 ────────────────────────────


def _scan_pack(pack: Path) -> list[tuple[str, int]]:
    """扫一个技能包目录 → `[(相对包根的路径, 字数), ...]`。现场扫盘，不缓存。

    排除入口 `SKILL.md`、隐藏目录、非文本扩展名、超大文件。目录里没东西就空列表：
    没有就是没有，不是错误。
    """
    out: list[tuple[str, int]] = []
    for p in sorted(pack.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(pack)
        if rel.as_posix() == ENTRY_NAME or any(part.startswith(".") for part in rel.parts):
            continue
        if p.suffix.lower() not in TEXT_EXTS or p.stat().st_size > MAX_REF_BYTES:
            continue
        # 名字用相对技能包根的路径，和 SKILL.md 里写的一模一样 —— 作者本来就是那么引用的。
        out.append((rel.as_posix(), len(p.read_text(encoding="utf-8", errors="replace"))))
    return out


@dataclass(frozen=True)
class LocalSkills:
    """技能从磁盘上的技能包来 —— 库自带的实现。

    `skills` 是技能表、`pack_dirs` 是"技能名 → 它的包目录"，两样都由 `load_skills_dir()` 扫
    出来。自带文件不进内存：列和读都按 `pack_dirs` 现场扫盘。
    """

    skills: Mapping[str, Skill]
    pack_dirs: Mapping[str, Path] = field(default_factory=dict)

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def values(self) -> list[Skill]:
        return list(self.skills.values())

    def files(self, name: str) -> list[tuple[str, int]]:
        pack = self.pack_dirs.get(name)
        return [] if pack is None else _scan_pack(pack)

    def read_file(self, name: str, path: str) -> str | None:
        """读一份自带文件的全文；不在 `files()` 里就返回 `None`（不变量 5 靠的就是这一条）。"""
        pack = self.pack_dirs.get(name)
        if pack is None or path not in {p for p, _ in _scan_pack(pack)}:
            return None
        return (pack / path).read_text(encoding="utf-8", errors="replace")


#: 库自带的技能来源：内置那张技能表，没有包目录。不注入就用它。
DEFAULT_SOURCE: SkillSource = LocalSkills(builtin_skills())


# ──────────────────────────── 技能自带文件：清单 + 提示块 ────────────────────────────


def skill_file_index_lines(skill_name: str, source: SkillSource = DEFAULT_SOURCE) -> list[str]:
    """技能包文件清单（渐进式披露第二层半）：路径 + 字数，不含正文。"""
    return [f"- {name}（{n} 字）" for name, n in source.files(skill_name)]


def skill_file_prompt(skill_name: str, source: SkillSource = DEFAULT_SOURCE) -> str:
    """【技能自带文件】提示块（清单常驻、正文按需），唯一一份；没文件返回空串。

    两个调用方（`read_skill` 的返回、`run_task` 的任务书）共用这一处。必须把 `skill_name`
    写进清单，不写执行器就只能猜。有才带这一段，不写"（无参考文件）"。
    """
    files = skill_file_index_lines(skill_name, source)
    if not files:
        return ""
    return (
        f"\n\n**技能自带文件** —— 要用哪一份就用 `read_skill_file` 读全文："
        f"**`skill_name` 传「{skill_name}」**，`path` 照抄下面的清单"
        "（**不要凭文件名猜内容**）：\n" + "\n".join(files)
    )


# ──────────────────────────── 两个读工具 ────────────────────────────


def make_read_skill(source: SkillSource = DEFAULT_SOURCE):
    """造一个 `read_skill` 工具，落在给定的技能来源上。"""

    @tool
    async def read_skill(skill_name: str) -> str:
        """按需读一个技能的**完整指令**（渐进式披露第二层），附带它自带哪些文件。

        常驻在你眼前的只有技能名和一句话描述。写任务要求前若不确定这个技能
        **交付什么、按什么标准交付**，用它读原文核对 —— **不要凭描述猜**。
        """
        # 住在这里：它同时要用领域对象（技能来源）和磁盘 I/O（本模块），住 `skills.py`
        # 就得函数内 lazy import 本模块（一提就成环）。
        skill = source.get(skill_name)
        if skill is None:
            names = "、".join(s.name for s in source.values())
            return f"没有技能「{skill_name}」。可用：{names}"
        # 不用 `【…】`（段名专用符号）—— 它是工具返回的头，不是段名。
        out = f"**技能「{skill.name}」** —— {skill.description}\n\n{skill.instructions}"
        # 参考文件清单现场扫盘（不缓存在 Skill 上）；提示块是唯一一份（任务书那边用的是
        # 同一个函数），没有自带文件时返回空串。
        out += skill_file_prompt(skill.name, source)
        return out

    return read_skill


def make_read_skill_file(source: SkillSource = DEFAULT_SOURCE):
    """造一个 `read_skill_file` 工具，落在给定的技能来源上。"""

    @tool
    async def read_skill_file(skill_name: str, path: str) -> str:
        """读技能包里某个**文件的全文**（渐进式披露第三层）。

        技能自带的规范 / 模板 / 标准（如《结构模板》《格式规范》《术语表》）——
        路径就是 `SKILL.md` 里写的那个（形如 `references/xxx.md`）。
        清单在 `read_skill` 的返回里（派给执行器的任务书里也有一份）。

        **确定这一份确实用得上再读** —— 别把几份都拉进上下文。
        """
        have = skill_file_index_lines(skill_name, source)
        if not have:
            return f"技能「{skill_name}」没有自带文件。"
        got = source.read_file(skill_name, path)
        if got is None:
            return f"「{skill_name}」里没有文件「{path}」。现有：\n" + "\n".join(have)
        # 不用 `【…】`（段名专用符号）—— 它是工具返回的头，不是段名。
        return f"**{skill_name} / {path}**\n\n{got}"

    return read_skill_file


# ──────────────────────────── 加载 ────────────────────────────


def load_skill_pack(pack_dir: Path) -> tuple[Skill | None, list[str]]:
    """读一个技能包目录 →（`Skill`，说明/警告）。

    返回 `None` 表示"这个目录不是技能包"（没有 `SKILL.md` 或没写 `name`），不是错误。
    参考文件不在这里读，也不在这里记位置 —— 那是 `load_skills_dir` 的事。
    """
    entry = pack_dir / ENTRY_NAME
    if not entry.is_file():
        return None, []
    meta, body = split_frontmatter(entry.read_text(encoding="utf-8"))
    name = str(meta.get("name") or "").strip()
    if not name:
        return None, [f"{pack_dir.name}/{ENTRY_NAME} 的 frontmatter 没写 name，跳过"]
    return (
        Skill(
            name=name,
            description=str(meta.get("description") or "").strip(),
            instructions=body,
        ),
        [],
    )


def load_skills_dir(path: str | Path) -> tuple[LocalSkills, list[str]]:
    """扫一个目录下的所有技能包 →（技能来源，说明/警告）。

    只认子目录里的 `SKILL.md`，散装 `*.md` 不看 —— 一个技能是一个包，散文件没有包的边界。
    每次调用现造一份，不碰任何全局：换一批技能就是换一个来源实例。
    """
    root = Path(path)
    skills: dict[str, Skill] = {}
    pack_dirs: dict[str, Path] = {}
    warn: list[str] = []
    if not root.is_dir():
        return LocalSkills(skills, pack_dirs), [f"技能目录不存在：{root}"]
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        skill, w = load_skill_pack(d)
        warn += w
        if skill is not None:
            skills[skill.name] = skill
            pack_dirs[skill.name] = d
    return LocalSkills(skills, pack_dirs), warn
