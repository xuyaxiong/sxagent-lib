"""技能包：文件夹形态的技能和它带的参考文件，管磁盘 IO 与格式。

    my-skill/
    ├── SKILL.md      ← 入口，只有这一个名字；frontmatter 给 name/description，正文即 instructions
    ├── references/   ← 可选，没有也行
    └── assets/       ← 目录名随作者，整个包都能读

加载包（`load_skill_pack` / `load_skills_dir`）、列和读包内文件（`list_skill_files` /
`load_skill_file`）、两个给 LLM 的读工具（`read_skill` / `read_skill_file`）、任务书用的提示块
（`skill_file_prompt`）都在这里。`registry` 管"技能是什么"，这里管"包在哪、里面有什么"。

## 不变量

1. 包里的文件不进 `Skill` 对象：磁盘上读得到，装进对象是多余的一份。`PACK_DIRS` 只记位置，
   清单与正文每次现场扫盘。
2. 读不读、读哪份是 LLM 的决定，这里只提供"有哪些"和"读一份"。
3. 只认 `SKILL.md`，不做"跟目录同名"的兼容：两种写法并存就得靠猜。子目录名不限定。
4. 收白名单扩展名的文本文件（`TEXT_EXTS`），不收二进制、超大文件、隐藏目录，在"列"这一步
   就过滤掉。
5. `read_skill_file` 只认 `list_skill_files` 报得出来的东西：路径穿越、二进制、超大文件全被
   这一条挡住，不需要第二套路径检查。
6. 清单必须带 `skill_name`：`read_skill_file` 的第一个参数是技能名。
7. 提示块只有一份（`skill_file_prompt`）：两个调用方都调它，别各拼一遍。"""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain.tools import tool

from sxagent.registry import registry
from sxagent.skills import Skill

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

#: 技能名 → 它的包目录，加载时记下，读参考文件时按它去找。
#: 存的是位置不是内容，所以不是 `Skill` 的字段。
PACK_DIRS: dict[str, Path] = {}


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


# ──────────────────────────── 技能包里的文件：列 + 读 ────────────────────────────


def list_skill_files(skill_name: str) -> list[tuple[str, int]]:
    """技能包里的文本文件 → `[(相对包根的路径, 字数), ...]`。现场扫盘，不缓存。

    排除入口 `SKILL.md`、隐藏目录、非文本扩展名、超大文件。没有技能包信息或目录里没东西，
    返回空列表：没有就是没有，不是错误。
    """
    pack = PACK_DIRS.get(skill_name)
    if pack is None:
        return []
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


def skill_file_index_lines(skill_name: str) -> list[str]:
    """技能包文件清单（渐进式披露第二层半）：路径 + 字数，不含正文。"""
    return [f"- {name}（{n} 字）" for name, n in list_skill_files(skill_name)]


def skill_file_prompt(skill_name: str) -> str:
    """【技能自带文件】提示块（清单常驻、正文按需），唯一一份；没文件返回空串。

    两个调用方（`read_skill` 的返回、`run_task` 的任务书）共用这一处。必须把 `skill_name`
    写进清单，不写执行器就只能猜。有才带这一段，不写"（无参考文件）"。
    """
    files = skill_file_index_lines(skill_name)
    if not files:
        return ""
    return (
        f"\n\n**技能自带文件** —— 要用哪一份就用 `read_skill_file` 读全文："
        f"**`skill_name` 传「{skill_name}」**，`path` 照抄下面的清单"
        "（**不要凭文件名猜内容**）：\n" + "\n".join(files)
    )


def load_skill_file(skill_name: str, path: str) -> str | None:
    """读技能包里某个文件的全文；不在清单里就返回 `None`。`path` 相对技能包根。"""
    if path not in {name for name, _ in list_skill_files(skill_name)}:
        return None
    return (PACK_DIRS[skill_name] / path).read_text(encoding="utf-8", errors="replace")


@tool
async def read_skill(skill_name: str) -> str:
    """按需读一个技能的**完整指令**（渐进式披露第二层），附带它自带哪些文件。

    常驻在你眼前的只有技能名和一句话描述。写任务要求前若不确定这个技能
    **交付什么、按什么标准交付**，用它读原文核对 —— **不要凭描述猜**。
    """
    # 住在这里：它同时要用领域对象（`registry`）和磁盘 I/O（本模块），住 `skills.py`
    # 就得函数内 lazy import 本模块（一提就成环）。
    skill = registry.get(skill_name)
    if skill is None:
        return f"没有技能「{skill_name}」。可用：{'、'.join(registry.names())}"
    # 不用 `【…】`（段名专用符号）—— 它是工具返回的头，不是段名。
    out = f"**技能「{skill.name}」** —— {skill.description}\n\n{skill.instructions}"
    # 参考文件清单现场扫盘（不缓存在 Skill 上）；提示块是唯一一份（任务书那边用的是
    # 同一个函数），没有自带文件时返回空串。
    out += skill_file_prompt(skill.name)
    return out


@tool
async def read_skill_file(skill_name: str, path: str) -> str:
    """读技能包里某个**文件的全文**（渐进式披露第三层）。

    技能自带的规范 / 模板 / 标准（如《结构模板》《格式规范》《术语表》）——
    路径就是 `SKILL.md` 里写的那个（形如 `references/xxx.md`）。
    清单在 `read_skill` 的返回里（派给执行器的任务书里也有一份）。

    **确定这一份确实用得上再读** —— 别把几份都拉进上下文。
    """
    have = skill_file_index_lines(skill_name)
    if not have:
        return f"技能「{skill_name}」没有自带文件。"
    got = load_skill_file(skill_name, path)
    if got is None:
        return f"「{skill_name}」里没有文件「{path}」。现有：\n" + "\n".join(have)
    # 不用 `【…】`（段名专用符号）—— 它是工具返回的头，不是段名。
    return f"**{skill_name} / {path}**\n\n{got}"


# ──────────────────────────── 加载 ────────────────────────────


def clear() -> None:
    """清空包目录表 —— 换一批技能之前调它。

    `registry.testing.reset()` 只管技能表，管不到这里；不清的话，上一个 case 的技能包目录
    会留在表里（同进程跑多 case 时，`read_skill_file` 还能按旧路径读到已经换掉的包）。
    """
    PACK_DIRS.clear()


def load_skill_pack(pack_dir: Path) -> tuple[Skill | None, list[str]]:
    """读一个技能包目录 →（`Skill`，说明/警告）。

    返回 `None` 表示"这个目录不是技能包"（没有 `SKILL.md` 或没写 `name`），不是错误。
    参考文件不在这里读，只记下包目录。
    """
    entry = pack_dir / ENTRY_NAME
    if not entry.is_file():
        return None, []
    meta, body = split_frontmatter(entry.read_text(encoding="utf-8"))
    name = str(meta.get("name") or "").strip()
    if not name:
        return None, [f"{pack_dir.name}/{ENTRY_NAME} 的 frontmatter 没写 name，跳过"]
    PACK_DIRS[name] = pack_dir
    return (
        Skill(
            name=name,
            description=str(meta.get("description") or "").strip(),
            instructions=body,
        ),
        [],
    )


def load_skills_dir(path: str | Path) -> tuple[dict[str, Skill], list[str]]:
    """扫一个目录下的所有技能包 →（`{name: Skill}`，说明/警告）。

    只认子目录里的 `SKILL.md`，散装 `*.md` 不看 —— 一个技能是一个包，散文件没有包的边界。
    """
    root = Path(path)
    skills: dict[str, Skill] = {}
    warn: list[str] = []
    if not root.is_dir():
        return skills, [f"技能目录不存在：{root}"]
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        skill, w = load_skill_pack(d)
        warn += w
        if skill is not None:
            skills[skill.name] = skill
    return skills, warn
