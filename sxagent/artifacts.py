"""产物库：运行期的交付物。它是文件系统的替身，也是上下文隔离的落点。

正文写进库里，只把 ID 交回主 agent，要读正文得显式调 `read_artifact`，所以执行器做出来的
东西不会自己流进主上下文。一条产物记 `title` / `summary` / `content` / `type` / `ext` /
`file`，落盘后补 `path`。

## 不变量

1. 对外只给函数，不给 dict（同 `registry`）。表私有，换实现只动这个文件。
2. 类型靠硬派发，不嗅探内容。认不出来存 `DEFAULT_EXT`，不报错：产物已经做出来了，存不下比
   存错更糟。
3. 落盘那份和库里那份一字不差，"这是哪一份"由文件名承载（`art_001_<标题>.md`）。
4. ID 用 `itertools.count()` 发，不用 `len(表) + 1`：后者先查后写，中间有窗口，产物会静默丢。
5. 落盘目录只能走 `use_dir()`，直接赋值写错模块不会当场报错。
6. 表、发号器、落盘目录三样都挂会话上，不挂模块级。`write_artifact` 按 run 记账靠闭包
   `created`，不靠事后扫消息文本猜。
7. 工具返回的标题不用 `【…】`：那是段名的专用符号，记录层靠"独占行首"切分。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from langchain.tools import tool

from sxagent import session, trace

# ──────────────────────────── 产物类型（声明 + 硬性派发） ────────────────────────────

#: 声明的文件类型 → 扩展名 —— 平台据此硬性派发保存。不嗅探内容：类型是产出方在交付时说的。
#: 要支持新类型就往这张表里加一行 —— 派发规则只有这一处。
FILE_TYPES: dict[str, str] = {
    "md": ".md",
    "markdown": ".md",
    "html": ".html",
    "htm": ".html",
}
#: 声明缺失 / 认不出来时存成什么。不报错：产物已经产出来了，存不下比存错更糟。
DEFAULT_EXT = ".md"


def file_ext(file_type: str) -> str:
    """声明的类型 → 扩展名（大小写、带不带点、`markdown`/`htm` 都认）。"""
    return FILE_TYPES.get((file_type or "").strip().lstrip(".").lower(), DEFAULT_EXT)


def artifact_slug(s: str, n: int = 40) -> str:
    """标题 → 文件名安全的一段（保留中英文，其余压成 `-`）。"""
    return (re.sub(r"[^\w一-鿿]+", "-", s).strip("-") or "untitled")[:n]


# ──────────────────────────── 产物库（= 文件系统替身） ────────────────────────────
#
# 对外只有函数、不给 dict：换实现只动这个文件，调用方只认下面这几个函数。


@runtime_checkable
class ArtifactStore(Protocol):
    """产物放哪 —— 库只认这几个方法，宿主可以用 OSS、数据库或别的什么实现它。

    ## 不变量

    1. `put` 返回那一刻，东西就得真的在："已写入产物 art_001"必须指得到。
    2. ID 由**库**发（`art_001` 这种短形状，模型要照着抄），只在**本会话内**唯一 ——
       一个实现服务多个会话时，自己按会话分前缀（OSS 分目录、数据库带 thread_id）。
    3. 记录里有 `title` / `summary` / `content` / `type` / `ext` / `file`，落了盘再补 `path`。
       `where()` 给**裸值**（本地路径或 URL），怎么排版是库的事。
    """

    def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
        """存一条产物。ID 由调用方（库）发好传进来。"""

    def get(self, aid: str) -> dict[str, str] | None:
        """按 ID 取记录（含正文）；没有就 None。"""

    def has(self, aid: str) -> bool:
        """这条在不在库里（"在不在"与"内容是什么"是两个问题）。"""

    def items(self) -> list[tuple[str, dict[str, str]]]:
        """（ID, 记录）的快照 —— 遍历用，加进来的新产物不会中途冒出来。"""

    def where(self, aid: str) -> str:
        """它在哪（本地路径 / URL）；没有落盘就是空串。"""

    def clear(self) -> None:
        """清空 —— 跑批每个 case 开始前调它。"""


class MemoryArtifacts:
    """库自带的产物库：内存表 + 本地落盘。

    数据挂在**会话**上（`session.current()`），所以这个实例本身无状态 —— 多会话共用一个实例
    也不会串，各是各的。
    """

    def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
        ext = file_ext(file_type)
        rec: dict[str, str] = {
            "title": title,
            "summary": summary,
            "content": content,
            "type": file_type,
            "ext": ext,
            "file": f"{aid}_{artifact_slug(title)}{ext}",
        }
        _dir = session.current().artifact_dir
        if _dir is not None:
            _dir.mkdir(parents=True, exist_ok=True)
            p = _dir / rec["file"]
            # 落盘 = 库里那份，一字不加：正文就是正文，"这是哪一份"由文件名承载。
            p.write_text(rec["content"], encoding="utf-8")
            rec["path"] = str(p)
        session.current().artifacts[aid] = rec

    def get(self, aid: str) -> dict[str, str] | None:
        """返回的是记录本身（不复制）—— 读没问题；写请走 `put`。"""
        return session.current().artifacts.get(aid)

    def has(self, aid: str) -> bool:
        return aid in session.current().artifacts

    def items(self) -> list[tuple[str, dict[str, str]]]:
        return list(session.current().artifacts.items())

    def where(self, aid: str) -> str:
        rec = session.current().artifacts.get(aid)
        return (rec or {}).get("path", "")

    def clear(self) -> None:
        session.current().artifacts.clear()


#: 库自带的那个实现 —— 入口不注入的话，一切都走它。
DEFAULT_STORE: ArtifactStore = MemoryArtifacts()


def get(aid: str) -> dict[str, str] | None:
    """按 ID 取产物记录（含正文）；没有就 None。"""
    return DEFAULT_STORE.get(aid)


def has(aid: str) -> bool:
    """这条产物在不在库里。"""
    return DEFAULT_STORE.has(aid)


def ids() -> list[str]:
    """全部产物 ID（顺序 = 写入顺序）。"""
    return [aid for aid, _ in DEFAULT_STORE.items()]


def items() -> list[tuple[str, dict[str, str]]]:
    """（ID, 记录）的快照。"""
    return DEFAULT_STORE.items()


def count() -> int:
    return len(DEFAULT_STORE.items())


def chars() -> int:
    """产物库正文合计字数 —— 隔离效果的读数之一（`snapshot` / `__main__` 用它）。"""
    return sum(len(a["content"]) for _, a in DEFAULT_STORE.items())


def clear() -> None:
    """清空 —— 跑批每个 case 开始前调它（`app.runtime.install`）。"""
    DEFAULT_STORE.clear()


# 产物 ID 走 `Session.artifact_seq` 的发号器。用 `len(表) + 1` 有窗口：两个执行器会拿到
# 同一个 ID，产物静默丢失。


def use_dir(path: Path | None) -> None:
    """告诉产物库"这个会话"的产物放哪。入口调这个，不要直接赋值那个字段。

    走函数，写错模块会当场 `AttributeError`（启动就报错，不用等人发现）。它设的是当前会话
    的目录 —— 多会话共享进程时各写各的，不互相覆盖。
    """
    session.current().artifact_dir = path


def store_artifact(title: str, summary: str, file_type: str, content: str) -> str:
    """把产物落进产物库，返回产物 ID。顺手落盘（设了落盘目录的话）。

    当场落盘：产物是交付物，用户当场就要能打开 —— "已写入产物 art_001"必须指得到文件。
    发号在这儿、不在实现里：ID 是模型要照着抄的东西，形状得库说了算。
    """
    aid = f"art_{next(session.current().artifact_seq):03d}"
    DEFAULT_STORE.put(aid, title, summary, file_type, content)
    # 落盘那一刻说一声。写产物是全程最长的一段静默："开始"和"进度"由传输层报，
    # "结束"必须由这里报 —— 报的是结构事实（ID / 标题 / 字数 / 在哪），不是从正文里读的。
    # 放在这儿而不是实现里：换一个产物库（OSS），这条日志照样得出现。
    where = DEFAULT_STORE.where(aid)
    trace.note(
        f"✅ 产物落盘 · {aid}《{title}》（{len(content)} 字）"
        + (f" → {where}" if where else "（本次只进内存，没落盘）"),
        kind="artifact",
    )
    return aid


def make_write_artifact(created: list[str] | None = None):
    """造一个 `write_artifact` 工具。`created` 非空时，顺手把产物 ID 记进去。

    要"造"而不是共用一个：执行器要回答"这次 run 写了哪些产物"（`outcome.artifacts`）——
    改成记账，谁写的谁自己记。用闭包、不用模块级全局：并行 delegate 时两个执行器会互相
    覆盖 —— 闭包 = 按 run 归属，并发窗口结构上不存在。
    """

    @tool
    async def write_artifact(title: str, summary: str, file_type: str, content: str) -> str:
        """把一份**产物**写进产物库。正文不会再回到主 agent 的上下文，只返回产物 ID。

        Args:
            title: 产物名。
            summary: **一句话梗概** —— 它是产物对外露出的那一层：主 agent 靠它决定
                要不要读全文、要不要把它传给下一个技能。不写梗概，别人只能读全文。
            file_type: **文件类型**（`md` / `html`）—— 由**产出方**在交付时一起说，
                平台据此**硬性按类型保存**：说 md 就存 `.md`，说 html 就存 `.html`。
            content: 产物正文全文。
        """
        aid = store_artifact(title, summary, file_type, content)
        if created is not None:
            created.append(aid)
        a = session.current().artifacts[aid]
        # 返回里带上文件类型和保存地址 —— 用户和后面的代码都要靠它找到这份产物
        where = f"\n文件：{a['path']}" if a.get("path") else ""
        return f"已写入产物 {aid}（{len(content)} 字，{a['ext'].lstrip('.')}）{where}"

    return write_artifact


#: 主 agent 用的那个（它不需要按 run 记账 —— 它写的东西直接进自己的对话）
write_artifact = make_write_artifact()


@tool
async def read_artifact(artifact_id: str) -> str:
    """按需读回产物全文（渐进式披露第二层）。

    ⚠️ **读了就进你的上下文**，长产物会一直占着。只有确实要看内容时才读；
    只是把它交给下一个任务，用 `delegate(..., artifacts=[...])` 传 ID 就够了。
    """
    a = get(artifact_id)
    # 不用 `【…】`（段名专用符号）—— 它是工具返回的头，不是段名。
    return f"**{a['title']}**\n\n{a['content']}" if a else f"产物 {artifact_id} 不存在"


def artifact_index_lines() -> list[str]:
    """产物清单（渐进式披露第一层）：ID / 名称 / 字数 / 类型 / 梗概 / 落盘路径，不含正文。

    路径必须在这一层：主 agent 要答"文件在哪"，而它在投递重跑（`feed`）那条路上没有回执
    （回执只属于 `delegate` 工具调用），台账的完成行又只有 ID。
    放在这一层而不是台账：台账是每轮注入的索引、必须短，这里是按需的；形状和回执
    【产出的产物】一致（ID ＋ 梗概一行，路径另起一行缩进 2 格）。
    """
    return [
        f"- {aid}《{a['title']}》（{len(a['content'])} 字，{a.get('ext', '.md').lstrip('.')}）"
        f"—— {a.get('summary') or '（无梗概）'}"
        + (f"\n  文件：{a['path']}" if a.get("path") else "")
        for aid, a in session.current().artifacts.items()
    ]


@tool
async def list_artifacts() -> str:
    """列出本会话已交付的产物：ID、名称、字数、梗概、**落盘路径**。**不含正文。**

    要判断"该把哪份产物交给下一个任务"、或**要答"文件在哪"**时用它；
    正文按需用 `read_artifact` 读。
    """
    if not session.current().artifacts:
        return "本会话还没有产物。"
    return "本会话已交付的产物：\n" + "\n".join(artifact_index_lines())
