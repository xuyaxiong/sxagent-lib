"""产物库：运行期的交付物。它是文件系统的替身，也是上下文隔离的落点。

正文写进库里，只把 ID 交回主 agent，要读正文得显式调 `read_artifact`，所以执行器做出来的
东西不会自己流进主上下文。一条产物记 `title` / `summary` / `content` / `type` / `ext` /
`file`，落盘后补 `path`。

## 不变量

1. 对外只给函数，不给 dict（同 `builtin_skills()`）。表私有，换实现只动这个文件。
2. 类型靠硬派发，不嗅探内容：类型是产出方（模型）在交付时说的，照它存；认不出来存
   `DEFAULT_EXT`，不报错：产物已经做出来了，存不下比存错更糟。
3. 落盘那份和库里那份一字不差，"这是哪一份"由文件名承载（`art_3f9a2c1d_<标题>.md`）。
4. ID 是随机短号（`art_` + 8 位十六进制，`_next_aid`），发之前问一次 `has()`：会话对象
   每请求一个（`with session()`），自增序号挂它就会跨轮重来；随机号不挂任何状态，跨轮、
   跨重启都不会重。
5. 落盘目录只能走 `use_dir()`，直接赋值写错模块不会当场报错。
6. 表、落盘目录两样都挂会话上，不挂模块级。`write_artifact` 按 run 记账靠闭包
   `created`，不靠事后扫消息文本猜。
7. 工具返回的标题不用 `【…】`：那是段名的专用符号，记录层靠"独占行首"切分。
8. `write_artifact` 落地失败**不掀桌**：包成错误结果回给模型（`additional_kwargs.tool_error`
   标记 + `status="error"`），模型可修正后重交；宿主与执行器按标记判成败，不解析文案。"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from sxagent import session, trace

# ──────────────────────────── 产物类型（声明 + 硬性派发） ────────────────────────────

#: 声明里的别名 → 标准扩展名。只有这两个：`markdown` / `htm` 和标准写法指同一件事，
#: 别的类型（`csv` / `json` / `svg`……）照原样当扩展名用 —— 类型是产出方在交付时说的。
FILE_TYPE_ALIASES: dict[str, str] = {"markdown": "md", "htm": "html"}
#: 声明缺失 / 认不出来时存成什么。不报错：产物已经产出来了，存不下比存错更糟。
DEFAULT_EXT = ".md"

#: 认得出的扩展名长什么样：字母数字，1 到 8 位（`md` / `csv` / `docx`…）。
_EXT_RE = re.compile(r"[a-z0-9]{1,8}")


def file_ext(file_type: str) -> str:
    """声明的类型 → 扩展名。类型是产出方（模型）在交付时说的，照它存。

    大小写、带不带点、`markdown` / `htm` 两个别名都认；说不像个扩展名的（空、带空格、
    带路径符号）落 `DEFAULT_EXT` —— 兜底不报错，见它上面那行。
    """
    name = FILE_TYPE_ALIASES.get((file_type or "").strip().lstrip(".").lower(), "")
    if not name:
        name = (file_type or "").strip().lstrip(".").lower()
    return f".{name}" if _EXT_RE.fullmatch(name) else DEFAULT_EXT


def artifact_slug(s: str, n: int = 40) -> str:
    """标题 → 文件名安全的一段（保留中英文，其余压成 `-`）。"""
    return (re.sub(r"[^\w一-鿿]+", "-", s).strip("-") or "untitled")[:n]


# ──────────────────────────── 产物库（= 文件系统替身） ────────────────────────────
#
# 对外只有函数、不给 dict：换实现只动这个文件，调用方只认下面这几个函数。


@runtime_checkable
class ArtifactStore(Protocol):
    """产物放哪 —— 库只认这几个方法，宿主可以用 OSS、数据库或别的什么实现它。

    写（`put`）可以 await：宿主落 OSS / 数据库是网络 IO，阻塞事件循环是更坏的选择。
    读（其余几个）保持同步 —— 库的语义是"读必须廉价"（宿主自己缓存 / 预载），廉价的东西
    不该逼调用方 await。

    ## 不变量

    1. `put` 返回那一刻，东西就得真的在："已写入产物 art_3f9a2c1d"必须指得到。
    2. ID 由**库**发（`art_` + 8 位十六进制这种短形状，模型要照着抄），**跨轮、跨重启都
       不会重**：随机号不挂任何状态，发之前只问一次 `has()`（一个实现服务多个会话时，
       自己按会话分前缀：OSS 分目录、数据库带 thread_id）。
    3. 记录里有 `title` / `summary` / `content` / `type` / `ext` / `file`，落了盘再补 `path`。
       `where()` 给**裸值**（本地路径或 URL），怎么排版是库的事。
    """

    async def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
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

    async def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
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


#: 发号最多试几次 —— 8 位十六进制的空间里连撞三次是不可能事件，真撞上只能是 `has()` 坏了。
_AID_ATTEMPTS = 3


def _next_aid(store: ArtifactStore) -> str:
    """发下一个产物 ID：`art_` + 8 位十六进制（随机号的理由见模块头不变量 4）。

    撞了就重发，但**有上界**：报错比死循环好，死循环会把执行器整趟卡死。
    """
    for _ in range(_AID_ATTEMPTS):
        aid = f"art_{uuid.uuid4().hex[:8]}"
        if not store.has(aid):
            return aid
    raise RuntimeError(f"连发 {_AID_ATTEMPTS} 个产物 ID 都撞了 —— 产物库的 `has()` 有问题")


def use_dir(path: Path | None) -> None:
    """告诉产物库"这个会话"的产物放哪。入口调这个，不要直接赋值那个字段。

    走函数，写错模块会当场 `AttributeError`（启动就报错，不用等人发现）。它设的是当前会话
    的目录 —— 多会话共享进程时各写各的，不互相覆盖。
    """
    session.current().artifact_dir = path


async def _store_into(
    store: ArtifactStore, title: str, summary: str, file_type: str, content: str
) -> str:
    """发号 → 存 → 报一声，返回产物 ID。

    发号在这儿、不在实现里：ID 是模型要照着抄的东西，形状得库说了算。落盘那条日志也在这儿
    ——换一个产物库（OSS），它照样得出现。
    """
    aid = _next_aid(store)
    await store.put(aid, title, summary, file_type, content)
    where = store.where(aid)
    trace.note(
        f"✅ 产物落盘 · {aid}《{title}》（{len(content)} 字）"
        + (f" → {where}" if where else "（本次只进内存，没落盘）"),
        kind="artifact",
    )
    return aid


async def store_artifact(title: str, summary: str, file_type: str, content: str) -> str:
    """把产物落进产物库（库自带那个），返回产物 ID。"""
    return await _store_into(DEFAULT_STORE, title, summary, file_type, content)


def make_write_artifact(created: list[str] | None = None, store: ArtifactStore = DEFAULT_STORE):
    """造一个 `write_artifact` 工具，落在给定的产物库上。

    `created` 非空时顺手把产物 ID 记进去（执行器要回答"这次 run 写了哪些产物"）；用闭包不用
    模块级全局：并行 delegate 时两个执行器会互相覆盖。`store` 是宿主注入的那个，不传就用库
    自带的。
    """

    @tool
    async def write_artifact(
        title: str,
        summary: str,
        file_type: str,
        content: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str | Command:
        """把一份**产物**写进产物库。正文不会再回到主 agent 的上下文，只返回产物 ID。

        落地失败（库拒绝 / 存储故障）会以错误结果**回到你这里**，不会中断这一轮 ——
        看清错在哪、修正后重新调用即可。

        Args:
            title: 产物名。
            summary: **一句话梗概** —— 它是产物对外露出的那一层：主 agent 靠它决定
                要不要读全文、要不要把它传给下一个技能。不写梗概，别人只能读全文。
            file_type: **文件类型**（`md` / `html` / `csv` / `json` / `svg`…… 文本格式的
                扩展名）—— 由**产出方**在交付时一起说，平台据此**硬性按类型保存**：
                说 md 就存 `.md`，说 csv 就存 `.csv`；认不出的按 `md` 存。
            content: 产物正文全文。
        """
        try:
            aid = await _store_into(store, title, summary, file_type, content)
        except Exception as e:
            # 失败不掀桌（掀桌=整轮崩、模型连重试的机会都没有）：包成错误结果回给模型。
            # `tool_error` 是结构标记 —— 宿主与执行器按它判成败，不解析这句文案。
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                f"❌ 交付失败：{e}\n"
                                "（问题修好后可以重新调用 `write_artifact` 交付。）"
                            ),
                            tool_call_id=tool_call_id or "",
                            additional_kwargs={"tool_error": {"reason": str(e)}},
                            status="error",
                        )
                    ]
                }
            )
        if created is not None:
            created.append(aid)
        a = store.get(aid) or {}
        # 返回里带上文件类型和保存地址 —— 用户和后面的代码都要靠它找到这份产物
        at = f"\n位置：{store.where(aid)}" if store.where(aid) else ""
        return f"已写入产物 {aid}（{len(content)} 字，{a.get('ext', '').lstrip('.')}）{at}"

    return write_artifact


def make_read_artifact(store: ArtifactStore = DEFAULT_STORE):
    """造一个 `read_artifact` 工具，读给定的产物库。"""

    @tool
    async def read_artifact(artifact_id: str) -> str:
        """按需读回产物全文（渐进式披露第二层）。

        ⚠️ **读了就进你的上下文**，长产物会一直占着。只有确实要看内容时才读；
        只是把它交给下一个任务，用 `delegate(..., artifacts=[...])` 传 ID 就够了。
        """
        a = store.get(artifact_id)
        # 不用 `【…】`（段名专用符号）—— 它是工具返回的头，不是段名。
        return f"**{a['title']}**\n\n{a['content']}" if a else f"产物 {artifact_id} 不存在"

    return read_artifact


def artifact_index_lines(store: ArtifactStore = DEFAULT_STORE) -> list[str]:
    """产物清单（渐进式披露第一层）：ID / 名称 / 字数 / 类型 / 梗概 / 位置，不含正文。

    位置必须在这一层：主 agent 要答"东西在哪"，而投回那条路跑完补的是**回执**（`graph._deliver`，
    与 `delegate` 的回执同一套渲染），台账的完成行又只有 ID。
    放在这一层而不是台账：台账是每轮注入的索引、必须短，这里是按需的；形状和回执
    【产出的产物】一致（ID ＋ 梗概一行，位置另起一行缩进 2 格）。

    标签用"位置"不用"文件"：库自带的是本地路径，宿主可能是 OSS 的 URL —— 标签只有一份，
    得两边都成立（`where()` 给裸值，怎么排版是这儿的事）。
    """
    return [
        f"- {aid}《{a['title']}》（{len(a['content'])} 字，{a.get('ext', '.md').lstrip('.')}）"
        f"—— {a.get('summary') or '（无梗概）'}"
        + (f"\n  位置：{store.where(aid)}" if store.where(aid) else "")
        for aid, a in store.items()
    ]


def make_list_artifacts(store: ArtifactStore = DEFAULT_STORE):
    """造一个 `list_artifacts` 工具，列给定的产物库。"""

    @tool
    async def list_artifacts() -> str:
        """列出本会话已交付的产物：ID、名称、字数、梗概、**在哪**。**不含正文。**

        要判断"该把哪份产物交给下一个任务"、或**要答"东西在哪"**时用它；
        正文按需用 `read_artifact` 读。
        """
        if not store.items():
            return "本会话还没有产物。"
        return "本会话已交付的产物：\n" + "\n".join(artifact_index_lines(store))

    return list_artifacts
