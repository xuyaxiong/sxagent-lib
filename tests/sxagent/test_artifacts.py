"""产物库：库自带的实现满足 `ArtifactStore` 接口，行为按接口上写的不变量来。

接口是给宿主用的（它可以实现成 OSS、数据库），所以"库自带的这个实现也满足它"要钉住 ——
不然宿主照接口写完了，库自己却对不上。
"""

from __future__ import annotations

import re
from pathlib import Path

from sxagent.artifacts import DEFAULT_STORE, ArtifactStore, MemoryArtifacts, make_write_artifact


class _DurableStore:
    """跨会话存活的产物库 —— 模拟宿主（OSS / 数据库）。

    `with session()` 每请求一个，所以它是唯一记得住上一轮的那一份 —— 跨轮可见靠它。
    """

    def __init__(self) -> None:
        self.table: dict[str, dict[str, str]] = {}

    async def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
        self.table[aid] = {
            "title": title,
            "summary": summary,
            "content": content,
            "type": file_type,
            "ext": ".md",
            "file": f"{aid}.md",
        }

    def get(self, aid: str) -> dict[str, str] | None:
        return self.table.get(aid)

    def has(self, aid: str) -> bool:
        return aid in self.table

    def items(self) -> list[tuple[str, dict[str, str]]]:
        return list(self.table.items())

    def where(self, aid: str) -> str:
        return ""

    def clear(self) -> None:
        self.table.clear()


def test_default_store_satisfies_the_interface() -> None:
    assert isinstance(DEFAULT_STORE, ArtifactStore)
    assert isinstance(MemoryArtifacts(), ArtifactStore)
    assert isinstance(_DurableStore(), ArtifactStore)


async def test_put_get_where_and_clear() -> None:
    """`put` 返回那一刻东西就得在；`where` 给的是裸值（本地路径）。"""
    store = MemoryArtifacts()
    store.clear()

    await store.put("art_001", "哨兵标题", "哨兵梗概", "md", "哨兵正文")

    assert store.has("art_001")
    assert store.get("art_001")["content"] == "哨兵正文"
    assert store.items() == [("art_001", store.get("art_001"))]
    assert store.where("art_001") == ""  # 没设落盘目录 → 只进内存

    store.clear()
    assert store.items() == []


async def test_put_writes_the_file_when_a_dir_is_set(tmp_path: Path) -> None:
    from sxagent import session

    with session.session("test-artifacts"):
        session.current().artifact_dir = tmp_path
        await MemoryArtifacts().put("art_001", "哨兵标题", "哨兵梗概", "md", "哨兵正文")
        where = MemoryArtifacts().where("art_001")

    assert Path(where).read_text(encoding="utf-8") == "哨兵正文"
    assert Path(where).name.startswith("art_001_")


async def test_ids_are_unique_across_requests() -> None:
    """web 用法是每请求开一次会话：两轮发出的号不许重。

    重了的话第二轮那份会顶掉第一轮 —— 而主 agent 的上下文和台账是跨轮存活的，两轮之后
    就指不清哪份是哪份。
    """
    from sxagent import session

    store = _DurableStore()
    created: list[str] = []
    with session.session("conv-1"):
        await make_write_artifact(created, store=store).ainvoke(
            {"title": "第一份", "summary": "梗概", "file_type": "md", "content": "正文一"}
        )
    with session.session("conv-1"):  # 第二个请求：新会话对象，同一个产物库
        await make_write_artifact(created, store=store).ainvoke(
            {"title": "第二份", "summary": "梗概", "file_type": "md", "content": "正文二"}
        )

    assert len(set(created)) == 2
    assert all(re.fullmatch(r"art_[0-9a-f]{8}", aid) for aid in created)


def test_declared_type_is_taken_as_the_extension() -> None:
    """类型是模型交付时说的，照它存：`csv` 就是 `.csv`，不再是静默的 `.md`。"""
    from sxagent.artifacts import file_ext

    assert file_ext("csv") == ".csv"
    assert file_ext("JSON") == ".json"
    assert file_ext(".svG") == ".svg"
    assert file_ext("markdown") == ".md"  # 别名照认
    assert file_ext("htm") == ".html"


class _FailingStore(MemoryArtifacts):
    """落库必失败的产物库 —— 测"交付失败不掀桌"这条。"""

    async def put(self, aid: str, title: str, summary: str, file_type: str, content: str) -> None:
        raise ValueError("产物正文不能为空")


async def test_failed_delivery_returns_to_the_model_instead_of_crashing(
    monkeypatch,
) -> None:
    """落地失败不掀桌：包成带 `tool_error` 标记的错误结果回给模型，模型看得到、整轮照走。

    掀桌版本（实测过）：异常直接冒出 `astream_events`、消息里没有 ToolMessage、
    模型连重试的机会都没有 —— 交付失败的"对模型可见"就成了假话。
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from sxagent import graph
    from tests.fakes import ToolCallingFakeModel

    fake = ToolCallingFakeModel()
    monkeypatch.setattr("sxagent.agent.get_model", lambda *a, **k: fake)
    app = graph.build(artifacts=_FailingStore())
    config = {"configurable": {"thread_id": "test-write-fail"}}

    await app.ainvoke({"messages": [HumanMessage("哨兵-A 的问题")]}, config)

    msgs = app.get_state(config).values["messages"]
    errored = [
        m
        for m in msgs
        if isinstance(m, ToolMessage) and (m.additional_kwargs or {}).get("tool_error")
    ]
    assert errored, "交付失败没有以带标记的错误结果回到消息里"
    assert "交付失败" in str(errored[0].content)
    assert errored[0].tool_call_id, "错误结果的 tool_call_id 丢了（runtime 没注入？）"
    # 模型被再叫了一次（看到错误结果后收尾）—— 那一轮没有崩
    assert isinstance(msgs[-1], AIMessage) and "收尾" in str(msgs[-1].content)


def test_unusable_type_falls_back_to_md() -> None:
    """说不像个扩展名的就落 md 兜底 —— 不报错：产物已经做出来了，存不下比存错更糟。"""
    from sxagent.artifacts import file_ext

    assert file_ext("") == ".md"
    assert file_ext("不要这种") == ".md"
    assert file_ext("a/b") == ".md"
    assert file_ext("way-too-long-type") == ".md"
