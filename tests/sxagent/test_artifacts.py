"""产物库：库自带的实现满足 `ArtifactStore` 接口，行为按接口上写的不变量来。

接口是给宿主用的（它可以实现成 OSS、数据库），所以"库自带的这个实现也满足它"要钉住 ——
不然宿主照接口写完了，库自己却对不上。
"""

from __future__ import annotations

from pathlib import Path

from sxagent.artifacts import DEFAULT_STORE, ArtifactStore, MemoryArtifacts


def test_default_store_satisfies_the_interface() -> None:
    assert isinstance(DEFAULT_STORE, ArtifactStore)
    assert isinstance(MemoryArtifacts(), ArtifactStore)


def test_put_get_where_and_clear(tmp_path: Path) -> None:
    """`put` 返回那一刻东西就得在；`where` 给的是裸值（本地路径）。"""
    store = MemoryArtifacts()
    store.clear()

    store.put("art_001", "哨兵标题", "哨兵梗概", "md", "哨兵正文")

    assert store.has("art_001")
    assert store.get("art_001")["content"] == "哨兵正文"
    assert store.items() == [("art_001", store.get("art_001"))]
    assert store.where("art_001") == ""  # 没设落盘目录 → 只进内存

    store.clear()
    assert store.items() == []


def test_put_writes_the_file_when_a_dir_is_set(tmp_path: Path) -> None:
    from sxagent import session

    with session.session("test-artifacts"):
        session.current().artifact_dir = tmp_path
        MemoryArtifacts().put("art_001", "哨兵标题", "哨兵梗概", "md", "哨兵正文")
        where = MemoryArtifacts().where("art_001")

    assert Path(where).read_text(encoding="utf-8") == "哨兵正文"
    assert Path(where).name.startswith("art_001_")
