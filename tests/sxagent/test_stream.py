"""上游确定性拒绝的判定：4xx（除 408/429）才叫拒绝，其余都当抖动。

兑现 `stream.upstream_rejected`。它定的是故障的类（`FaultKind`），回执、台账、决策记录四处
引导句按它渲染 —— 判错两头都贵：把抖动当拒绝，主 agent 会去改一份本来没问题的内容；把拒绝
当抖动，它会原样重发一份上游一定会再拒的东西。

按 `status_code` 判、不认识 provider 的类型（换一家 SDK 就失效），所以这里只用带
`status_code` 的异常验。

另有一条写产物的进度：只有 `write_artifact` 报（别的工具的长参数报出来像是"正在交付"，误导）。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessageChunk

from sxagent import trace
from sxagent.stream import _stream_agent, upstream_rejected


class _UpstreamError(Exception):
    def __init__(self, status_code: object) -> None:
        super().__init__(f"status_code={status_code!r}")
        self.status_code = status_code


def test_only_4xx_counts_as_rejected() -> None:
    assert upstream_rejected(_UpstreamError(400)) is True
    assert upstream_rejected(_UpstreamError(404)) is True
    # 408 超时、429 限流：再来一次也许就好了，不算拒绝
    assert upstream_rejected(_UpstreamError(408)) is False
    assert upstream_rejected(_UpstreamError(429)) is False
    # 5xx 和"连不上"这类没有状态码的：抖动
    assert upstream_rejected(_UpstreamError(503)) is False
    assert upstream_rejected(Exception("连不上")) is False


class _ChunkGraph:
    """假图：只吐 `messages` 流（`_stream_agent` 要的形状），驱动参数累计。"""

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks

    async def astream(
        self, state: object, config: object, stream_mode: object = None
    ) -> AsyncGenerator[tuple[str, tuple[object, dict[str, object]]], None]:
        for chunk in self._chunks:
            yield "messages", (chunk, {"langgraph_node": "model"})


def _arg_chunk(tool: str, size: int) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": tool, "args": "x" * size, "id": "c1", "index": 0}],
    )


async def _writing_events(chunks: list[object]) -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []
    subscription = trace.subscribe(seen.append)
    try:
        async for _ in _stream_agent(_ChunkGraph(chunks), {}, None, "哨兵", trace.AGENT):
            pass
    finally:
        subscription.cancel()
    return [e for e in seen if e.get("kind") == "writing"]


async def test_writing_progress_only_reports_write_artifact() -> None:
    """只报写产物：主 agent 拼 delegate 的长任务描述不报 —— 报出来像是"正在交付"，误导。"""
    assert await _writing_events([_arg_chunk("delegate", 500)]) == []
    assert await _writing_events([_arg_chunk("ask_user", 500)]) == []

    events = await _writing_events([_arg_chunk("write_artifact", 500)])
    assert len(events) == 1, events
    assert events[0]["tool"] == "write_artifact", "工具名要走结构字段，展示层不解析文案"
    assert "已收到 500 字" in str(events[0]["text"])
    # 时间是"写这个产物的累计"（零点=这一趟开始），走结构字段给展示层
    assert isinstance(events[0]["written_s"], float)
