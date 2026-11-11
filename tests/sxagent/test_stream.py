"""上游确定性拒绝的判定：4xx（除 408/429）才叫拒绝，其余都当抖动。

兑现 `stream.upstream_rejected`。它定的是故障的类（`FaultKind`），回执、台账、决策记录四处
引导句按它渲染 —— 判错两头都贵：把抖动当拒绝，主 agent 会去改一份本来没问题的内容；把拒绝
当抖动，它会原样重发一份上游一定会再拒的东西。

按 `status_code` 判、不认识 provider 的类型（换一家 SDK 就失效），所以这里只用带
`status_code` 的异常验。
"""

from __future__ import annotations

from sxagent.stream import upstream_rejected


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
