"""跑 agent 图：流式驱动，抖动重试，外加区分上游的确定性拒绝。

`stream_agent` 把每一步、每个字都发成事件；重试和拒绝分类是传输层的两件事。

    agent 图 ──stream_agent──▶ 事件总线 ──▶ 记录器 / 展示层

## 不变量

1. 只管跑，不管记。发事件走总线，落盘归 `recorder`，本模块不碰文件。
2. 重试只在"第一步还没走完"时。从头重跑，state 干净，副作用不干净。
3. 确定性拒绝不重试：4xx（除 408/429）是上游对这份内容的拒绝，原样重发还会被拒。别把它搬去
   `llm.safe_structured`——那条路上的 400 是它自己的抖动（比如 DashScope 的 grammar 编译
   超时），分类掉就把唯一有效的重试掐了。"""

from __future__ import annotations

import time

from sxagent import trace


def stream_agent(graph, state, config, who: str, *, role: str):
    """跑一个 agent 图，每一步、每个字都录下来。

    Yields：每个 super-step 的完整 state。第一个 chunk 是输入 state 本身，消费方别把它当
    成"走了一步"，重试边界靠这条。

    只发 `node='model'` 的 chunk：工具内部的结构化调用流出来是裸 JSON，既不可读、又和已
    解开的展示重复。多个执行器并行时事件会交错，所以 `who` 要能分辨是谁。
    """
    # 校验在生成器外面：漏传 / 传错立刻报错，而不是等第一次 next()。
    return _stream_with_retry(graph, state, config, who, trace.check_role(role, "stream_agent"))


def upstream_rejected(e: BaseException) -> bool:
    """这个异常是不是「上游确定性拒绝」—— 4xx（除 408/429）？

    是的话原样重发会再被拒，重试只是白烧一次。按 `status_code` 判、不用 provider 的类型：
    这一层不认识 openai，换一家 SDK 就失效。说不准就当抖动，照旧重试一次。
    """
    code = getattr(e, "status_code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in (408, 429)


async def _stream_with_retry(graph, state, config, who: str, role: str):
    """跑 `_stream_agent`，基础设施抖动就重试一次（只在第一步之前），仍失败才抛。

    只重试"还没走完第一步"的情况：从头重跑 state 干净、副作用不干净，第一步之后抖就
    如实抛。降级不在这层，两条链怎么退化不一样；重试前发 `retry` 事件，已流出去的字不收回。
    """
    #: 已收到的 values chunk 数。第一个 chunk 是"输入 state 本身"，
    #: 所以"至少一步动过 state"等于数到 2。
    chunks_seen = 0
    for attempt in (1, 2):
        try:
            async for chunk in _stream_agent(graph, state, config, who, role):
                chunks_seen += 1
                yield chunk
            return
        except Exception as e:  # noqa: BLE001 —— 抖动的种类不可穷举（网络/超时/限流/网关）
            # 确定性拒绝不重试：原样重发会再被拒，重试只是白烧一次。照实抛给调用方，
            # 由它按 `fault_kind="rejected"` 说人话。
            if attempt == 2 or chunks_seen >= 2 or upstream_rejected(e):
                raise
            # 走 `trace.note()` 而不直调 `_publish`：直调并写死 kind 会绕过 `_check_kind`，
            # 改名 / 打错时校验抓不到。
            trace.note(
                f"⚠️ {who} 这一轮**基础设施抖动**（{type(e).__name__}: {e}）"
                "—— 这就重试一次（上面流出去的那半段作废）",
                kind="retry",
            )


#: 写产物的进度。模型写产物时输出的是工具调用参数，`_chunk_text` 只认 content，那几分钟
#: 一个字都上不了屏，所以这里按参数累计字数报进度。不点名工具：小工具几十字不报，
#: `write_artifact` 一写几千字会报。
_WRITE_REPORT_AT = 200  # 参数超过这么多字才开始报
_WRITE_REPORT_EVERY = 1000  # 之后每多这么多字报一次
_WRITE_REPORT_EVERY_S = 20  # 或者每这么久报一次（字少但很慢时）


def _tool_arg_delta(chunk) -> tuple[str, int]:
    """这一片 chunk 里的工具参数增量，返回 `(工具名, 新增字数)`。

    只数字数不看内容：那是 JSON 片段，流上屏是噪音，而且它不是产物。
    """
    name = ""
    n = 0
    for tc in getattr(chunk, "tool_call_chunks", None) or []:
        if isinstance(tc, dict):
            nm, ar = tc.get("name"), tc.get("args")
        else:
            nm, ar = getattr(tc, "name", None), getattr(tc, "args", None)
        if nm:
            name = str(nm)
        n += len(str(ar or ""))
    return name, n


async def _stream_agent(graph, state, config, who: str, role: str):
    buf = ""
    cur_id: str | None = None  # 当前这条 AI 消息的 id —— 换了消息就把 buf 落盘
    #: 工具参数累计，写产物那一段靠它报进度
    tool_name = ""
    arg_chars = 0
    reported_chars = 0
    reported_at = time.monotonic()

    def flush_line(line: str) -> None:
        if line.strip():
            trace.emit(
                {
                    "type": "stream",
                    "kind": "",
                    "role": role,
                    "who": who,
                    "text": line,
                    "ts": trace.stamp(),
                }
            )

    def maybe_report() -> None:
        """参数够多了吗？够了就报一行进度。报的是事实（已收到多少字、过了多久），不是预测。"""
        nonlocal reported_chars, reported_at
        if arg_chars < _WRITE_REPORT_AT:
            return
        now = time.monotonic()
        # 第一次跨过 `_WRITE_REPORT_AT` 就报（那一行就是用户要的"开始"）—— 无条件比
        # "每多 1000 字"的话，第一条要等到 1000 字才出来（前 500 字仍然静音）。
        if reported_chars and (
            arg_chars - reported_chars < _WRITE_REPORT_EVERY
            and now - reported_at < _WRITE_REPORT_EVERY_S
        ):
            return
        trace.note(
            f"⏳ {who} 正在写 `{tool_name or '?'}`：**已收到 {arg_chars} 字**"
            f"（这一段已 {now - reported_at:.0f}s；写完会落盘，正文不在这里流）",
            kind="writing",
            # 工具名走结构：展示层要把它和后面那条工具回调（executor_tool）对上，
            # 不许从这句话的反引号里抠。
            fields={"tool": tool_name},
        )
        reported_chars, reported_at = arg_chars, now

    # try/finally 是必须的：`buf` 是生成器局部变量，异常时不 flush 的话"这半行说了什么"
    # 在日志里一个字都没有 —— 而 retry note 只说"上面那半段作废"，读的人看不到被作废的
    # 是什么（切了必须标）。重试不收回已发出的 stream 事件，那半行会留在日志里，
    # 紧跟着一条 retry note 说它作废。
    try:
        async for mode, payload in graph.astream(state, config, stream_mode=["values", "messages"]):
            if mode == "messages":
                chunk, meta = payload
                # 只发 agent 自己说的话（`node='model'`）。`node='tools'` 是工具内部的 LLM
                # 调用（试答那个结构化输出），流出来是裸 JSON，既不可读、又和已解开的展示重复。
                if meta.get("langgraph_node") != "model":
                    continue
                # 换了消息就先把缓冲区落盘：模型调工具前先说一句、工具跑完再接着说，
                # 两段之间没有换行符，留在同一个 buf 里会被黏成一句。
                # 判据用 chunk.id（同一条 AI 消息的流式 chunk 共享 id）。
                # 有些 provider 不给 id，那时保持原行为，不要猜边界。
                cid = getattr(chunk, "id", None)
                if cid and cur_id and cid != cur_id:
                    flush_line(buf)
                    buf = ""
                    # 换了消息，进度计数也得归零：不归零的话，工具跑完、模型接着说时
                    # arg_chars 还在涨，屏幕上会把"已收到 3346 字"排在产物落盘那行后面。
                    # 计数是这一条消息的，不是这一趟的。
                    tool_name = ""
                    arg_chars = 0
                    reported_chars = 0
                    reported_at = time.monotonic()
                if cid:
                    cur_id = cid

                # 模型从"说话"转成"调工具"了：先把缓冲区里那句落盘。不落的话它会一直躺在
                # `buf` 里，直到整条流结束 —— 而工具是在那之前跑完的（执行器那 60–120 秒
                # 就夹在中间），读日志的人会先看见执行器干活、后看见主 agent 说要去委派，
                # 因果正好反了。判断用结构信号（这一片 chunk 带工具参数），不看正文内容。
                if getattr(chunk, "tool_call_chunks", None):
                    flush_line(buf)
                    buf = ""
                # 工具参数的增量也要数：模型写产物时输出的是参数、content 是空的，
                # 不数就那几分钟全静音。
                _nm, _n = _tool_arg_delta(chunk)
                if _nm:
                    tool_name = _nm
                arg_chars += _n
                maybe_report()

                text = _chunk_text(chunk)
                if not text:
                    continue
                buf += text
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    flush_line(line)
                continue
            yield payload
    finally:
        # 收尾：最后一行没有换行符。异常时也要走这里，原因见上面那条 try/finally。
        # 留在 `buf` 里的那半行照发，读日志的人至少知道"它说到哪儿断的"。
        flush_line(buf)


def _chunk_text(chunk) -> str:
    """从一个流式 chunk 里取纯文本（内容可能是 str，也可能是分块列表）。"""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in content)
    return ""
