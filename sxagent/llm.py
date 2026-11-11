"""LLM 调用的公共工具：`safe_structured()` 和一条 pydantic 警告的压制。

结构化输出失败重试一次，仍失败就用调用方给的保守默认值。独立成模块，是为了让 `agent` 和
`executor` 不必互相 import。

## 不变量

1. 一次抖动不该让整轮失败，所以有兜底。
2. 退化值必须保守，由 `default` 给：宁可"答不上"，不许编一个结论。
3. 只压那一条警告，按消息匹配。别用 `simplefilter("ignore")`：看不到就观察不到。
4. 一套，别分叉。调用方自己会渲染决策记录时，`log_response` 传 False。"""

from __future__ import annotations

import time
import warnings

from sxagent import trace
from sxagent.config import get_model, log_prompt_verbatim

# 压掉一条已知的 pydantic 序列化警告 —— 它是 `stream_mode="messages"` 的副作用，返回的结果
# 是对的（同一个 runnable：`.invoke()` 不警告、`.stream()` 警告）。
# 只压这一条（按消息正则匹配），不要用 `simplefilter("ignore")` 把别的警告一起埋了。
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*", category=UserWarning)


async def safe_structured[T](
    schema: type[T],
    prompt: str,
    default: T,
    *,
    tag: str = "",
    kind: str = "",
    log_prompt: bool = True,
    log_response: bool = True,
) -> T:
    """结构化输出调用 + 兜底（重试一次 → 仍失败退化到 `default`）。

    实测 DashScope 偶发 `grammar validation or compilation failed: sandbox grammar compile
    timed out`，一次抖动就让整个 turn 失败。这类基础设施抖动不该让整轮失败。

    Args:
        schema: Pydantic 模型类。
        prompt: 提示词全文。
        default: 重试仍失败时的返回值 —— 必须是保守的那个。
        tag: 日志里显示的标题。
        kind: 这次调用的语义标签 —— 原样发给 section 事件；它的 PROMPT / RESPONSE 块一律带
            `kind="llm_io"`，展示层据此归并成同一块。
        log_prompt: 这一步的 prompt 允不允许逐字写进日志（调用方声明，默认允许）。prompt 里
            嵌了产物正文时传 False —— 产物本身已另存，由调用方先记一行"喂了什么"的摘要即可。
            它只管"允不允许"；"这次运行要不要真记"是 `SXAGENT_LOG_PROMPT`（默认不记）——
            两个问题分开：谁决定（调用方）vs 什么时候看（开关）。
        log_response: 是否把返回原样写进日志（`str(r)` = pydantic repr）。调用方自己会渲染一条
            「决策记录」时传 False —— 否则同一件事在日志里两副面孔。
    """
    trace.section(tag or f"结构化输出 · {getattr(schema, '__name__', schema)}", kind=kind)
    # 默认不逐字记（渐进式披露：清单常驻、正文按需）—— 清单是调用方上面那条 `trace.handoff`
    # 摘要（多少字 ＋ 什么构成），正文要看时置 `SXAGENT_LOG_PROMPT=1`。
    # 不记也必须说（"切了要标"）—— 下面两条 note 就是那个声明，缺了才是违反。
    if log_prompt and log_prompt_verbatim():
        trace.block("PROMPT（发给 LLM 的全文）", prompt, kind="llm_io")
    elif log_prompt:
        # 别写"构成见上面那条交接摘要"：`trace.handoff` 的行只进日志、不上屏 —— 那样终端上
        # 就指了一个不存在的东西。"喂了什么"由调用方写在两边都上屏的地方（试答那步是【前提】）。
        trace.note(
            f"PROMPT 未逐字记录（共 {len(prompt)} 字）—— 要看全量置 `SXAGENT_LOG_PROMPT=1`",
            kind="llm_io" if kind else "",
        )
    else:
        trace.note(
            f"PROMPT 未逐字记录（共 {len(prompt)} 字）—— 内含**产物正文**，产物另存；"
            "喂了哪几份见调用方上面记的那条摘要",
            kind="llm_io" if kind else "",
        )

    for attempt in (1, 2):
        # 计时起点在循环里：放外面的话，第二次尝试的 `RESPONSE（N.Ns）` 显示的是两次累计的
        # 秒数 —— 日志里那个数字会骗人。
        t0 = time.time()
        try:
            r = await get_model().with_structured_output(schema).ainvoke(prompt)
            # `None` 也算失败：`with_structured_output` 在模型返回空的时候可能返回 `None`
            # 而不抛 —— 那样调用方下一行直接 AttributeError、整轮失败，而这条路上
            # `safe_structured` 的全部意义就是"别让一次抖动导致整轮失败"。
            if r is None:
                raise ValueError("结构化输出返回了 None（模型返回了空）")
            if log_response:
                trace.block(f"RESPONSE（{time.time() - t0:.1f}s）", str(r), kind="llm_io")
            return r
        except Exception as e:  # noqa: BLE001
            trace.note(
                f"⚠️ 结构化输出失败（第 {attempt}/2 次）：{type(e).__name__}: {e}",
                kind="llm_io" if kind else "",
            )
    trace.note(f"⚠️ 两次都失败 → 退化到保守默认值 {default}", kind="llm_io" if kind else "")
    return default
