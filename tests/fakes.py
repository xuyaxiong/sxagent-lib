"""测试替身：一个能驱动工具调用的假模型。

`create_agent` 建图时一定会调 `bind_tools`，而 langchain 自带的 `GenericFakeChatModel` 那个
方法直接 `NotImplementedError`，所以这里自己写一个最小的。

**它验的是机制，不是模型行为**——模型行为那层归真会话和跑批。

## 不变量

1. 无状态：同一份 messages 永远得到同一个结果。两个 run 共用一个实例，谁先谁后都对。
2. `_agenerate` 里让出控制权（`await asyncio.sleep(0)`）——不然两个 run 根本不会交错，
   拿它写的并发测试就成了"顺序跑两遍"，什么也没验到。
3. 每次调用记进 `calls`（标记 + 时刻）：测试据此断两个 run 确实重叠过。
4. `with_structured_output` 返回事先摆好的那个对象（`structured`，按 schema 类名查）——
   `safe_structured` 那条路（试答）靠它，不碰真模型。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

#: 假模型认的标记。它从消息里找出一个，就拿它当产物名去调 `write_artifact`。
MARKERS: tuple[str, ...] = ("哨兵-A", "哨兵-B")


class ToolCallingFakeModel(BaseChatModel):
    """按脚本发工具调用的假模型：先调一次 `write_artifact`，工具跑完就收尾。"""

    markers: tuple[str, ...] = MARKERS
    #: 每次调用：(认出的标记, 时刻)。测试拿它断两个 run 重叠过。
    calls: list[tuple[str, float]] = Field(default_factory=list)
    #: 结构化调用的固定回答，按 schema 类名查（试答那条路）。
    structured: dict[str, object] = Field(default_factory=dict)
    #: 结构化调用真发生了几次。断"用的是它"只能靠这个：`safe_structured` 把异常全吞了，
    #: "没炸"证明不了什么（它会退默认值，看起来一样正常）。
    structured_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCallingFakeModel:
        """`create_agent` 要这个；假模型不需要真绑定。"""
        return self

    def with_structured_output(self, schema: type, **kwargs: Any) -> RunnableLambda:
        """`safe_structured` 要这个；返回摆好的那个对象，不碰真模型。"""
        answer = self.structured[getattr(schema, "__name__", "")]

        def _answer(_prompt: object) -> object:
            self.structured_calls += 1
            return answer

        return RunnableLambda(_answer)

    def _respond(self, messages: list[BaseMessage]) -> ChatResult:
        text = "\n".join(str(m.content) for m in messages)
        marker = next((m for m in self.markers if m in text), "哨兵-未知")
        self.calls.append((marker, time.monotonic()))
        # 工具已经跑过了（消息里出现了 ToolMessage）→ 收尾，别再调
        if any(isinstance(m, ToolMessage) for m in messages):
            return self._result(AIMessage(f"{marker} 收尾"))
        return self._result(
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "write_artifact",
                        "args": {
                            "title": marker,
                            "summary": f"{marker} 的梗概",
                            "file_type": "md",
                            "content": f"正文：{marker}",
                        },
                        "id": f"call-{marker}",
                    }
                ],
            )
        )

    @staticmethod
    def _result(message: AIMessage) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._respond(messages)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(0)  # 见不变量 2
        return self._respond(messages)
