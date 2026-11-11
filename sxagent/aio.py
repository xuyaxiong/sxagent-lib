"""从同步代码里驱动协程。一个进程只有一个事件循环。

宿主入口都是同步的（CLI 的 `input()`、跑批子进程、评分层），要驱动 `async def` 的节点和工具，
得有这个统一入口。

## 不变量

1. 一个进程一个循环，不用 `asyncio.run()`。它跑完就关循环，而 `config.get_model()` 被
   `lru_cache` 缓存，里面的 httpx 客户端绑在创建它的循环上，第二轮会撞
   `RuntimeError: Event loop is closed`。
2. 只此一处。每个入口各写一份 `_LOOP` 是同一语义两处定义，必然漂。
3. 只给"进程即会话"的宿主。web 那种一个进程跑多会话的，自己在异步上下文里 await。"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine

#: 进程级那个循环 —— 见不变量 1（为什么不能是 `asyncio.run`）。
_LOOP = asyncio.new_event_loop()


def run[T](coro: Coroutine[object, object, T]) -> T:
    """在进程级那个循环上把 `coro` 跑完，返回结果。

    别在异步上下文里调它（会报 "cannot be called from a running event loop"）——
    这个函数是给同步宿主用的入口，异步代码里直接 `await`。
    """
    return _LOOP.run_until_complete(coro)
