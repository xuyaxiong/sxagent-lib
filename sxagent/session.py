"""会话：一次会话的全部运行期状态，多个会话共用一个进程时靠它隔离。

`Session` 装状态，`current()` 取当前会话，`session(thread_id)` 开一个，最后这个是库对外的
API。状态挂模块级的话，用户 B 会看到用户 A 的全部产物。

## 不变量

1. 对外显式、对内隐式：库的 API 是一个会话对象；内部传播走 contextvar，所以所有
   `trace.xxx()` / `artifacts.xxx()` 的调用点一个都不用改。
2. 两道回落只放行一道：从没开过会话的宿主照旧回落 `_DEFAULT`；开过之后再在上下文外读，
   当场报错，那只有一种解释，漏包了 `with session()`。
3. 会话是运行期上下文，不是领域概念。`handoff/` 的纯渲染函数不许知道它。
4. 要隔离的状态只许登记在这里，加一样就在 `Session` 上加一个字段。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TextIO

#: 事件订阅者的类型：收一个事件、不返回。
Sink = Callable[[dict[str, object]], None]


class Session:
    """一次会话的全部运行期状态。要隔离的状态只许加在这儿，不许再挂模块级。"""

    def __init__(self, thread_id: str = "") -> None:
        #: 这次会话的标识 —— 与 LangGraph 的 `config["configurable"]["thread_id"]` 同一个。
        self.thread_id = thread_id
        # ── 产物库（`artifacts.py` 用）──
        #: 产物记录：`aid → {title, kind, file_type, content, file, path}`。
        #: 对外只有函数，不给这个 dict（`artifacts` 模块头不变量 1）。
        self.artifacts: dict[str, dict[str, str]] = {}
        #: 产物落盘到哪。`None` = 只进内存（跑批那条路径就是这样）。
        self.artifact_dir: Path | None = None
        # ── 事件总线（`trace.py` 用）──
        #: 这个会话的订阅者（展示层）—— A 会话的展示层收不到 B 会话的事件。
        #: 每个元素是 `[fn, handle]`：退订要能定位到"哪一个"（见 `trace._Subscription`）。
        self.subs: list[list[Sink]] = []
        # ── 日志（`recorder.py` 用）──
        #: 日志放哪 —— `None` = 用默认目录（`SXAGENT_LOG_DIR`，没设就是应用根下的 `logs/`）。
        #: 宿主走 `recorder.use_dir()` 设，多会话共享进程时各写各的。
        self.log_dir: Path | None = None
        #: 这个会话的日志文件 —— 惰性建（没写过日志的会话不产生文件）。
        #: 建文件 / 起名字是 `recorder` 的事（它才知道 `LOG_DIR` 和那个后缀），这里只存。
        self.log_file: Path | None = None
        #: 打开着的句柄 —— 跟着 `log_file` 一起由 `recorder` 建。
        self.log_fh: TextIO | None = None


#: 默认会话 —— 只有上下文里没设时才取得到。**它的生命周期就是进程的**：产物表、发号器、
#: 日志句柄都跟着进程活，进程结束才没。拿它跑很多轮"逻辑上的会话"，产物就会一直涨 ——
#: 多会话的宿主必须每会话开一个（`session()`），那是库对外的 API。
_DEFAULT = Session(thread_id="")

_current: ContextVar[Session | None] = ContextVar("sxagent_session", default=None)

#: 这个进程开过会话没有（`use()` / `session()` 一调就置真）——
#: `current()` 靠它区分"从来就是单会话的宿主"和"漏包了 `with session()`"。
_ever_had_session = False


def current() -> Session:
    """当前会话。两道回落只给一道放行：上下文里有就用它；没有但进程从没开过会话，回落
    `_DEFAULT`；没有但开过会话，当场报错。

    最后那条是故意的：静默回落的话，多会话宿主里漏包一次，产物和日志就串进公共桶而一声
    不吭。宁可当场报错。
    """
    s = _current.get()
    if s is not None:
        return s
    if _ever_had_session:
        raise RuntimeError(
            "当前上下文里没有会话，但这个进程**已经开过会话** —— 多半是漏包了 "
            "`with session(thread_id=…)`（多会话宿主里每个请求 / 每个后台任务都得包一层）。"
            "⚠️ **别改成静默回落到默认会话**：那样产物和日志会串进一个公共桶，而且一声不吭。"
        )
    return _DEFAULT


def default() -> Session:
    """默认会话那个对象 —— 给宿主预置状态用（比如 `recorder` 在 import 时把进程日志文件
    挂上去）。别拿它当"当前会话"，那是 `current()` 的事。
    """
    return _DEFAULT


def use(thread_id: str = "") -> Session:
    """把当前上下文的会话设成一个新的，不开作用域（进程退出即会话结束）。

    只给"进程就是一次会话"的宿主用（CLI / 跑批子进程）。web 那种多会话宿主必用
    `session()`——用 `use()` 的话后一个请求会把还在跑的前一个顶掉。
    """
    global _ever_had_session
    s = Session(thread_id=thread_id)
    _current.set(s)
    _ever_had_session = True
    return s


@contextmanager
def session(thread_id: str = "") -> Iterator[Session]:
    """开一个会话（库的对外 API，web 每个请求一层）。

    with sxagent.session(thread_id=user_id) as s:
        ...            # 这段里所有 artifacts / trace 调用都归这个会话
    """
    global _ever_had_session
    s = Session(thread_id=thread_id)
    token = _current.set(s)
    _ever_had_session = True
    try:
        yield s
    finally:
        _current.reset(token)
