"""**会话**（Session）：一次会话的**全部运行期状态** —— 多会话共享进程时，按它隔离。

## 这个文件是什么

`sxagent` 原来把运行期状态挂在**模块级**（产物库 / 日志句柄 / 订阅者…），那是
"**一个进程一个会话**"的形状 —— 要包成**库**、接**多用户 web** 时这个形状就串了
（实测最重的一处：用户 B 会看到用户 A 的全部产物）。

本模块给出那个**隔离单位**：`Session` 装状态、`current()` 取"当前会话"、
`session(thread_id)` 开一个（**这就是库的对外 API**）。

## 设计原则（不变量）

1. **对外显式、对内隐式** —— 库的 API 是一个**会话对象**（顶层看得见）；
   内部传播走 **contextvar**，所以全部 `trace.xxx(...)` / `artifacts.xxx(...)`
   调用点**一个不改**。
2. **两道回落，只给其中一道放行** —— **从没开过会话**的宿主照旧回落 `_DEFAULT`
   （CLI / 跑批 / 单测一个字不变）；而**开过会话之后**再在上下文外读 → **当场炸**
   （那只有一种解释：漏包了 `with session()`）。
3. **会话是运行期上下文，不是领域概念** —— `handoff/` 那些纯渲染函数**不许**知道它。
   所以**不走**"每会话一个对象、一路传参"（那会把会话塞进领域层）。
4. **要隔离的状态只许登记在这里**，不许再挂模块级 —— 加一样就在 `Session` 上加一个字段。

## 关键取舍

**① 为什么是 contextvar，不是一路传参。** 传参要改上百个调用点、每个业务函数多一个参数
（分层污染）；contextvar 在 **async web 框架里天然对齐**（每个请求一个 context）。

**② 回落默认会话会不会藏出一条分叉。** 不会：它**只有一处定义**（`_DEFAULT`），
而且**只有上下文里没设时**才取得到 —— 生产路径（web）永远在 `session()` 里跑。

## 遗留问题

- **`_DEFAULT` 那个对象本身仍然永不回收**（它是进程级的）—— 但"**该不该取到它**"
  这件事已经收紧了（开过会话之后取不到就是炸）。剩下的是"对象不回收"，那是内存问题，
  不是串数据问题。
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator


class Session:
    """一次会话的**全部运行期状态** —— 字段按**归属的模块**分组。

    ⚠️ 加字段的规矩（设计原则 4）：**要隔离的状态只许加在这儿**，不许再挂模块级。
    """

    def __init__(self, thread_id: str = "") -> None:
        #: 这次会话的标识 —— 与 LangGraph 的 `config["configurable"]["thread_id"]` **同一个**。
        self.thread_id = thread_id
        # ── 产物库（`sxagent/artifacts.py` 用）──
        #: 产物记录：`aid → {title, kind, file_type, content, file, path}`。
        #: ⚠️ **对外只有函数，不给这个 dict**（`artifacts` 模块头原则 1）。
        self.artifacts: dict[str, dict[str, str]] = {}
        #: 产物 ID 发号器 —— **每会话一条序列**（`next()` 原子，不用 `len(表) + 1` 那种"先查后写"）。
        self.artifact_seq: Iterator[int] = itertools.count(1)
        #: 产物落盘到哪。`None` = 只进内存（跑批那条路径就是这样）。
        self.artifact_dir: Path | None = None
        # ── 事件总线（`sxagent/trace.py` 用）──
        #: 这个会话的**订阅者**（展示层）—— A 会话的展示层收不到 B 会话的事件。
        #: ⚠️ 每个元素是 `[fn, handle]`：退订要能定位到"哪一个"（见 `trace._Subscription`）。
        self.subs: list[list] = []
        # ── 日志（`sxagent/trace.py` 用）──
        #: 这个会话的日志文件 —— **惰性建**（没写过日志的会话**不产生文件**）。
        #: ⚠️ 建文件 / 起名字是 `trace` 的事（它才知道 `LOG_DIR` 和那个后缀），这里只**存**。
        self.log_file: Path | None = None
        #: 打开着的句柄 —— 跟着 `log_file` 一起由 `trace` 建。
        self.log_fh: Any = None


#: 默认会话 —— **只有上下文里没设时才取得到**（见「关键取舍 ②」）。
_DEFAULT = Session(thread_id="")

_current: ContextVar[Session | None] = ContextVar("sxagent_session", default=None)

#: 这个进程**开过会话没有**（`use()` / `session()` 一调就置真）——
#: `current()` 靠它区分"**从来就是单会话的宿主**"和"**漏包了 `with session()`**"。
_ever_had_session = False


def current() -> Session:
    """**当前会话**。

    🔴 **两道回落，只给其中一道放行**：

    | 情形 | 怎么办 |
    |---|---|
    | 上下文里有会话 | 用它（正常路径） |
    | 上下文里没有，而**这个进程从没开过会话** | 回落 `_DEFAULT` —— CLI / 跑批 / 单测都在这条上，**行为一个字不变** |
    | 上下文里没有，**但进程开过会话** | 🔴 **当场炸** —— 那只有一种解释：**漏包了 `with session()`** |

    ⚠️ 最后那条**是故意的**：静默回落的话，多会话宿主里漏包一次，产物和日志就串进
    "公共桶"（`_DEFAULT` 的 `thread_id` 是空串），**一声不吭**。宁可当场炸，
    也不要"悄悄接住"（同"禁止静默降级"那条）。
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
    """**默认会话**那个对象 —— 给宿主**预置状态**用（比如 `trace` 在 import 时把
    进程日志文件挂上去）。⚠️ 别拿它当"当前会话"，那是 `current()` 的事。
    """
    return _DEFAULT


def use(thread_id: str = "") -> Session:
    """把当前上下文的会话设成一个新的 —— **不开作用域**（进程退出即会话结束）。

    ⚠️ **只给"进程就是一次会话"的宿主用**（CLI / 跑批的子进程）：那种宿主里
    "当前会话"从启动到退出**只有一个**，包 `with` 只是白白多一层缩进。
    ⚠️ **web 那种"一个进程多个会话"的宿主必须用 `session()`** —— 用 `use()` 的话
    后一个请求会把前一个顶掉，而前一个还在跑。
    """
    global _ever_had_session
    s = Session(thread_id=thread_id)
    _current.set(s)
    _ever_had_session = True
    return s


@contextmanager
def session(thread_id: str = "") -> Iterator[Session]:
    """开一个**会话**（库的对外 API —— web 每个请求一层）。

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
