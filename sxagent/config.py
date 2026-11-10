"""模型配置：全部走环境变量 / .env，不写死在代码里。

    MODEL_NAME      模型名，如 deepseek-chat / qwen-max / glm-4-plus / gpt-4o-mini
    MODEL_API_KEY   API Key
    MODEL_BASE_URL  兼容端点，如 https://api.deepseek.com/v1（官方 OpenAI 可不填）
    MODEL_TIMEOUT_S 单次调用的**读超时**（秒，默认 300）—— 卡住时的闸门，
                    见 `get_model` 里那条注释（它治的是"一次阻塞调用"，不是墙钟）
    SXAGENT_LOG_PROMPT  PROMPT 全文是否**逐字**写进日志（默认**不写**）—— 渐进式披露：
                    清单常驻、正文按需。见 `log_prompt_verbatim`

## 这个文件是什么

配置与运行环境的唯一出处：`.env` 从哪读（`app_root`）、打包后随包带的只读资源在哪
（`RESOURCE_ROOT`）、三个标准流的编码、以及 `get_model()` 造出来的那个 OpenAI 兼容
chat model。**不碰任何业务**。

## 设计原则（不变量）

1. **路径一律现算，不靠 cwd** —— 打包后"从哪双击"都不该影响读不读得到配置。
2. **三个流都设 UTF-8**（`stdin` / `stdout` / `stderr`）—— 缺哪个坏哪种，
   见 `load_dotenv` 之后那条注释。
3. **只走真实 LLM 调用，没有 fake 开关** —— 曾经有一个，拆了：它**只假一半**
   （主 agent 仍调真实 API）、又**短路执行器的整个轮次循环**，测多轮行为时给的是
   **假信号**。想省额度就少跑几轮，别造假信号。
4. **单次调用有读超时闸门**（`MODEL_TIMEOUT_S`）—— 墙钟治不了"一次阻塞调用"。
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def app_root() -> Path:
    """**应用根** —— 打包后是**程序所在目录**，开发时是仓库根。

    🔴 **不能靠 cwd**（实测踩过）：`load_dotenv()` 从 **cwd** 往上找，
    不是从代码位置找 —— 而双击程序时 cwd 也不保证是程序目录。

    `sys.frozen` 是 PyInstaller 打的包会设的标记；`sys.executable` 那时就是程序自己。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]  # sxagent/ 的上一层 = 仓库根


#: 打包后随包带的只读资源在 `sys._MEIPASS`（PyInstaller 的解包目录），开发时就是应用根。
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "_MEIPASS", "") else app_root()

#: **显式指定路径**，不让它去猜 cwd —— 打包后"从哪双击"都不该影响读不读得到配置。
load_dotenv(app_root() / ".env")

# 🔴 **三个流都要设 UTF-8**（打包后实测踩过两次）：缺哪个坏哪种 ——
#   stderr 缺了，报错落回系统编码（中文 Windows 上是 GBK），"报错"和"正常输出"
#   在同一份重定向里就是**两种编码**；stdin 缺了，管道喂进来的中文按 GBK 解，
#   解不了的字变成**代理字符**，接着写日志时 `UnicodeEncodeError` —— 一条坏字符整场崩。
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 —— 老终端/特殊流不支持就算了，不该因此起不来
        pass


@lru_cache(maxsize=8)
def get_model(temperature: float = 0.0):
    """返回一个 OpenAI 兼容的 chat model（同一温度只建一次）。"""
    from langchain_openai import ChatOpenAI

    name = os.getenv("MODEL_NAME", "").strip()
    key = os.getenv("MODEL_API_KEY", "").strip()
    base = os.getenv("MODEL_BASE_URL", "").strip()

    if not name or not key:
        # ⚠️ 路径要**现算**：打包后 `.env` 在程序旁边，写死具体目录名
        # 会把人指到一个根本不存在的目录。
        raise SystemExit(
            f"\n缺少模型配置。请在 {app_root() / '.env'} 里填：\n"
            "  MODEL_NAME      模型名\n"
            "  MODEL_API_KEY   API Key\n"
            "  MODEL_BASE_URL  兼容端点（可选）\n"
            "\n（没有 fake 模式 —— 见模块 docstring 的说明。）\n"
        )
    # 🔴 **单次调用的读超时**（默认 300s）—— 卡住时的闸门在这儿，不在墙钟：
    #    墙钟在轮次之间查表，而卡住的是**一次阻塞调用**，轮次边界根本走不到。
    #    ⚠️ 它是**读间隔**超时（httpx 按"两次读到数据之间"算）—— 一直在吐字的流
    #    不会被误杀，只有真卡住才触发 ✓。
    #    ⚠️ 触发后走既有故障路：`APITimeoutError` 没有 `status_code` → `transient`
    #    → 回执说"原样再派"（不新增分支）。
    timeout_s = float(os.getenv("MODEL_TIMEOUT_S", "") or 300)
    return ChatOpenAI(
        model=name,
        api_key=key,  # type: ignore[arg-type]
        base_url=base or None,
        temperature=temperature,
        timeout=timeout_s,  # type: ignore[arg-type]
    )


def describe() -> str:
    """返回一行配置描述，便于运行时确认用的哪个模型。"""
    name = os.getenv("MODEL_NAME", "?")
    base = os.getenv("MODEL_BASE_URL", "").strip() or "api.openai.com"
    return f"{name} @ {base}"


def log_prompt_verbatim() -> bool:
    """PROMPT 全文是否**逐字**写进日志 —— **默认不写**（置 `SXAGENT_LOG_PROMPT=1` 才写）。

    🔴 这是渐进式披露的落地：**清单常驻、正文按需**。清单是调用方自己那条
    `trace.handoff` 摘要（**多少字 ＋ 什么构成**），一直在；正文（拼好的整份 PROMPT）
    默认不落 —— 它的**内容已经逐条记过**（每条消息进上下文时各记一条），再抄一遍是
    同一件事第二副面孔，而且随会话**无界增长**（实测 4 次试答 621 → 1225 → 1846 → 2411 字，
    同期交接书 289 → 824 字，**有界**）。

    ⚠️ **不写也必须说**（"切了要标"）—— `llm.safe_structured` 会留一条 note 报字数
    并指明这个开关。**没有那条 note 才是违反。**

    ⚠️ 开关放这儿、不写在 `llm.py` —— 本模块是"配置与运行环境的唯一出处"，
    env 的名字只许有一处定义（一套，别分叉）。

    ⚠️ **故意不缓存**（不加 `lru_cache`）：测试要能当场翻它。
    """
    return os.getenv("SXAGENT_LOG_PROMPT", "").strip().lower() in {"1", "true", "yes", "on"}
