"""打包门禁：`uv build` 的两个产物各装进一个干净 venv，能用公共 API 跑通一轮。

守两件事：wheel 里只有库本体（`sxagent/` ＋ `dist-info`）；wheel 和 sdist 装进别人的环境
真能 import、真能跑（sdist 那条顺带验了它打得出 wheel）。要网络（装依赖）和几十秒，所以
不进 pytest，发版前手动跑：

    uv run python scripts/wheel_check.py

`--probe` 是内部模式：装好之后由 venv 里那个解释器再调一次本文件，跑那一轮检查 —— 这样
`import sxagent` 拿到的是装进去的那份，不是仓库里这份。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# 管道里（CI、`| tail`）父进程的 print 是块缓冲，子进程的输出会插到前面 —— 顺序会骗人。
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp" / "wheel-check"


def probe() -> None:
    """在 venv 里跑：注入 model / artifacts / skills → `build()` → 一轮 `ainvoke`。"""
    import asyncio
    from typing import Any

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    import sxagent
    from sxagent import LocalSkills, MemoryArtifacts, Skill, build, session

    where = str(Path(sxagent.__file__).resolve())
    print("sxagent 从哪来：", where)
    if "site-packages" not in where:
        raise SystemExit(f"✗ import 到的不是装进去的那份（{where}）—— 这份检查没验到包")

    class _Fake(BaseChatModel):
        """最小假模型：不调工具、直接答一句 —— 只验注入的模型真被用上。"""

        @property
        def _llm_type(self) -> str:
            return "wheel-check-fake"

        def bind_tools(self, tools: Any, **kwargs: Any) -> _Fake:
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=AIMessage("WHEEL-CHECK-OK"))])

    with session.session("wheel-check"):
        app = build(
            model=_Fake(),
            artifacts=MemoryArtifacts(),
            skills=LocalSkills({"check": Skill("check", "自检技能", "按任务要求回一句话")}),
        )
        state = asyncio.run(
            app.ainvoke(
                {"messages": [HumanMessage("你好")]},
                {"configurable": {"thread_id": "wheel-check"}},
            )
        )
    last = state["messages"][-1]
    if "WHEEL-CHECK-OK" not in str(last.content):
        raise SystemExit(f"✗ 注入的模型没被用上，最后一跳是 {last!r}")
    print("✓ 注入模型 → build() → 一轮跑通")


def build_artifacts() -> tuple[Path, Path]:
    print("→ uv build")
    subprocess.run(["uv", "build"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    dist = ROOT / "dist"
    return (
        max(dist.glob("*.whl"), key=lambda p: p.stat().st_mtime),
        max(dist.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime),
    )


def check_wheel(whl: Path) -> None:
    """wheel 里只许有 `sxagent/` 和 `dist-info` —— 混进别的，说明包发现配置被改坏了。"""
    tops = {n.split("/")[0] for n in zipfile.ZipFile(whl).namelist()}
    extra = sorted(t for t in tops if t != "sxagent" and not t.endswith(".dist-info"))
    if extra:
        raise SystemExit(f"✗ {whl.name} 里混进了 {extra}（只该有 sxagent/ 和 dist-info）")
    print(f"✓ {whl.name}：顶层只有 {sorted(tops)}")


def install_and_run(artifact: Path, name: str) -> None:
    """一个产物一个干净 venv：装进去（sdist 会先被打成 wheel），再跑一轮。"""
    python = WORK / name / "bin" / "python"
    print(f"→ 建 {name}，装 {artifact.name}")
    subprocess.run(
        ["uv", "venv", str(WORK / name)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(artifact)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"→ 在 {name} 里跑一轮")
    subprocess.run([str(python), str(Path(__file__).resolve()), "--probe"], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="打包门禁：产物装进干净 venv 各跑一轮")
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe:
        probe()
        return
    shutil.rmtree(WORK, ignore_errors=True)
    whl, sdist = build_artifacts()
    check_wheel(whl)
    install_and_run(whl, "venv-wheel")
    install_and_run(sdist, "venv-sdist")
    print("\n✓ 发布清单这两条过了：产物干净、两个产物装进干净环境都能跑。")


if __name__ == "__main__":
    main()
