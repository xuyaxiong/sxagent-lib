"""把验证 CLI（`app/chat.py`）打成能直接发给测试者的独立程序，单目录。

    uv run python build_cli.py          # → dist/sxagent/

用 onedir 不用 onefile：onefile 每次启动都要把上百 MB 解压到临时目录，慢好几秒；而我们本来
就要给用户一个目录（`.env` / `skills/` / `runs/` 都在程序旁边）。代价是多了 `_internal/`，
所以要提醒用户整个文件夹拷走。

## 不变量

1. 打出来必须在一台没装 Python 的机器上跑通一次完整对话才算数。PyInstaller 漏收依赖是常态，
   langchain / langgraph 这种动态导入重灾区尤其是，那串 `--collect-*` 不能假定够了。
2. 重建不许删用户数据：`--noconfirm` 会把目标目录整个删掉，而里面住着 `.env`（真 key）、
   自己加的 `skills/`、每次对话的 `runs/`。打包前挪走，打完（哪怕打失败）原样放回。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "sxagent"
ENTRY = ROOT / "app" / "chat.py"

#: 随包带的只读资源 —— `skills/` 下每个包是首次运行要铺给用户的示例技能。
#: 落点必须和 `sxagent.config.RESOURCE_ROOT` 的约定对上（冻结后 = `sys._MEIPASS`）。
ADD_DATA: list[tuple[str, str]] = [
    ("skills", "skills"),
]

#: 动态导入重灾区 —— 不加这些，打出来的包会在用户那里报 ImportError。
#: `tiktoken` 靠 entry point 动态找 `tiktoken_ext.openai_public`，是最经典的漏收点。
COLLECT_ALL: list[str] = [
    "tiktoken",
    "langchain_openai",
]
COLLECT_SUBMODULES: list[str] = [
    "langchain_core",
    "langgraph",
    "langchain",
]

#: 明确不用的重家伙，排掉省体积（排错了会在运行时报 ImportError，所以只排确定不用的）。
EXCLUDE: list[str] = [
    "tkinter",
    "matplotlib",
    "pytest",
    "PyInstaller",
]

#: 重建时必须保住的东西 —— `--noconfirm` 会把目标目录整个删掉再重建，
#: 而这几样是用户的数据，不是构建产物。打包前挪到 `<dist>/_keep-<时间>`，
#: 打完（哪怕打失败了）原样放回。`logs/` 不在列 —— 那是原始 trace，
#: `runs/<tag>/` 里的 `_run.md` 已经附了一份。
KEEP: list[str] = [
    ".env",  # 填好的 key / model / base_url —— 丢了包就跑不起来
    "skills",  # 用户自己往里丢的技能包
    "runs",  # 每次对话的记录（收回来分析用的）
]


def _stash_user_data(built: Path, stash: Path, moved: list[str]) -> None:
    """把 `built` 里属于用户的数据挪进 `stash`（`KEEP` 里没有的就跳过）。

    挪走了哪些记在 `moved` 里（由调用方传进来）：挪到一半失败时，
    已经挪走的那几样得靠它在 `finally` 里放回去。
    """
    for name in KEEP:
        src = built / name
        if not src.exists():
            continue
        stash.mkdir(parents=True, exist_ok=True)
        try:
            # 用 `os.rename`（同卷改名，原子）而不用 `shutil.move` —— `move` 在改名失败时
            # 会回退成"先复制、再删源"，删到一半失败会在暂存里留一份孤儿副本。
            # 两边都在 `dist/` 下，同卷改名用不上那条回退。
            os.rename(src, stash / name)
        except OSError as e:
            # 打好的程序还开着时，它占着 `runs/` 里的 `_console.log`，Windows 不许改名。
            # 这里不强行处理：说清怎么办，已经挪走的交给 `finally` 放回去。
            raise RuntimeError(
                f"挪不动 {src}（{e}）\n"
                f"  ⚠️ 多半是**打好的程序还在跑** —— 它占着 `runs/` 里的文件。\n"
                f"     先把它关掉；不想关就换个落点：`--out dist/_new`"
            ) from e
        moved.append(name)


def _restore_user_data(built: Path, stash: Path, moved: list[str]) -> None:
    """原样放回去。

    落点上已经有同名东西时合并不覆盖 —— 只有"打包中途失败"才会遇到，
    而那时落点上的东西（正在跑的程序刚写下的记录）同样是用户的数据，
    不能拿暂存里那份旧的盖掉。
    """
    for name in moved:
        src, dst = stash / name, built / name
        if not src.exists():
            continue
        if not dst.exists():
            os.rename(src, dst)
        elif dst.is_dir() and src.is_dir():
            for child in src.iterdir():
                if not (dst / child.name).exists():
                    os.rename(child, dst / child.name)
        # 单文件撞名（`.env`）：以落点上已有的为准，暂存那份留着 ——
        # 调用方看到暂存没清空会提示路径，不会一声不吭删掉。


def build(out_dir: Path | None = None) -> None:
    """`out_dir` 不传就打进 `dist/`。

    上一个包还在跑的时候必须传 `--out dist/_new`：PyInstaller 要删除目标目录再重建，
    而正在运行的程序占着那个目录，会报 PermissionError（WinError 5）。
    """
    dist = out_dir or (ROOT / "dist")
    # 落点跟着 `dist` 走（`--out` 传了就换地方）：提示和实际落点必须是同一个。
    built = dist / NAME
    # 暂存目录每次换一个名字（`_keep-1102-093512`）：固定名字会跟"上一次没放回去的暂存"
    # 撞上，而那种时候更要紧的是别覆盖它 —— 宁可多留一个目录，也不能把用户的东西盖掉。
    stash = dist / f"_keep-{time.strftime('%m%d-%H%M%S')}"
    args: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(ENTRY),
        "--name",
        NAME,
        "--onedir",  # ← 见模块 docstring：不要改成 onefile
        "--console",  # 双击弹一个控制台窗口跑 chat —— 不用用户敲命令行
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist),
        "--workpath",
        str(ROOT / "build" / dist.name),  # 临时目录也跟着分开，免得互相踩
        "--paths",
        str(ROOT),  # 让 `app` / `sxagent` 两个包可被解析
    ]
    for src, dst in ADD_DATA:
        args += ["--add-data", f"{src}{';' if sys.platform == 'win32' else ':'}{dst}"]
    for m in COLLECT_ALL:
        args += ["--collect-all", m]
    for m in COLLECT_SUBMODULES:
        args += ["--collect-submodules", m]
    for m in EXCLUDE:
        args += ["--exclude-module", m]

    kept: list[str] = []
    try:
        _stash_user_data(built, stash, kept)
        if kept:
            print(f"先把用户数据挪到一边（打完放回来）：{'、'.join(kept)}")
        print("打包中（几分钟）……")
        subprocess.run(args, cwd=ROOT, check=True)

        # 随包放一份 `.env.example` 当模板，绝不拷真的 `.env`（那是密钥）。
        example = ROOT / ".env.example"
        if example.is_file():
            shutil.copy2(example, built / ".env.example")
            print(f"\n已放入模板：{built / '.env.example'}")
    finally:
        # 放回放在 `finally`：打失败时更要放 —— `check=True` 一抛异常、
        # 甚至上面挪到一半就失败（`kept` 是逐个追加的），都不能让用户的 `.env`
        # 留在暂存目录里。
        if kept:
            built.mkdir(parents=True, exist_ok=True)
            try:
                _restore_user_data(built, stash, kept)
                print(f"用户数据已放回：{'、'.join(kept)}")
            except OSError as e:
                print(f"⚠️ 放回时出错：{e}")
        # 只有真放空了才删暂存目录 —— 里面但凡还剩东西，宁可留着让人自己拿。
        if stash.exists():
            if any(stash.iterdir()):
                print(f"⚠️ 有东西没能放回，暂存在（没删）：{stash}")
            else:
                stash.rmdir()
    print(
        f"\n完成：{built}\n"
        f"  ⚠️ 发出去之前把 `.env` 填好放进这个目录（模板已给）\n"
        f"  ⚠️ 告诉用户**整个文件夹拷走**，别只拷可执行文件"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="打包成能发给别人的程序")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="产物落到哪个 dist 目录（默认 dist/）。⚠️ **上一个包还在跑时必须指定**，"
        "否则 PyInstaller 删不掉目标目录会报 PermissionError",
    )
    build(ap.parse_args().out)
