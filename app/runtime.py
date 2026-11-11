"""跑 agent 的装配与记录层，交互式入口和跑批共用。

    from app.runtime import install, snapshot, dump_run_record

装什么：把技能读进 `sxagent.registry`；再给 `store_artifact` / `run_task` / 模型调用各包一层
记录器，记产物流水、执行器流水、token 计数。录什么：`snapshot()` 抓一轮全量快照，
`dump_artifacts()` / `dump_run_record()` 落成逐轮全记录加原始 trace 附录。

两个入口必须共用这一层，因为要产出同一种记录格式：交互式入口就是把预置 turns 换成人当场打
的字，格式不一致，收回来的一堆 run 就没法比。故障注入不在这里，那是纯测试旋钮。"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage

from sxagent import config as sxagent_config
from sxagent.registry import registry
from sxagent.skill_pack import ENTRY_NAME
from sxagent.config import app_root


# ──────────────────────────── token / 调用计数 ────────────────────────────


class UsageCounter(BaseCallbackHandler):
    """挂在模型实例上，数所有 LLM 调用（主 agent + 执行器 + 试答）。"""

    def __init__(self) -> None:
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.per_call: list[dict[str, object]] = []

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        self.calls += 1
        u = None
        try:
            u = response.generations[0][0].message.usage_metadata
        except Exception:  # noqa: BLE001
            pass
        if u:
            self.in_tokens += u.get("input_tokens", 0)
            self.out_tokens += u.get("output_tokens", 0)
        self.per_call.append(
            {"in": (u or {}).get("input_tokens", 0), "out": (u or {}).get("output_tokens", 0)}
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "llm_calls": self.calls,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "total_tokens": self.in_tokens + self.out_tokens,
        }


# ──────────────────────────── 应用根 / 目录约定 ────────────────────────────

#: PyInstaller 打的包里 `sys.frozen` 为真 —— 目录约定据此分叉。
FROZEN = bool(getattr(sys, "frozen", False))
#: 应用根：冻结后 = 程序所在目录，开发时 = 仓库根（见 `sxagent.config.app_root`）。
ROOT = app_root()
#: 只读资源（随包带）：冻结后是 `sys._MEIPASS`，开发时同 ROOT。
RESOURCE_ROOT = sxagent_config.RESOURCE_ROOT

#: 技能目录：程序旁边的 `skills/`。冻结与开发用同一个表达式，两边指的是同一个地方。
DEFAULT_SKILLS_DIR = ROOT / "skills"

#: 运行记录：应用根下的 `runs/`，程序旁边（便携、找得到、方便整包发回来）。
RUNS_DIR = ROOT / "runs"

#: 首次运行铺进用户技能目录的示例技能，从只读资源里取（开发时就是 `skills/`）。
SEED_SKILLS_DIR = RESOURCE_ROOT / "skills"


def ensure_app_dirs() -> list[str]:
    """首次运行把目录铺好，只在冻结后做。返回做了哪些事，给启动时打一行用。"""
    if not FROZEN:
        return []
    done: list[str] = []
    for d in (DEFAULT_SKILLS_DIR, RUNS_DIR):
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            done.append(f"建目录 {d.name}/")
    if not any(DEFAULT_SKILLS_DIR.glob(f"*/{ENTRY_NAME}")) and SEED_SKILLS_DIR.is_dir():
        n = 0
        for src in sorted(p for p in SEED_SKILLS_DIR.iterdir() if p.is_dir()):
            # 只铺真能当技能的目录：有 `SKILL.md` 才算
            if not (src / ENTRY_NAME).is_file():
                continue
            shutil.copytree(src, DEFAULT_SKILLS_DIR / src.name, dirs_exist_ok=True)
            n += 1
        if n:
            done.append(f"放入 {n} 个示例技能")
    return done


# ──────────────────────────── 技能加载 ────────────────────────────


def _load_skills_dir(path_str: str) -> dict[str, object]:
    """从目录装技能包（相对路径按仓库根解析，子进程 cwd 变了也不受影响）。

    格式解析在 `sxagent.skill_pack`，这里只把路径定到仓库根。
    """
    # 局部导入是有意的：`trace` 一被导入就开日志文件，而本模块也被不该产生日志的地方导入。
    from sxagent import trace
    from sxagent.skill_pack import load_skills_dir

    root = Path(path_str)
    if not root.is_absolute():
        root = ROOT / root
    skills, warn = load_skills_dir(root)
    for w in warn:
        trace.note(f"技能加载：{w}")
    return skills


# ──────────────────────────── 装配 ────────────────────────────


def _install_skills(case: dict[str, object]) -> None:
    """技能集：从 `case.skills_dir` 读目录或从 `case.skills` 内联，换进注册表。

    先 `reset()` 再叠加，`extra` 为空时也要 reset，同进程多 case 才不会串味。
    """
    import sxagent.skill_pack
    import sxagent.skills

    registry.testing.reset()
    # 技能包目录表也要清：它不归注册表管，不清会残留上一个 case 的包目录
    sxagent.skill_pack.clear()
    extra: dict[str, object] = {}
    if case.get("skills_dir"):
        extra.update(_load_skills_dir(str(case["skills_dir"])))
    if case.get("skills"):
        extra.update({s["name"]: sxagent.skills.Skill(**s) for s in case["skills"]})
    if extra:
        if case.get("skills_mode", "extend") == "replace":
            registry.testing.replace(extra)
        else:
            registry.testing.extend(extra)


def _record_artifacts() -> list[dict[str, object]]:
    """产物写入流水：记谁在什么时候写了什么，用来判并发与归属。

    签名要跟着 `sxagent.artifacts.store_artifact` 走，漏改就是整轮 `TypeError`。
    """
    import sxagent.artifacts

    writes: list[dict[str, object]] = []
    orig_store = sxagent.artifacts.store_artifact

    def rec_store(title: str, summary: str, file_type: str, content: str) -> str:
        aid = orig_store(title, summary, file_type, content)
        writes.append(
            {
                "aid": aid,
                "title": title,
                "summary": summary,
                "type": file_type,
                "chars": len(content),
                "thread": threading.get_ident(),
                "t": round(time.time(), 3),
            }
        )
        return aid

    # `write_artifact` 的工具体在 `artifacts.py` 里按模块全局找 `store_artifact`，所以这里改得到
    sxagent.artifacts.store_artifact = rec_store
    return writes


def _record_run_tasks() -> list[dict[str, object]]:
    """执行器调用流水：记哪个线程在跑哪个 skill，用来判产物归属。

    两条路都汇到 `agent.dispatch`，那里走模块属性 `executor.run_task`（晚绑定）。
    """
    import sxagent.artifacts
    import sxagent.executor

    run_tasks: list[dict[str, object]] = []
    orig_run_task = sxagent.executor.run_task

    async def rec_run_task(skill, briefing: str = "", user_turns=None):
        # 前后产物库的差集 = 这次真写出来的。不能按线程归属：产物写入发生在 langgraph 的
        # 工具线程里，和 run_task 的线程对不上。签名跟着 `executor.run_task` 走。
        before = set(sxagent.artifacts.ids())
        out = await orig_run_task(skill, briefing, user_turns)
        run_tasks.append(
            {
                "thread": threading.get_ident(),
                "skill": skill.name,
                "t": round(time.time(), 3),
                # 这次真交付的产物（差集）
                "produced": sorted(set(sxagent.artifacts.ids()) - before),
            }
        )
        return out

    sxagent.executor.run_task = rec_run_task
    return run_tasks


def _count_tokens() -> UsageCounter:
    """token 与调用计数：三处值导入一起换（`agent` / `executor` / `llm`）。漏一处，那条路上
    的调用就不计费。"""
    import sxagent.agent
    import sxagent.config
    import sxagent.executor
    import sxagent.llm

    counter = UsageCounter()
    orig_get_model = sxagent.config.get_model

    def counting(temperature: float = 0.0):
        return orig_get_model(temperature).model_copy(update={"callbacks": [counter]})

    sxagent.agent.get_model = counting
    sxagent.executor.get_model = counting
    sxagent.llm.get_model = counting
    return counter


def install(case: dict[str, object]) -> dict[str, object]:
    """装好全部观测与注入，必须在 `build()` 之前调用。"""
    import sxagent.agent
    import sxagent.artifacts

    # ① 产物库清空 —— 全局 dict，不清就会带上上一轮
    sxagent.artifacts.clear()
    # ② 技能集
    _install_skills(case)
    # ③ / ③′ 两处打桩：产物写入流水、执行器调用流水
    writes = _record_artifacts()
    run_tasks = _record_run_tasks()
    # ④ 执行路径开关（case 可选），默认 auto（两条路都开）。想验执行器的正文交付就得让
    # 委派真的发生，靠碰运气压路径的断言必然脆。见 `sxagent/agent.EXEC_PATH`。
    sxagent.agent.EXEC_PATH = str(case.get("exec_path") or "auto")
    # ④′ token / 调用计数
    counter = _count_tokens()

    # ⑤ 故障注入不在这里：返回的 `fault` / `stream_fault` 是留给故障注入的旋钮 —— 往这两个
    #    dict 里写就行（同一个对象，运行记录照旧带着它们），目前还没有调用方。
    return {
        "counter": counter,
        "writes": writes,
        "run_tasks": run_tasks,
        "fault": {"left": 0, "fired": 0},
        "stream_fault": {"left": 0, "fired": 0},
    }


# ──────────────────────────── 快照 ────────────────────────────


#: 主 agent 上下文告警线：只告警，不判红。
#:
#: 读产物是主 agent 的合法能力，拿上下文大小判红等于罚一个合法能力，聊得久也会误伤；
#: 告警线给的是读数加一句提示，代价由看的人自己判断。
#:
#: 参照读数（历史跑批）：单技能隔离 517–1109 字，多技能长产物 7890–14832 字。
CTX_WARN_CHARS = 12000


def ctx_warn(total: int) -> str:
    """超告警线时给一句提示，正常时返回空串。只提示，不判红。"""
    if total <= CTX_WARN_CHARS:
        return ""
    return (
        f"⚠️ 主上下文超告警线 {CTX_WARN_CHARS} 字"
        "（看看是不是把产物正文读进来了 —— 依赖转交只需传 ID，不必读全文）"
    )


def _ctx_chars(messages: list[BaseMessage]) -> int:
    """主 agent 上下文体积，与 `sxagent/__main__._ctx_chars` 同一口径，两边可比。"""
    total = 0
    for m in messages:
        total += len(str(m.content))
        for c in getattr(m, "tool_calls", None) or []:
            total += len(str(c.get("args", "")))
    return total


def _last_ai_text(state: dict[str, object]) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage) and str(m.content).strip():
            return str(m.content)
    return ""


def _sys_prompt_chars() -> int:
    """主 agent 的 system prompt 体积。

    它不在 `state["messages"]` 里，`main_ctx_chars` 量不到；真实上下文是两者之和。
    """
    import sxagent.agent

    # 口径必须与 `agent.build_agent` 拼的那份一致，所以两边都调 `registry.catalog()`。
    return len(sxagent.agent._MAIN_PROMPT) + len("\n\n可用技能：\n" + registry.catalog())


def _msg_record(m) -> dict[str, object]:
    """一条 LangChain 消息 → 记录里那份（原文，不截断）。交接对象读
    `additional_kwargs`，不从 `content` 反推。"""
    kwargs = getattr(m, "additional_kwargs", None) or {}
    return {
        "kind": type(m).__name__,
        "content": str(m.content),
        "tool_call_id": getattr(m, "tool_call_id", None),
        #: 交接对象：回执的结构化形态，记录层读它。
        "handoff": kwargs.get("handoff"),
        #: 平台注入的标记（`sxagent.handoff.INJECTED`），标明这条不是主 agent 说的。
        #: 透传注入的是 `AIMessage`，不打标记就会被标成"主 agent 输出"。
        "injected_by": kwargs.get("injected_by"),
        "tool_calls": [
            {"id": c.get("id"), "name": c.get("name"), "args": c.get("args")}
            for c in (getattr(m, "tool_calls", None) or [])
        ],
    }


def snapshot(state: dict[str, object], user: str, prev_len: int, idx: int, wall_s: float) -> dict[str, object]:
    from sxagent import trace
    from sxagent import artifacts

    msgs = state.get("messages") or []
    new = msgs[prev_len:]
    delegates = [
        c.get("name")
        for m in new
        for c in (getattr(m, "tool_calls", None) or [])
        if c.get("name") == "delegate"
    ]
    #: 全部工具调用（名字 + 参数摘要）：主 agent 读了哪个产物、写了什么，这里直接可查。
    tool_calls = [{"name": n, "args": a} for n, a in trace.tool_calls_of(new)]
    return {
        "turn": idx,
        "user": user,
        #: 这句话是谁给的：`scripted`（脚本预置）或 `human`（交互式入口当场打的）。
        #: 日志要如实标，读的人不能把预置的答复当成真实用户行为。
        "input_source": "scripted",
        "reply": _last_ai_text(state),
        "tasks": state.get("tasks") or [],
        "main_ctx_chars": _ctx_chars(msgs),
        "sys_prompt_chars": _sys_prompt_chars(),
        "artifacts_chars": artifacts.chars(),
        "artifacts": {k: len(v["content"]) for k, v in artifacts.items()},
        "delegate_calls": delegates,
        "tool_calls": tool_calls,
        #: 本轮全部新消息（原文，不截断），逐轮审计文件靠它渲染运行全记录。
        #: `tool_call_id` 与 `tool_calls[].id` 配对，用来判这条返回是谁的。
        "messages": [_msg_record(m) for m in new],
        "new_msg_kinds": [type(m).__name__ for m in new],
        "log": state.get("log") or [],          # 渲染记录（给人看 / `_run.md`）
        "actions": state.get("actions") or [],  # 结构化事实（断言读它，不读 log）
        "wall_s": wall_s,
    }


# ──────────────────────────── 落盘：给人工审计用 ────────────────────────────


def _slug(s: str, n: int = 40) -> str:
    """转发 `sxagent.artifacts.artifact_slug` —— 文件名怎么起，落盘和这里必须是同一套。"""
    from sxagent.artifacts import artifact_slug

    return artifact_slug(s, n)


def dump_artifacts(out_dir: Path, case_id: str, rep: int, result: dict[str, object]) -> Path:
    """把产物原文落盘，主 agent 上下文里没有的东西，人能看到。

    只有把原文摊开给人看，"这份完不完整"才有答案。落 `runs/<tag>/artifacts/<case>__r<rep>/`，
    每个产物一个文件，外加一份 `_index.md`。
    """
    from sxagent import artifacts

    d = out_dir / "artifacts" / f"{case_id}__r{rep}"
    d.mkdir(parents=True, exist_ok=True)
    if not artifacts.count():
        (d / "_index.md").write_text("# （本次没有产物）\n", encoding="utf-8")
        return d

    rows = ["| 产物 | 标题 | 字数 | 类型 | 文件 |", "|---|---|---|---|---|"]
    for aid, a in artifacts.items():
        # 类型和文件名都取记录里那份（`store_artifact` 按声明的类型定好的），
        # 这里另算一遍会和它当场落盘写的那份对不上。
        fn = a.get("file") or f"{aid}_{_slug(a['title'])}{a.get('ext', '.md')}"
        # 落盘就是库里那份，一字不加，和 `store_artifact` 当场落盘同一条规矩。
        (d / fn).write_text(a["content"], encoding="utf-8")
        rows.append(
            f"| `{aid}` | {a['title']} | {len(a['content'])} "
            f"| {a.get('ext', '.md').lstrip('.')} | [{fn}]({fn}) |"
        )

    (d / "_index.md").write_text(
        f"# {case_id} · 第 {rep} 遍 · 产物原文\n\n"
        f"共 {artifacts.count()} 个，合计 {artifacts.chars()} 字。\n\n"
        + "\n".join(rows)
        + "\n\n> **这是主 agent 上下文里没有的原文** —— 判「完整性」以这里为准。\n",
        encoding="utf-8",
    )
    return d


#: 出口分类 → 给读日志的人看的记号。
#:
#: 判据不在本文件里，从 `sxagent.handoff`（段名与出口分类的唯一来源）推。
#: 这里手抄一份对应关系，回执改格式就静默失效，所以覆盖性要有检查盯着。
RECEIPT_KINDS: dict[str, str] = {
    "completed": "✅ **交付回执**（执行器已写入产物库 —— 平台只说这个事实）",
    "waiting": "❓ **执行器申报要信息**（`ask_user`：平台先试答，答不上转达给用户）",
    "failed": "⚠️ **既没交付、也没说要什么**（推满轮次，交给主 agent 判）",
    # 加新出口时，这张表和 `RECEIPT_LABEL` 要同时补，漏一个就是静默回落成裸 key。
    "fault": "⚠️ **平台故障**（这一轮**没跑成** —— 不是执行器干不了）",
    "rejected_skill": "⛔ **被平台拒绝**：没有这个技能",
    "rejected_artifact": "⛔ **被平台拒绝**：产物 ID 不存在",
}


#: 工具返回的来源（读日志的人最需要知道的一条：这段字是谁写的）
TOOL_SOURCE: dict[str, str] = {
    "delegate": "平台（`delegate` 的回执：**平台的框架话 + 执行器的原话**拼在一起）",
    "read_skill": "平台（技能原文 —— 主 agent 主动读的，渐进式披露第二层）",
    "read_skill_file": "技能包（技能自带文件的全文 —— 渐进式披露第三层）",
    "read_artifact": "产物库（**产物正文全文**）",
    "write_artifact": "产物库（写入回执）",
    "list_artifacts": "产物库（清单，不含正文）",
}


#: 段的标签（给读日志的人看）。段名本身不在这里，从 `sxagent.handoff.SECTION` 取。
RECEIPT_LABEL: dict[str, str] = {
    "deliver": "平台的框架话 · 这条回执属于哪种出口",
    "asked": "**它申报要的那句原话**（`ask_user` 的参数，未经转述）",
    "produced": "平台给的产物清单（**不含正文**）",
    "report": "**执行器的原话**（它说了什么、在问什么）",
    "agent_answer": "**主 agent 补的信息**（它说的）",
    "next": "平台给主 agent 的指示",
    # 措辞与 `handoff.SOURCE_NOTE["fault"]` 同一句，一套别分叉
    "fault": "**平台说的** —— 不是执行器说的",
    "not_dispatched": "平台的框架话 · 这次**没有派发**",
    "available_skills": "平台列出的可用技能",
    "available_artifacts": "平台列出的现有产物",
}


def _injected_label(tag: str | None) -> str:
    """平台注入的标记 → 记录层怎么标这条消息。读标记，不读内容。"""
    if not tag:
        return ""
    from sxagent.handoff import INJECTED

    return "📮 " + INJECTED.get(tag, f"（未知的注入标记：{tag}）")


def _receipt_sections(handoff: dict[str, object]) -> list[tuple[str, str]]:
    """回执对象 → 哪一段、写了什么。读对象，不读文本。

    "这条回执是不是派发被拒"由 `handoff.receipt` 判（拒绝口那张表在它那儿）：这里不按前缀
    近似 —— 前缀认得出表外的 kind，认出来了也是错的。
    """
    from sxagent.handoff import HOPS
    from sxagent.handoff.receipt import is_rejection

    sections = handoff.get("sections") or {}
    keys = (HOPS["receipt_rejected"] if is_rejection(str(handoff.get("kind") or ""))
            else HOPS["receipt"])
    return [(RECEIPT_LABEL.get(k, k), sections[k]) for k in keys if sections.get(k)]


def _turn_summary(turns: list[dict[str, object]], i: int) -> list[str]:
    """每轮开头的小结：有没有反问、用户输入是谁给的、平台做了什么。"""
    t = turns[i]
    prev = turns[i - 1] if i > 0 else None
    human = t.get("input_source") == "human"
    out = ["**本轮小结**", ""]
    if prev is None:
        out.append("- 上一轮：无（首轮）")
    else:
        waiting = [x for x in (prev.get("tasks") or []) if x.get("stage") == "waiting_for_input"]
        # 读 `actions`（对象），不读 `log`（句子），见 `State.actions` 的注释
        relayed = [a for a in (prev.get("actions") or []) if a.get("did") == "relay"]
        if waiting:
            who = "、".join(f"`{x['id']}`（{x['skill']}）" for x in waiting)
            out.append(
                f"- 上一轮：系统**反问**了 —— {len(waiting)} 个 Task 在要信息：{who}"
                + ("；平台已**透传**给用户" if relayed else "；⚠️ 没看到透传")
            )
        else:
            out.append("- 上一轮：系统没有反问（没有 Task 在要信息）")
    out.append(
        "- 本轮输入：**真人当场输入**（`app.chat` 交互式跑，不是预置脚本）"
        if human
        else "- 本轮输入：**脚本预置**（跑批 case 里写死的字符串）"
        "，**不是真人即时输入**"
    )
    # 平台动作读对象自己渲染一句，不切日志句子。
    own = (
        (t.get("actions") or [])[len(prev.get("actions") or []) :] if prev else (t.get("actions") or [])
    )
    fed = [a for a in own if a.get("did") == "feed"]
    if fed:
        who = "、".join(f"`{x}`" for x in fed[0]["tasks"])
        out.append(f"- 平台动作：把用户原话**原样**投给了 {len(fed[0]['tasks'])} 个待答任务（{who}）")
    elif prev is not None and any(x.get("stage") == "waiting_for_input" for x in (prev.get("tasks") or [])):
        out.append("- 平台动作：⚠️ 上一轮有待答，但本轮没看到投递记录")
    out.append("")
    return out


class _ToolPairing:
    """把工具调用和它的返回配起来：优先按 `tool_call_id`，退化为按顺序。

    并行调用时按顺序配会串，所以 id 优先；老数据没有 id 才退回按顺序，那是有损的。
    """

    def __init__(self) -> None:
        #: tool_call_id → 工具名（用来判返回的来源）
        self._names: dict[str, str] = {}
        #: tool_call_id → 调用参数（"读了哪份"要靠它）
        self._args: dict[str, dict[str, object]] = {}
        #: 还没配上返回的调用（老数据没有 id 时按顺序配）
        self._pending: list[tuple[str, dict[str, object]]] = []

    def remember(self, call: dict[str, object]) -> None:
        """记一次调用 —— `id` 有就按 id 记，同时进顺序队列兜底。"""
        args = call.get("args") or {}
        if call.get("id"):
            self._names[str(call["id"])] = str(call.get("name"))
            self._args[str(call["id"])] = args
        self._pending.append((str(call.get("name")), args))

    def take(self, tid: object) -> tuple[str, dict[str, object]]:
        """取这条返回对应的调用 —— 优先按 `tool_call_id` 精确配对，退化为按顺序。"""
        key = str(tid) if tid else ""
        if key and key in self._names:
            name = self._names.pop(key)
            args = self._args.pop(key, {})
            for i, (nm, _) in enumerate(self._pending):
                if nm == name:
                    self._pending.pop(i)
                    break
            return name, args
        return self._pending.pop(0) if self._pending else ("?", {})


def _render_ai(m: dict[str, object], pairing: _ToolPairing, n: int) -> tuple[list[str], int]:
    """主 agent 的一段话和它调的工具，每条各占一个编号。"""
    out: list[str] = []
    content = str(m.get("content") or "")
    if content.strip():
        n += 1
        # 按标记判来源，不按内容像不像：透传注入的 AIMessage 里装的是执行器的原话。
        who = _injected_label(m.get("injected_by"))
        out += [
            f"**{n}. {who or '🤖 主 agent 输出'}**"
            + ("" if who else "（它自己说的话 —— 决定下一步做什么）"),
            "",
            "```",
            content.strip(),
            "```",
            "",
        ]
    for c in m.get("tool_calls") or []:
        n += 1
        args = c.get("args") or {}
        pairing.remember(c)
        out += [
            f"**{n}. 🔧 主 agent 调用工具 `{c.get('name')}`**"
            + ("　→ **交接给执行器**" if c.get("name") == "delegate" else ""),
            "",
            "```",
            *[f"{k} = {v!r}" for k, v in args.items()],
            "```",
            "",
        ]
    return out, n


def _render_tool(m: dict[str, object], pairing: _ToolPairing, n: int) -> tuple[list[str], int]:
    """四种形状各有处置：`read_*`（正文另有出处，只写读了啥）、`delegate`（回执对象在
    `additional_kwargs` 上）、没有对象的 `delegate`（机制破了，如实说）、其余（原样照录）。"""
    out: list[str] = []
    content = str(m.get("content") or "")
    n += 1
    tool, target = pairing.take(m.get("tool_call_id"))
    handoff = m.get("handoff")
    mark = RECEIPT_KINDS.get((handoff or {}).get("kind", ""), "")
    out += [
        f"**{n}. ↩️ 返回给主 agent**（工具 `{tool}`）{mark}",
        "",
        f"<sub>来源：{TOOL_SOURCE.get(tool, '平台')}</sub>",
        "",
    ]
    if tool in ("read_artifact", "read_skill", "read_skill_file"):
        # 正文和原文不进日志，只写读了啥、谁读的：产物落盘在 `art_*`，技能原文在 `skills/`，
        # 都另有出处。三个读工具各写各的"读了啥"——读技能文件时，路径比技能名更该记。
        if tool == "read_artifact":
            what, where = f"产物 `{target.get('artifact_id', '?')}`", "见同目录 `art_*` 落盘文件"
        elif tool == "read_skill":
            what, where = f"技能 `{target.get('skill_name', '?')}`", "见 `skills/`"
        else:
            what = (f"技能文件 `{target.get('skill_name', '?')}"
                    f"/{target.get('path', '?')}`")
            where = "见 `skills/`"
        out += [
            f"　**主 agent 读了{what} 的全文**"
            f"（{len(content)} 字，原文不在这里复述 —— {where}）",
            "",
        ]
        return out, n
    # `delegate` 的返回按对象渲染，不解析文本：
    # 平台在 `_cmd` 里把回执对象挂在 `additional_kwargs["handoff"]` 上搭同一趟车。
    parts = _receipt_sections(handoff) if (tool == "delegate" and handoff) else []
    if parts:
        for label, text in parts:
            # label 自带加粗记号，这里不要再包一层
            out += [f"　{label}：", "", "```", text, "```", ""]
    elif tool == "delegate":
        # 走不到：所有 delegate 返回都带对象。真到了这里说明机制破了 ——
        # 如实说出来，不要退回解析文本。
        out += [
            "⚠️ **这条回执没有结构化对象**（机制破了，不该发生）—— 原文照录，不做解析：",
            "",
            "```",
            content,
            "```",
            "",
        ]
    else:
        out += [
            "```",
            content if len(content) <= 1200 else content[:1200] + f"\n…（共 {len(content)} 字）",
            "```",
            "",
        ]
    return out, n


def _render_system(m: dict[str, object], n: int) -> tuple[list[str], int]:
    """平台注入：每轮开头自动塞给主 agent 的那条任务台账。"""
    n += 1
    return [
        f"**{n}. 📌 平台注入**（每轮开头自动塞给主 agent 的**任务台账** —— "
        "它靠这个知道已经派过什么、谁在要信息）",
        "",
        "```",
        str(m.get("content") or ""),
        "```",
        "",
    ], n


def _render_unknown(m: dict[str, object], n: int) -> tuple[list[str], int]:
    """未知消息类型的兜底。切了必须标，否则读的人分不清"它就这么多"和"被切了"。"""
    content = str(m.get("content") or "")
    n += 1
    return [
        f"**{n}. {m.get('kind', '?')}**",
        "",
        "```",
        content if len(content) <= 800
        else content[:800] + f"\n…（**平台截断：还有 {len(content) - 800} 字未显示**）",
        "```",
        "",
    ], n


def _render_messages(turn: dict[str, object]) -> list[str]:
    """把本轮消息渲染成带来源与类型的记录。

    每条都要能看出谁说的和什么性质。它只做分发，配对状态在 `_ToolPairing` 里。
    """
    out: list[str] = []
    pairing = _ToolPairing()
    n = 0
    for m in turn.get("messages") or []:
        kind = m.get("kind", "?")
        if kind == "HumanMessage":
            continue  # 上面"用户输入"已经有了
        if kind == "AIMessage":
            lines, n = _render_ai(m, pairing, n)
        elif kind == "ToolMessage":
            lines, n = _render_tool(m, pairing, n)
        elif kind == "SystemMessage":
            lines, n = _render_system(m, n)
        else:
            lines, n = _render_unknown(m, n)
        out += lines
    return out


def _turn_header(turns: list[dict[str, object]], i: int, t: dict[str, object]) -> list[str]:
    """一轮的头：标题、那行"几件事"、用户输入、上下文读数（超线加告警）。"""
    lines = [
        f"## 第 {t['turn']} 轮",
        "",
        *_turn_summary(turns, i),
        "### 用户输入（**真人当场输入**）"
        if t.get("input_source") == "human"
        else "### 用户输入（**测试脚本预置**，不是真人即时输入）",
        "",
        f"> {t['user']}",
        "",
        f"<sub>主上下文 {t['main_ctx_chars']} 字（不含 system prompt）"
        f"｜产物库 {t['artifacts_chars']} 字｜delegate {len(t['delegate_calls'])} 次</sub>",
        "",
    ]
    warn = ctx_warn(t["main_ctx_chars"] + t.get("sys_prompt_chars", 0))
    if warn:
        lines += [f"> {warn}", ""]
    return lines


def _task_table(tasks: list[dict[str, object]]) -> list[str]:
    """Task 台账：一张表，加每个 Task 的原话（`<details>` 里给全文）。"""
    lines = [
        "### Task 台账",
        "",
        # "进度 / 还缺"那列在判定退休后没有来源了，整列删掉；
        # 执行器的原话在下面每个 Task 的 `<details>` 里有全文。
        "| Task | 技能 | 状态 | **产出** | **输入引用** |",
        "|---|---|---|---|---|",
    ]
    for x in tasks:
        lines.append(
            f"| `{x['id']}` | {x['skill']} | {x['stage']} "
            f"| {', '.join(x.get('artifacts') or []) or '—'} "
            f"| {', '.join(x.get('refs') or []) or '—'} |"
        )
    lines.append("")
    for x in tasks:
        if x.get("report"):
            lines += [
                f"<details><summary>{x['id']} 执行器原话（交接给主 agent 的就是它）</summary>",
                "",
                x["report"],
                "",
                "</details>",
                "",
            ]
    return lines


def dump_run_record(out_dir: Path, case_id: str, rep: int, result: dict[str, object]) -> None:
    """落盘运行全记录（一个 run 一份），人工审计"过程对不对、交接了什么"。

    结构：概览 → 运行全记录（按轮）→ 软判断留档 → 原始 trace 附录，后两段由父进程追加。
    """
    d = out_dir / "artifacts" / f"{case_id}__r{rep}"
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {case_id} · 第 {rep} 遍 · 运行全记录",
        "",
        "> 一份文件读完：**概览 → 逐轮全记录 → 软判断留档 → 原始 trace 附录**。",
        "",
        f"- 耗时 {result['wall_s']}s｜token {result['usage'].get('total_tokens', 0)}"
        f"｜LLM 调用 {result['usage'].get('llm_calls', 0)}",
        "",
    ]
    if result.get("error"):
        err = result["error"]
        cut = len(err) - 3000
        lines += [
            "## ⚠️ 这次运行没跑成",
            "```",
            err[:3000] + (f"\n…（**平台截断：还有 {cut} 字未显示**，全文见同目录 `results.json`）"
                          if cut > 0 else ""),
            "```",
            "",
        ]
    for _i, t in enumerate(result["turns"]):
        lines += _turn_header(result["turns"], _i, t)
        # ── 本轮全部消息，按发生顺序（用户输入 / 平台注入 / AI 输出 / 工具调用与返回）──
        lines += ["### 本轮消息（按发生顺序）", ""]
        lines += _render_messages(t)
        # 平台日志：机制是否生效的直接证据（投递 / 透传 / 取代 / 注入 / 工具）
        if t.get("log"):
            lines += ["### 平台日志", "", "```"] + [f"  {x}" for x in t["log"]] + ["```", ""]
        if t["tasks"]:
            lines += _task_table(t["tasks"])
    (d / "_run.md").write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────── 原始 trace 附录 ────────────────────────────


def append_trace(result: dict[str, object], tag: str) -> None:
    """把该 run 的原始 trace 作为附录并进它自己的 `_run.md`，一份文件读完。

    它和 `dump_run_record` 是一件事，交互式入口也要用，所以住在这儿。
    """
    src = result.get("trace_log")
    if not src or not Path(src).exists():
        return
    d = RUNS_DIR / tag / "artifacts" / f"{result['case']}__r{result['rep']}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "_run.md").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n\n## 附录 · 原始 trace\n\n"
            "执行器每一轮、被推回去时的提示词、判定逐条理由、以及软判断的逐字 "
            "prompt/response，都在这里。\n\n"
            "````\n"  # 四个反引号：内容里本身就有 ``` （技能指令、产物正文）
        )
        fh.write(Path(src).read_text(encoding="utf-8", errors="replace"))
        fh.write("\n````\n")
