"""台账（平台 → 主 agent）：每轮开头把"派过什么、干到哪了"摆到它面前。

`node_prepare` 注入的那条 `SystemMessage`。不注入的话主 agent 不知道派过什么，用户补一句
它会把三个任务全部重跑。

## 不变量

1. 只提供事实，不替它做决定。"要不要重跑"是理解问题。
2. 说事实、不下判决：三态由平台按结构定，但「已完成」「失败」是判定的口吻，"够不够"由主
   agent 判。
3. 故障标记压过 `stage`：出故障的 Task `stage` 也是 `failed`，照 stage 说就是假话，而这两种
   情况主 agent 该做的事完全不同。
4. 切了必须标；全文在哪由 `run_by` 决定，不许写死，也不许现编——指向"存在但内容不对"的
   出处比指向不存在的更坏。"""

from __future__ import annotations

from sxagent import trace
from sxagent.handoff import DELIVERED_DESPITE_FAULT, TaskStage, render_hop
from sxagent.handoff.briefing import Task

#: `Task.stage` 的全部取值 —— 与 `OutcomeStatus` 前三个同名；唯一的映射是
#: `incomplete → failed`。`superseded` 是"不用管了"，台账里不展开细节、只列一行。
#: 键 = `TaskStage`，打错 mypy 会报。
STAGE_MARK: dict[TaskStage, str] = {
    "running": "⏳ 进行中",
    "completed": "✅ 它交付了产物",
    "waiting_for_input": "⚠️ 它在要信息",
    "failed": "❌ 它没交付、也没说要什么",
    "superseded": "🔁 已被新的委派取代（不用管了）",
}

#: 需要展开细节（执行器原话 `report`）的状态。
NEEDS_DETAIL = ("waiting_for_input", "failed")

#: 平台故障那一档的标记，压过 `stage` 标记（故障的 Task `stage` 也是 `failed`）。
#: 键 = `FaultKind`（`""` 除外），不认识就回落 `transient`。事实来源是 `Task.fault`，
#: 和回执的【平台故障】是同一个事实的两个视图。
FAULT_MARK: dict[str, str] = {
    "transient": "⚠️ 平台故障（这一轮**没跑成** —— 原样再委派一次即可）",
    "rejected": "⚠️ 上游**拒绝了这次请求**（原样重发会再被拒 —— **换措辞/改内容**再委派）",
    "timeout": "⚠️ **时间用完了**（这一趟没跑完就被叫停 —— 原样再委派可以；"
    "**老超时就说明活太大**，拆小一点再委派）",
}


#: 台账截断时"全文在哪"，由 `Task.run_by` 决定：`delegate` 在回执里，`feed` 没有回执。
#: 指错了比指不到更坏，所以 `feed` 那条直说"你手上没有"。
_SRC = {
    "delegate": "全文见那次的回执（`delegate` 的返回）",
    "feed": "全文**不在你的上下文里** —— 这轮是平台投递你的答复后重跑的，没有回执",
}
#: `run_by` 缺失（比如手写 state / fixture 里注入的 Task）时的默认 —— fail-safe 到"没有全文"。
_SRC_DEFAULT = "全文**不在你的上下文里**"


def brief(text: str, limit: int, what: str = "长文本", *, run_by: str = "") -> str:
    """台账里的长文本 —— 要短就得说清"这是短的"（切了必须标）。

    顺手把换行压成空格：台账是一行一条的列表，多行值会把缩进结构撑坏 —— 这不算内容损失
    （只动了空白），但别在这里做别的加工。
    """
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    # 切了一刀就报一刀：标记在文本里（读的人看得见），这一行在日志里（复盘的人看得见）。
    src = _SRC.get(run_by, _SRC_DEFAULT)
    trace.cut(f"台账·{what}", limit, len(text), see=src)
    return f"{text[:limit]}……【平台截断：还有 {len(text) - limit} 字，{src}】……"


def render_ledger(tasks: list[Task]) -> str:
    """把 Task 列表渲染成给主 agent 的台账（`node_prepare` 注入的那条 `SystemMessage`）。

    调用方保证 `tasks` 非空（空的时候 `node_prepare` 直接不注入 —— 闲聊轮次里没有这段）。
    走唯一那条渲染路（`render_hop`）：段名【平台提示】登记在 `SECTION` 里，不许就地拼字符串。
    """
    # 缩进一律相对写 —— 下面这些行会被 `render_hop` 整块缩 2 格。
    body: list[str] = [f"你已委派出去的工作（共 {len(tasks)} 项）："]
    for t in tasks:
        # 平台故障压过 stage 标记（见 `FAULT_MARK`），标记按故障的类选；
        # 取不到 kind 回落 `transient`（保守）。
        faulted = bool(t.get("fault"))
        mark = (
            FAULT_MARK.get(str(t.get("fault_kind") or ""), FAULT_MARK["transient"])
            if faulted
            else STAGE_MARK.get(t.get("stage", ""), t.get("stage", "?"))
        )
        arts = f" → {', '.join(t['artifacts'])}" if t.get("artifacts") else ""
        body.append(f"- {t['id']}  「{t['skill']}」  {mark}{arts}")
        # 没完成的任务，把执行器原话也带上 —— 只说"没完成"，主 agent 没法决定怎么处置。
        # `run_by` 一路传下去：这两段都来自同一轮。
        by = str(t.get("run_by") or "")
        if faulted:
            # 故障时打故障原因、不打「执行器说」：它一个字都没说，照打就是冒号后面空白。
            line = f"  - 平台故障：{brief(t['fault'], 150, '平台故障', run_by=by)}"
            # 交过产物的那趟要说清 —— 故障话术默认"什么都没交"，账如实之后这个假设不成立。
            if t.get("artifacts"):
                line += f"；{DELIVERED_DESPITE_FAULT.format(len(t['artifacts']))}"
            body.append(line)
        elif t.get("stage") == "waiting_for_input":
            # 不重印问题：全文由【待答】100% 覆盖，这里重印的那份是截断的 ——
            # 撤截断的那份、留全文那份，信息不丢；撤反了才丢（`feed` 那条路上没有回执）。
            # 打 `asked`、不打 `report`：主 agent 需要知道的是"它在等什么"。
            # 不许点名那个段名（写死【待答】是第二份定义）—— 描述，不点名。
            q = str(t.get("asked") or "").strip()
            if q:
                body.append(f"  - 它要的信息：（{len(q)} 字，**平台已原样转达**，此处不重贴）")
            else:
                # 不许留一个冒号后面空白 —— 说清是"没有"。
                body.append("  - 它要的信息：（**这一条没留下问题原话**）")
        elif t.get("stage") in NEEDS_DETAIL:
            body.append(f"  - 执行器说：{brief(t['report'], 150, '执行器说', run_by=by)}")

    waiting = [t for t in tasks if t.get("stage") == "waiting_for_input"]
    # 这里不再叫主 agent 去重新委派 —— 待答的投递和转达都由平台做。
    body.append(
        "**「在要信息」的那些，平台已经负责把问题转达给用户、也会把用户的答复投回去** ——"
        "你**不用**重新委派它们，更不要自己替用户编一个答案。"
        if waiting
        else "（用户补充信息时，**只重跑受影响的任务**，已完成且不受影响的不动。）"
    )
    # 走唯一那条渲染路：段名登记在 `SECTION` 里，不许就地拼字符串。
    return render_hop("ledger", {"ledger": "\n".join(body)})
