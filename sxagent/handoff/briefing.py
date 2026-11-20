"""委托单（平台 → 执行器）：Task 的结构与渲染模板。

平台每次交给执行器的都是完整一份，不是增量：Task / Turn / QA 三个 schema 加
`render_briefing`，外加结果落回 Task 的 `absorb_outcome` / `settle` 和哑待答的显式取代
`moot_older_waiting`。执行器用完即弃，它该知道的全部都得由平台重新交代。

## 不变量

1. 段序 = 读的人理解这件事的顺序：要做什么 → 用户原话 → 做过什么问过什么 → 产出了什么 →
   接着干什么。一条时间序，不按来源分段，也不"最新在前"。
2. 来源必须标：执行器要分得清哪句是用户说的、哪句是模型补的，模型补的不能当成用户的事实。
3. Task 的每个字段要么出现在渲染里，要么在排除名单里，加了字段忘了渲染要被检查出来。
4. 两处硬机制都是确定性的：排版，和去重（逐字比较我们自己的两个字段，不从文本反结构化）。
5. 缺的东西显式写出来，说不清就不动：有 ID 取不到算异常；哑待答的取代要两条结构事实都
   成立，fixture 和手写 state 一律不碰。一问一答成对（`report` / `asked` 都是覆盖写）。"""

from __future__ import annotations

from typing import Literal, TypedDict

from sxagent import trace
from sxagent.artifacts import DEFAULT_STORE, ArtifactStore
from sxagent.handoff import (
    BODY,
    FaultKind,
    Outcome,
    OutcomeStatus,
    TaskStage,
    body_lines,
    section_head,
    working_data_lines,
)

# ──────────────────────────── schema：委托单上有什么 ────────────────────────────


class Turn(TypedDict):
    """一轮用户原话（平台硬搬的快照，不经任何转述）。

    `after` 是紧邻的前一条 AI 消息：用户答「A」时，脱离它回答的那句就没意义。
    """

    n: int
    text: str
    after: str


class QA(TypedDict):
    """一问一答成对，免得「答」脱离了「问」。

    `report` / `asked` 都是覆盖写的，只存答案的话，第二次重入之后旧答案就成了没有指涉
    的孤儿。`q` 取 `asked`（`ask_user` 的原话）不取 `report`；取不到存空串，渲染时明说。
    """

    #: 谁答的。取值就这两个 —— 收成枚举，不靠注释约定。
    by: Literal["用户", "主 agent"]
    q: str  # 执行器问的原话（`ask_user` 的参数 —— 取不到时是空串，不是 `report`）
    a: str  # 答复


class Task(TypedDict, total=False):
    """委托单：一次委派出去的工作的全部已知。

    `total=False` 是如实：注入的 Task 可能只有部分字段。没有运行时校验，靠 `delegate`
    建全 ＋ 断言兜底。
    """

    id: str
    skill: str
    task: str  # 主 agent 写的"这次要什么"（它说的，可能加了料）
    stage: TaskStage  # 五态，词汇表在 `handoff/__init__`
    artifacts: list[str]  # 产出的产物 ID
    #: 这套工作留下的中间数据 ID（会话内暂存，不是交付物）—— 与 `artifacts` 两本账，别混。
    working_data: list[str]
    refs: list[str]  # 输入引用的产物 ID（两个方向，别混）
    report: str  # 执行器最后一轮的原话（覆盖写）
    #: 它申报要的那条信息 —— `ask_user(question)` 参数里的原话（覆盖写）。
    #: 结构性事实必须落进 Task，下游才有对象可用（不然只能用散文）—— 与 `fault` 同一个病。
    asked: str
    #: `ask_user` 给的备选（空 = 没给）。调用方（通道）可能把它渲染成按钮 —— 是结构，
    #: 谁都不许回头从 `asked` 那句话的括号里解析。
    asked_options: list[str]
    #: 平台故障的原因（非空 = 这一轮根本没跑成，不是执行器干不了；空 = 没故障）。
    #: 事实只有这一处定义：回执的【平台故障】与台账的标记都是它的视图。
    fault: str
    #: 平台故障的"类"（`FaultKind`）—— 台账据此选引导句（原样再委派 vs 换措辞），
    #: 不许从 `fault` 文本反解析。与 `fault` 同生共死：每次 `settle` 覆盖写（正常跑完清空）。
    fault_kind: FaultKind
    qa: list[QA]
    user_turns: list[Turn]  # 派发那一刻的快照
    #: 这个 Task 最近一次运行是谁起的 —— `"delegate"`（主 agent 派发）或
    #: `"feed"`（用户答复后平台投递重跑）。台账截断时要说清"全文在哪"，而两条路的答案
    #: 不一样（投递那条路没有回执）—— 指错会让主 agent 翻到上一轮的回执。
    run_by: Literal["delegate", "feed"]


# ──────────────────────────── 渲染：注入前那一瞬才拼 ────────────────────────────


#: 索引行里每轮露出多少个字（`_one_line` 的默认 limit）。它是「一行多长」，不是「切到多少」。
_one_line_limit = 40


def asked_text(t: Task) -> str:
    """待答任务 → 用户该答的那一句（`Task.asked`）。取不到就显式说清。

    只负责"取哪句"不负责排版：两个视图载体不同，各排各的。返回单行，能直接用在
    `input()` 的提示符里。

    `asked` 为空只有两个来源（fixture、手写 state），产品路径上不该发生。退回 `report`
    会把"数据缺失"藏成"看起来正常"。
    """
    q = _flat(t.get("asked"))
    if q:
        return q
    # 不许静默退回 `report`：说清"没有"，再把它最后说的那段附上。
    return (
        f"（**这一条没留下问题原话**；它最后说的是："
        f"{_flat(t.get('report')) or '（它一个字都没说）'}）"
    )


def _flat(text: object) -> str:
    """`_one_line` 压成一行之后的样子 —— 压平与比长度必须用同一个口径。

    否则换行多的短消息会被误判成"超长"（`_one_line` 先压平再判长度）。
    """
    return " ".join(str(text or "").split())


def _one_line(text: str, limit: int = _one_line_limit) -> str:
    """压成一行 + 截到 limit 字，索引里一行一轮。

    这是索引不是截断：装得下的这一行就是全文，装不下的全文走 `read_user_turn(轮次)`。
    所以不报 `trace.cut`、也不加"切了"的标记，`（X 字）` 和 `…` 已经说清。
    """
    flat = _flat(text)
    return flat if len(flat) <= limit else flat[:limit] + "…"


def turn_lines(turns: list[Turn]) -> list[str]:
    """用户原话索引：一行一轮——序号 · 字数 · 开头 · 它在回答什么。

    常驻的是每轮一行，成本与文本量无关；全文由 `read_user_turn(n)` 按需取。用户粘五万字，
    这一层也只涨一行。平台不猜哪几轮相关，只保证原话一字不改地都在索引里。
    """
    out: list[str] = []
    for t in turns:
        n, text = t.get("n"), str(t.get("text") or "")
        out.append(f"{BODY}- 第 {n} 轮（{len(text)} 字）：{_one_line(text)}")
        after = str(t.get("after") or "").strip()
        if after:
            # 短应答必须带上它在回答什么：光一个「A」既解读不了、也核对不了主 agent 的
            # 转述 —— 而这一段存在的全部理由就是对冲转述失真。
            out.append(f"{BODY}  ↳ **它在回答的上一句**：「{_one_line(after)}」")
    return out


def _index(aids: list[str], store: ArtifactStore) -> list[str]:
    """产物 ID 列表 → 清单行（含缺失）。有 ID 而取不到是异常，要显式写出来。"""
    out: list[str] = []
    for aid in aids:
        a = store.get(aid)
        if not a:
            out.append(f"- {aid}（**产物库里没有这一条**）")
            continue
        out.append(
            f"- {aid}《{a['title']}》（{len(a['content'])} 字）"
            f"—— {a.get('summary') or '（无梗概）'}"
        )
    return out


def _index_block(hop: str, aids: list[str], tail: str, store: ArtifactStore) -> str | None:
    """一个产物清单段（已产出 / 可读），清单空就不渲染这一块。两段形状一样，只这一处拼。"""
    ids = [a for a in aids if a]
    if not ids:
        return None
    return "\n".join(
        [
            *section_head(hop),
            *[f"{BODY}{x}" for x in _index(ids, store)],
            f"{BODY}{tail}",
        ]
    )


def _block_task(task: Task) -> str:
    """① 任务：这次要什么。

    来源标注必须标：执行器得分得清哪句是用户说的、哪句是模型说的，否则模型补的东西会被
    当成用户的事实。
    """
    return "\n".join([*section_head("task"), *body_lines(task.get("task", ""), BODY)])


def _block_user_turns(task: Task) -> str | None:
    """② 用户原话（平台硬搬，未经转述）—— 清单常驻、全文按需。"""
    turns = task.get("user_turns") or []
    if not turns:
        return None
    # 索引的形状在 `turn_lines`；这里只负责加段名、来源标注和块尾那句逃生口。
    block = section_head("user_turns") + turn_lines(turns)
    block.append(
        f"{BODY}（**这是事实来源**；【任务】是这次的要求，两者冲突时**以原话为准**。"
        "要看哪一轮的**完整上下文**（它回答的上一句 ＋ 原话）就用 `read_user_turn(轮次)` 读，"
        "不要凭那几十个字猜。）"
    )
    # 这里不报 `trace.cut`：索引行不是"被切掉的正文"，是这一层本来就只放一行。
    return "\n".join(block)


def _block_answered(task: Task) -> str | None:
    """③ 问答经过，一条时间序（历次问→答，末尾接上它最后说的那句）。

    要成对：`report` 是覆盖写的，只存答案的话第二次重入之后旧答案就成了没有指涉的孤儿。
    """
    qa = [
        x for x in (task.get("qa") or []) if isinstance(x, dict) and str(x.get("a") or "").strip()
    ]
    report = (task.get("report") or "").strip()
    if not (qa or report):
        return None
    # 确定性去重：逐字比较两个我们自己写的字段，不从文本里反结构化 ——
    # 成立与否是"恒成立"而不是"碰巧"，所以能断言。
    folded = bool(qa) and bool(report) and str(qa[-1].get("q") or "").strip() == report
    block = section_head("answered")
    for i, x in enumerate(qa, 1):
        q = str(x.get("q") or "").strip()
        by = str(x.get("by") or "用户")
        if q:
            # `q` 是 `ask_user` 的参数原话（结构），所以这里没有"抽取哪句才是问题"
            # 那种语义判断要做。
            block.append(f"{BODY}- **问 {i}**（执行器原话）：")
            block += body_lines(q, BODY * 2)
        else:
            # 空 = 这一条没记下 `ask_user` 的参数（fixture / 手写 state），不等于"它没
            # 说话" —— 它说的那段话在下面「它最后说的那句」里。
            block.append(f"{BODY}- **问 {i}**：（**这一条没留下问题原话**）")
        # 「答」是和「问」平级的一项，不是它的续行 —— 同缩进会让答看起来像问的一部分。
        block.append(f"{BODY}- **答 {i}**（**{by}**）：")
        block += body_lines(str(x["a"]), BODY * 2)
    if folded:
        block.append(
            f"{BODY}（**它最后说的那句**就是上面「问 {len(qa)}」，逐字相同 —— "
            "原样，**不重复贴**。）"
        )
    elif report:
        # `report` 不是任何一问。两条路都会走到这里：首跑 / 试答后重跑时它那段话是唯一
        # 的进展；有 `qa` 的轮里「问」取的是 `asked`，与 `report` 本就不是同一句，那段话
        # 是它自己的收尾，一个字都不能少。
        tail = "" if qa else "，这一轮还没有人答"
        block.append(f"{BODY}- **它最后说的那句**（原样{tail}）：")
        block += body_lines(report, BODY * 2)
    if len(qa) > 1:
        block.append(f"{BODY}（如前后答复冲突，**以后面的为准**）")
    if folded:
        # 折叠也要报一刀：标记在文本里（读的人看得见），这一行在日志里（复盘的人看得见）。
        trace.note(
            f"🔁 折叠重复 · 交接书：它最后说的那句 == 上面「问 {len(qa)}」"
            f"（{len(report)} 字，原样不重复贴）"
        )
    return "\n".join(block)


def _block_working(task: Task) -> str | None:
    """中间数据段 —— 这套工作留下的会话内暂存：接着干时可能用得上，但它不是交付物。"""
    lines = working_data_lines(task.get("working_data") or [])
    if not lines:
        return None
    return "\n".join([*section_head("working"), *(f"{BODY}{line}" for line in lines)])


def _block_resume(task: Task) -> str | None:
    """⑤ 续跑提示 —— 只有重入才说：首跑说"接着干"会让它以为有前情。"""
    if not (task.get("report") or task.get("qa")):
        return None
    return "\n".join(
        [
            *section_head("resume"),
            f"{BODY}接着已有的工作继续，**不要重复已完成的工作**。",
        ]
    )


def render_briefing(task: Task, store: ArtifactStore = DEFAULT_STORE) -> str:
    """把 Task 渲染成执行器的输入，所有重入路径的唯一入口。

    它是整份交接而不是增量。渲染出来的就是全部：没有"给人看的另一份"，日志里 INJECT
    那一行 dump 的就是这个函数的返回值本身。
    """
    blocks = [
        _block_task(task),
        _block_user_turns(task),
        _block_answered(task),
        _index_block(
            "already_produced",
            task.get("artifacts") or [],
            "（这次要**在它基础上补全**，提交完整版；不要从零重写。）",
            store,
        ),
        _block_working(task),
        _index_block(
            "readable",
            task.get("refs") or [],
            "（这是清单，不是正文。要用哪一份就用 read_artifact 读全文，不要凭标题猜。）",
            store,
        ),
        _block_resume(task),
    ]
    return "\n\n".join(b for b in blocks if b)


def absorb_outcome(rec: Task, outcome: Outcome) -> None:
    """把执行器的结果写进 Task，四个字段一个都不能少。

    `artifacts` **累加**：一个任务可以交很多份产物（技能每阶段一份很常见，不区分中间与
    最终），这一趟交的不许把上一趟的顶掉 —— 顶掉了，下一趟的交接书里就"忘"了前面的。
    累加按 id **去重保序**：本函数会被同一个 outcome 调两次（`dispatch` 先 absorb、
    `settle` 里再 absorb 一次），不去重的话同一批产物会被记两遍，台账与交接书的
    「已交付」清单跟着重复。其余字段照旧覆盖写。
    """
    rec["report"] = outcome.report
    # 它申报要的那条信息 —— 与 `report` 一样覆盖写：这一轮没申报就清空（它已经答上了 /
    # 已经交付了，"待答"这个状态就结束了）。
    rec["asked"] = outcome.asked
    # 备选跟着一起覆盖写：这一轮没申报就没备选，免得上一轮的选项阴魂不散。
    rec["asked_options"] = list(outcome.asked_options)
    # 平台故障也落进 Task，同样覆盖写：正常跑完就清空，不然一个瞬时的故障会在台账里
    # 阴魂不散。
    rec["fault"] = outcome.fault
    # 故障的"类"跟着一起落（台账据此选引导句），同样覆盖写。
    rec["fault_kind"] = outcome.fault_kind
    if outcome.artifacts:
        rec["artifacts"] = list(dict.fromkeys([*(rec.get("artifacts") or []), *outcome.artifacts]))
    # 中间数据同样累加去重保序 —— 它是另一本账：它不是交付物，不进出口判定。
    if outcome.working_data:
        rec["working_data"] = list(
            dict.fromkeys([*(rec.get("working_data") or []), *outcome.working_data])
        )


#: 执行器三态 → `Task.stage`，只有这一处映射。三态是封闭枚举，所以这里穷举。
_STATUS_TO_STAGE: dict[OutcomeStatus, TaskStage] = {
    "completed": "completed",
    "waiting_for_input": "waiting_for_input",
    "incomplete": "failed",
}


def settle(rec: Task, outcome: Outcome) -> str:
    """把一次执行结果落进 Task，状态迁移的唯一入口。

    先 `absorb_outcome` 再写 `stage`，顺序不能反。调用方常常已经先 absorb 过一次，这里
    再写一遍是幂等的（产物按 id 去重保序），换来的是"不可能漏写"。
    """
    absorb_outcome(rec, outcome)
    rec["stage"] = _STATUS_TO_STAGE[outcome.status]
    # 落账也记一行：三态是"这一趟跑成什么样"，`Task.stage` 才是"这个任务现在算什么"，
    # 后者不写就只在 state 里，trace 里接不上。不改取值（映射仍只有一处），只把它记出来；
    # 无 kind = 纯记录、不上屏（终端那行快照已经在说同一件事）。
    trace.note(
        f"Task 落账 · {rec.get('id')}「{rec.get('skill')}」："
        f"status={outcome.status} → stage={rec['stage']}"
        + ("｜**平台故障**：这一轮没跑成" if outcome.fault else "")
    )
    return rec["stage"]


def moot_older_waiting(done: Task, tasks: list[Task]) -> list[Task]:
    """刚完成的这条让哪些更早的待答成了哑问题 → 显式标 `superseded`，不静默丢。

    判据两条，都是结构事实：新任务已完成；用户在这条之后又说过话（`user_turns` 快照前缀
    延长）。快照没延长（同一轮重派）或没有快照（fixture）一律不动——那正是"主 agent 自己
    编了答案重派、用户从没被问过"，问题必须留着让它被看见。

    和 `_supersede_old` 分工：那条管重派取代 failed，这条管完成了且用户已另起话题。返回改
    过 stage 的副本，原 dict 不动；取代要留痕（`trace.note`，kind=`supersede`）。
    """
    done_turns = done.get("user_turns") or []
    out: list[Task] = []
    for t in tasks:
        if t.get("id") == done.get("id") or t.get("skill") != done.get("skill"):
            continue
        if t.get("stage") != "waiting_for_input":
            continue
        t_turns = t.get("user_turns") or []
        # 判据②：快照必须前缀延长（用户在这条之后又说过话）。
        if not t_turns or len(t_turns) >= len(done_turns) or done_turns[: len(t_turns)] != t_turns:
            continue
        out.append({**t, "stage": "superseded"})
        # 取代要留痕（"显式"那一半）：说清谁被谁取代、依据是什么 ——
        # 判据可审计（依据"没有"也要显式写）。
        trace.note(
            f"🔁 {t['id']}（{t['skill']}）的待答**已被新任务 {done['id']} 取代** —— "
            f"依据：用户在这条之后又说过话（原话快照 {len(t_turns)} → {len(done_turns)} 轮），"
            f"且新任务已完成；旧问题不再摆出（**显式取代，不是静默丢**）",
            kind="supersede",
        )
    return out
