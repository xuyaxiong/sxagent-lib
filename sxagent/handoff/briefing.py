"""**委托单**（平台 → 执行器）：Task 的结构 ＋ 渲染模板。

    sxagent/handoff/
      __init__.py    ← 共享词汇表（段名 / 来源标注 / 排版，所有交接点共用那一份）
      briefing.py    ← 本文件：委托单（平台 → 执行器）
      receipt.py     ← 回执（执行器 → 主 agent）
      ask.py · nudge.py · ledger.py · decision.py  ← 另外四跳（schema ＋ 模板同址）

## 这个文件是什么

平台**每一次**把工作交给执行器时，交给它的**那份完整交接**（不是增量）：Task / Turn /
QA 三个 schema ＋ 渲染入口 `render_briefing`，外加执行结果落回 Task 的
`absorb_outcome` / `settle` 与"哑待答"的显式取代 `moot_older_waiting`。
执行器**用完即弃**（每次投递都是一次全新的 `run_task`）—— 所以"这件事到目前为止的
全部已知"必须由平台**重新交给它**。

## 设计原则（不变量）

1. **段序 ＝ 读的人理解这件事的顺序**：

       我要做什么 → 用户原话 → 我做过什么 / 问过什么 → 我产出了什么 → 接着干什么

   ⚠️ **一条时间序**：不按来源分段、不"最新在前" —— 后者要让读的人在两段之间来回跳，
   才能把一件事的经过拼起来。
2. **来源标注是硬要求**：执行器必须能分清"**哪句是用户说的、哪句是模型说的**"。
   平台**不阻止**主 agent 补充信息（那是它的活），但**必须让下游知道这是谁的话** ——
   否则模型补的东西会被当成用户的事实。
3. **Task 的每个字段要么渲染在这里、要么在排除名单里** —— 加字段忘了渲染要被检查出来。
4. **两处硬机制，都是确定性的（不靠措辞）**：
   ① **排版** —— 段名 0 列 / 段体 2 列 / 每层 +2（`sxagent.handoff.BODY` ＋ `body_lines`）；
   ② **去重** —— 逐字比较两个**我们自己写的字段**（`qa[-1].q` vs `report`），
      不是从文本里反结构化（成立与否是"恒成立"而不是"碰巧"，所以能断言）。

## 关键取舍

**① 一问一答成对**：`report` / `asked` 都是**覆盖写**，只存答案的话，第二次重入之后
旧答案就成了**没有指涉的孤儿** —— 所以「问」由平台在**收下答案的那一刻**从 `Task.asked`
（`ask_user` 的参数原话）原样截下，不靠任何 LLM 转述。

**② 只折叠逐字相同的最后一条，且必须标出来**：读的人分不出"它没说"和"被折叠了" ——
悄悄不贴和悄悄切一刀一样糟；折叠的**字数**另报一行日志（同 `trace.cut` 的做法）。

**③ 缺的东西显式写出来，不许静默跳过**：有 ID 而取不到是异常（写"产物库里没有这一条"），
`asked` 为空也一样明说 —— **有就说有，没有就说没有**。

**④ 说不清就不动**：哑待答的取代只在**两条结构事实**都成立时才做；fixture / 手写 state
一律不碰 —— fail-safe。

**⑤ 措辞写死"以新的为准"**：读交接书的是**执行器**（每次重入、用完即弃）——
【问答经过】按时间序排，块尾那句「如前后答复冲突，**以后面的为准**」（`render_briefing`）
就是防它把旧答复当成当前的。

## 遗留问题

- `Task` **没有运行时校验**（`total=False` 是如实）：靠 `delegate` 建全 ＋ 断言兜底 ——
  见 `Task` 的 docstring。
"""

from __future__ import annotations

from typing import Any, TypedDict

# ⚠️ 顶层 import（无环，纯习惯）—— `artifacts` 只依赖 langchain 的工具装饰器，
# `trace` 只依赖 `config`。
from sxagent import artifacts as artifacts_mod
from sxagent import trace
from sxagent.handoff import (
    BODY,
    FaultKind,
    OutcomeStatus,
    TaskStage,
    body_lines,
    section_head,
)

# ──────────────────────────── schema：委托单上有什么 ────────────────────────────


class Turn(TypedDict):
    """**一轮用户原话**（平台硬搬的快照，不经任何转述）。

    🔴 `after` 是**紧邻的前一条 AI 消息**（纯位置关系，平台搬，不判断语义）——
    用户答「A」时，脱离它回答的那句（选项列表在里面）就没意义。
    """

    n: int
    text: str
    after: str


class QA(TypedDict):
    """**一问一答成对** —— 免得「答」脱离了「问」。

    `report` / `asked` 都是**覆盖**写的（`absorb_outcome`），所以只存答案的话，
    第二次重入之后旧答案就成了没有指涉的孤儿。问题由**平台**在收下答案的那一刻
    从 `Task.asked`（`ask_user` 的参数原话）原样截下，不靠任何 LLM 转述。

    🔴 **`q` 是 `asked`，不是 `report`**：`report` 是执行器最后说的**那段话**，
    混着过场 —— 拿它当「问」等于没有指涉。⚠️ `asked` 取不到（fixture / 手写 state）时
    存**空串**，渲染那边会明说"这一条没留下问题原话"；**不许**拿 `report` 顶替。
    """

    by: str  # "用户" | "主 agent"
    q: str  # 执行器问的原话（`ask_user` 的参数 —— 取不到时是空串，不是 `report`）
    a: str  # 答复


class Task(TypedDict, total=False):
    """**委托单**：一次委派出去的工作的全部已知。

    ⚠️ `total=False` 是**如实**：注入的 Task 可能只有部分字段。
    **没有运行时校验** —— 靠 `delegate` 建全 + 断言兜底。
    """

    id: str
    skill: str
    task: str  # 主 agent 写的"这次要什么"（**它说的，可能加了料**）
    stage: TaskStage  # 五态，词汇表在 `handoff/__init__`
    artifacts: list[str]  # **产出**的产物 ID
    refs: list[str]  # **输入引用**的产物 ID（两个方向，别混）
    report: str  # 执行器最后一轮的原话（**覆盖写**）
    #: 🔴 **它申报要的那条信息** —— `ask_user(question)` 参数里的**原话**（**覆盖写**）。
    #: 结构性事实必须落进 Task，下游才有对象可用（不然只能用散文）—— 与 `fault` 同一个病。
    asked: str
    #: 🔴 **平台故障的原因**（非空 = 这一轮**根本没跑成**，**不是执行器干不了**；空 = 没故障）。
    #: 事实只有这一处定义：回执的【平台故障】与台账的标记都是它的视图。
    fault: str
    #: 🔴 **平台故障的"类"**（`FaultKind`：`transient` / `rejected` / `timeout` / `""`）—— 台账据此选
    #: 引导句（"原样再委派" vs "换措辞/改内容"）。**结构，不许从 `fault` 文本反解析**。
    #: ⚠️ 与 `fault` 同生共死：每次 `settle` 由 `absorb_outcome` **覆盖写**（正常跑完清空）。
    fault_kind: FaultKind
    qa: list[QA]
    user_turns: list[Turn]  # 派发那一刻的快照
    agent_tried: bool  # 平台强制试答过没有
    #: 这个 Task **最近一次运行是谁起的** —— `"delegate"`（主 agent 派发的）
    #: / `"feed"`（用户答复后平台投递重跑的）。
    #:
    #: 🔴 台账截断时要说清"全文在哪"，而两条路的答案**不一样**（投递那条路
    #: **没有回执**）—— 指错会让主 agent 翻到**上一轮**的回执、把旧的当成当前的。
    run_by: str  # "delegate" | "feed"


# ──────────────────────────── 渲染：注入前那一瞬才拼 ────────────────────────────


#: **索引行**里每轮露出的开头字数（`_one_line` 的默认 limit）。
#:
#: ⚠️ 它不是"切到多少"，是"**一行有多长**" —— 见 `turn_lines` 的分层说明。
_one_line_limit = 40


def asked_text(t: Task) -> str:
    """待答任务 → **用户该答的那一句**（`Task.asked`）。取不到就**显式说清**。

    🔴 **只负责"取哪句"，不负责排版** —— 两个视图**载体不同**（一个是对话里的
    一条消息、一个是输入通道的标签），**该各自排版**（"视图可以有很多个，
    事实源只能有一个"）。所以这里返回的是**内容**，不带 `- 「skill」问：` 那种框架。

    ⚠️ 返回**单行**（`_flat` 压平）：它要能直接用在 `input()` 的提示符里。
    压平只动空白、不动内容（同 `_flat` 的注释）。

    ⚠️ `asked` 为空**只有两个来源**（产品路径上不该发生）：注入的 fixture、手写的 state。

    🔴 这条规则（"取哪句、取不到怎么办"）**只许有一处实现** —— 曾经实现过三遍，
    最坏的一遍**静默退回 `report`**：把"数据缺失"藏成了"看起来正常"。
    """
    q = _flat(t.get("asked"))
    if q:
        return q
    # ⚠️ **不许静默退回 `report`**：说清"没有"，再把它最后说的那段附上 ——
    #    有就说有，没有就说没有。
    return (f"（**这一条没留下问题原话**；它最后说的是："
            f"{_flat(t.get('report')) or '（它一个字都没说）'}）")


def _flat(text: object) -> str:
    """`_one_line` **压成一行**之后的样子 —— 压平与比长度必须用**同一个口径**。

    否则换行多的短消息会被误判成"超长"（`_one_line` 先压平再判长度）。
    """
    return " ".join(str(text or "").split())


def _one_line(text: str, limit: int = _one_line_limit) -> str:
    """压成一行 + 截到 limit 字 —— 索引里**一行一轮**，不能让它撑成多行。

    🔴 **这不是"截断"，是"索引"**（见 `turn_lines`）：短到一行装得下的，
    这一行**就是全文**；装不下的，全文走 `read_user_turn(轮次)` 按需取。
    所以这里既不报 `trace.cut`，也不需要"切了"的标记 —— `（X 字）` 和 `…`
    已经把"这轮多大、你只看到开头"说清楚了。
    """
    flat = _flat(text)
    return flat if len(flat) <= limit else flat[:limit] + "…"


def turn_lines(turns: list[dict]) -> list[str]:
    """**用户原话索引** —— 一行一轮：序号 · 字数 · 开头 · 它在回答什么。

    ## 分层（渐进式披露）

    | 层 | 内容 | 成本 |
    |---|---|---|
    | **常驻（本函数）** | 每轮**一行** | **∝ 轮次数**，与文本量**无关** |
    | **按需** | `read_user_turn(n)` 取那一轮全文 ＋ 它回答的那句全文 | 读的人自己触发 |

    🔴 **关键性质：短到一行装得下的轮次，这一行就是全文**（实测 226 份会话里
    **91% 的用户消息 ≤40 字**、中位 5 字）—— 所以"索引"和"全文"在绝大多数情况下
    是同一个东西；真正的长正文一律走按需。**用户粘 5 万字，这一层也只涨一行。**

    🔴 **平台不"点名"**：不猜"哪几轮相关"，只保证 ① 原话**一字不改**地搬进索引、
    ② **每一轮都在索引里**；哪几轮要读全文，由**读的人**按需取。
    """
    out: list[str] = []
    for t in turns:
        n, text = t.get("n"), str(t.get("text") or "")
        out.append(f"{BODY}- 第 {n} 轮（{len(text)} 字）：{_one_line(text)}")
        after = str(t.get("after") or "").strip()
        if after:
            # 🔴 **短应答必须带上它在回答什么**：光一个「A」，执行器既解读不了、
            # 也核对不了主 agent 的转述 —— 而这一段存在的全部理由就是**对冲转述失真**。
            out.append(f"{BODY}  ↳ **它在回答的上一句**：「{_one_line(after)}」")
    return out


def _index(aids: list[str]) -> list[str]:
    """产物 ID 列表 → 清单行（**含缺失**）。

    🔴 **缺失要显式写出来，不许静默跳过**：有 ID 而取不到是**异常**，必须说 ——
    同一份坏数据在回执那边写的是「**产物库里没有这一条**」，两处不许两副面孔。
    """
    out: list[str] = []
    for aid in aids:
        a = artifacts_mod.get(aid)
        if not a:
            out.append(f"- {aid}（**产物库里没有这一条**）")
            continue
        out.append(
            f"- {aid}《{a['title']}》（{len(a['content'])} 字）"
            f"—— {a.get('summary') or '（无梗概）'}"
        )
    return out


def _index_block(hop: str, aids: list[str], tail: str) -> str | None:
    """一个**产物清单段**（已产出 / 可读）—— 清单空就不渲染这一块。

    ⚠️ 两段的形状**一模一样**（段名 ＋ 清单行 ＋ 尾句），所以只有这一处拼 ——
    原来是两段各写一遍（`already_produced` / `readable`）。
    """
    ids = [a for a in aids if a]
    if not ids:
        return None
    return "\n".join([
        *section_head(hop),
        *[f"{BODY}{x}" for x in _index(ids)],
        f"{BODY}{tail}",
    ])


def _block_task(task: Task) -> str:
    """① **任务**：这次要什么。

    🔴 **来源标注**：执行器必须能分清"**哪句是用户说的、哪句是模型说的**"。
    平台**不阻止**主 agent 补充信息（那是它的活），但**必须让下游知道这是谁的话** ——
    否则模型补的东西会被当成用户的事实。
    """
    return "\n".join([*section_head("task"), *body_lines(task.get("task", ""), BODY)])


def _block_user_turns(task: Task) -> str | None:
    """② **用户原话**（平台硬搬，未经转述）—— 渐进式披露：清单常驻、全文按需。"""
    turns = task.get("user_turns") or []
    if not turns:
        return None
    # 索引的**形状**在 `turn_lines`（分层 ＋ 平台不"点名"那条规矩都在那里）；
    # 这里只负责加段名、来源标注和块尾那句逃生口。
    block = section_head("user_turns") + turn_lines(turns)
    block.append(
        f"{BODY}（**这是事实来源**；【任务】是这次的要求，两者冲突时**以原话为准**。"
        "要看哪一轮的**完整上下文**（它回答的上一句 ＋ 原话）就用 `read_user_turn(轮次)` 读，"
        "不要凭那几十个字猜。）"
    )
    # ⚠️ 这里**不报 `trace.cut`**：索引行**不是"被切掉的正文"**，是这一层**本来
    #    就只放一行** —— "这轮多大、你只看到开头"由 `（X 字）` ＋ `…` ＋
    #    块尾那句逃生口说清楚。
    return "\n".join(block)


def _block_answered(task: Task) -> str | None:
    """③ **问答经过** —— **一条时间序**（历次问→答，末尾接上"它最后说的那句"）。

    🔴 **一问一答成对**：`report` 是**覆盖**写的，所以只存答案的话，第二次重入之后
    旧答案就成了**没有指涉的孤儿**。问题由**平台**在收下答案的那一刻从 `Task.asked`
    （`ask_user` 的参数原话）原样截下，不靠任何 LLM 转述。
    """
    qa = [x for x in (task.get("qa") or [])
          if isinstance(x, dict) and str(x.get("a") or "").strip()]
    report = (task.get("report") or "").strip()
    if not (qa or report):
        return None
    # 🔴 **确定性去重**（判据见模块头「设计原则 4②」）—— 逐字比较两个**我们自己写的字段**，
    # 不是从文本里反结构化。成立与否是"恒成立"而不是"碰巧"，所以能断言。
    folded = bool(qa) and bool(report) and str(qa[-1].get("q") or "").strip() == report
    block = section_head("answered")
    for i, x in enumerate(qa, 1):
        q = str(x.get("q") or "").strip()
        by = str(x.get("by") or "用户")
        if q:
            # `q` 是 `ask_user` 的**参数原话**（结构 ✓，不是从散文里抽的）——
            # 所以这里**没有**"抽取哪句才是问题"那种语义判断要做。
            block.append(f"{BODY}- **问 {i}**（执行器原话）：")
            block += body_lines(q, BODY * 2)
        else:
            # ⚠️ 空 = **这一条没记下 `ask_user` 的参数**（fixture / 手写 state），
            #    不等于"它没说话" —— 它说的那段话在下面「它最后说的那句」里。
            block.append(f"{BODY}- **问 {i}**：（**这一条没留下问题原话**）")
        # ⚠️ 「答」是**和「问」平级的一项**，不是「问」的续行 ——
        # 原来是 `  **用户** 答：`（和问的续行同一个缩进），答看起来像问的一部分。
        block.append(f"{BODY}- **答 {i}**（**{by}**）：")
        block += body_lines(str(x["a"]), BODY * 2)
    if folded:
        block.append(
            f"{BODY}（**它最后说的那句**就是上面「问 {len(qa)}」，逐字相同 —— "
            "原样，**不重复贴**。）"
        )
    elif report:
        # `report` 不是任何一问。**两条路都会走到这里**：
        #   - 首跑 / 试答后重跑：还没有人答过，它那段话是**唯一的进展**；
        #   - 有 `qa` 的那些轮：「问」取的是 `asked`，与 `report` 本就不是同一句 ——
        #     那段话是它自己的收尾，不该丢。**一个字都不能少**，所以照原样给它一块。
        tail = "" if qa else "，这一轮还没有人答"
        block.append(f"{BODY}- **它最后说的那句**（原样{tail}）：")
        block += body_lines(report, BODY * 2)
    if len(qa) > 1:
        block.append(f"{BODY}（如前后答复冲突，**以后面的为准**）")
    if folded:
        # 折叠**也要报一刀**（同 `trace.cut`）：标记在**文本里**（读的人看得见），
        # 这一行在**日志里**（复盘的人看得见）—— 两者都要有。
        trace.note(
            f"🔁 折叠重复 · 交接书：它最后说的那句 == 上面「问 {len(qa)}」"
            f"（{len(report)} 字，原样不重复贴）"
        )
    return "\n".join(block)


def _block_resume(task: Task) -> str | None:
    """⑤ **续跑提示** —— **只有重入才说**：首跑说"接着干"会让它以为有前情。"""
    if not (task.get("report") or task.get("qa")):
        return None
    return "\n".join([
        *section_head("resume"),
        f"{BODY}接着已有的工作继续，**不要重复已完成的工作**。",
    ])


def render_briefing(task: Task) -> str:
    """把 Task 渲染成**执行器的输入** —— 所有重入路径的唯一入口。

    ⚠️ 名字是 task 不是 given：它是**整份交接**（任务 / 上次做到哪 / 历次答复 /
    已产出产物 / 可读产物），不是"这一轮的增量"—— 执行器用完即弃，每次都是重入。

    ⚠️ **渲染出来的就是全部**：没有"给人看的另一份"（用户："你显示给我看的，
    就是传给 llm 看的"）—— 日志里 `📥 INJECT` dump 的是这个函数的返回值本身。

    ⚠️ **它自己只做拼装**：六块各在一个私有件里，空块不渲染。
    """
    blocks = [
        _block_task(task),
        _block_user_turns(task),
        _block_answered(task),
        _index_block("already_produced", task.get("artifacts") or [],
                     "（这次要**在它基础上补全**，提交完整版；不要从零重写。）"),
        _index_block("readable", task.get("refs") or [],
                     "（这是清单，不是正文。要用哪一份就用 read_artifact 读全文，不要凭标题猜。）"),
        _block_resume(task),
    ]
    return "\n\n".join(b for b in blocks if b)


def absorb_outcome(rec: Task, outcome: Any) -> None:
    """把执行器的结果**写进 Task** —— 四个字段一个都不能少（存了才可能喂）。

    ⚠️ `artifacts` 只在非空时覆盖：执行器可能这一轮没产出，不能把上一轮的抹掉。
    """
    rec["report"] = outcome.report
    # 它申报要的那条信息（`ask_user` 的原话）—— 与 `report` 一样**覆盖写**：
    # 这一轮没申报就清空（它已经答上了 / 已经交付了，"待答"这个状态就结束了）。
    rec["asked"] = outcome.asked
    # 平台故障**也落进 Task**（理由见 `Task.fault`）—— 与 `report` 一样**覆盖写**：
    # 正常跑完就清空，不留上一次的故障（不然一个瞬时的故障会在台账里阴魂不散）。
    rec["fault"] = outcome.fault
    # 故障的"类"跟着一起落（台账据此选引导句，见 `Task.fault_kind`）—— 同样覆盖写。
    rec["fault_kind"] = outcome.fault_kind
    if outcome.artifacts:
        rec["artifacts"] = list(outcome.artifacts)


#: 执行器三态 → `Task.stage` —— **只有这一处**映射（`ExecOutcome.status` 是封闭枚举，
#: 所以这里是穷举的：`completed` / `waiting_for_input` / **其余一律 failed**）。
#: 键 = `OutcomeStatus`、值 = `TaskStage` —— **两边都是词汇表里的类型**，
#: 打错一个键名 mypy 会报；"键是不是刚好覆盖三态"要有检查盯着。
_STATUS_TO_STAGE: dict[OutcomeStatus, TaskStage] = {
    "completed": "completed",
    "waiting_for_input": "waiting_for_input",
    "incomplete": "failed",
}


def settle(rec: Task, outcome: Any) -> str:
    """**把一次执行结果落进 Task** —— 状态迁移的唯一入口。

    两件事一起做完，顺序不能反：
      ① `absorb_outcome`：report / artifacts
         （**artifacts 只在非空时覆盖** —— 这一轮没产出，不能把上一轮交付的抹掉）；
      ② `stage`：由 `outcome.status` 映射（见 `_STATUS_TO_STAGE`）。

    ⚠️ 调用方常常已经先 `absorb_outcome` 过一次（重入前要 `render_briefing`）——
    这里再写一遍是**幂等**的，换来的是"不可能漏写"。
    """
    absorb_outcome(rec, outcome)
    rec["stage"] = _STATUS_TO_STAGE[outcome.status]
    # 🔴 **落账也记一行**：三态是"**这一趟**跑成什么样"，`Task.stage` 才是"**这个任务**
    #    现在算什么" —— 后者原来只活在 state 里，trace 里没有"tac2700f3 → completed"
    #    这一行，复盘时接不上（用户要审计的"完成没"里，这就是**任务那一侧的结果**）。
    #    ⚠️ **不改取值**（`_STATUS_TO_STAGE` 仍是唯一映射）—— 这里只是把它**记出来**。
    #    ⚠️ **无 kind = 纯记录、不上屏**：终端那行快照已经在说同一件事，
    #       再上屏就是同一个事实两副面孔。
    trace.note(
        f"Task 落账 · {rec.get('id')}「{rec.get('skill')}」："
        f"status={outcome.status} → stage={rec['stage']}"
        + ("｜**平台故障**：这一轮没跑成" if outcome.fault else "")
    )
    return rec["stage"]


def moot_older_waiting(done: Task, tasks: list[Task]) -> list[Task]:
    """**刚完成的这条**让哪些更早的待答成了哑问题 → 显式标 `superseded`（**不是静默丢**）。

    ## 判据（两条，都是**结构事实**，不是猜）

      ① **新任务已完成**（`done` 刚 settle 成 completed）—— 活已经干完了；
      ② **用户在这条之后又说过话** —— `user_turns` 快照**前缀延长**
         （`len(t) < len(done)` 且 `done` 的快照以 `t` 的为前缀）。

    快照**没延长**（同一轮里重派）或**没有快照**（fixture / 手写 state）→ **一律不动**：
    那正是"主 agent 自己编了答案重派、用户从没被问过"的场景 —— 问题必须留着，
    等用户真的看见它。**说不清就不动**（同 `ledger._SRC_DEFAULT` 的 fail-safe）。

    ⚠️ **和 `_supersede_old` 的分工**：那条管"重派取代 failed"（在**建**的时候）；
    这条管"完成了、且用户已另起话题"（在**落账**的时候）—— 两条都不碰
    "用户还没机会说话"的待答。

    ⚠️ 返回的是**改过 stage 的副本**（原 dict 不动，同 `_supersede_old`）——
    交给 reducer 按 id 合并。取代**留痕**（`trace.note`，kind=`supersede`）。

    为什么要它：用户没走 `↩` 答复通道、又发了新消息时，旧待答**永远挂着**
    （每轮重摆、答它＝重跑旧任务）。
    """
    done_turns = done.get("user_turns") or []
    out: list[dict] = []
    for t in tasks:
        if t.get("id") == done.get("id") or t.get("skill") != done.get("skill"):
            continue
        if t.get("stage") != "waiting_for_input":
            continue
        t_turns = t.get("user_turns") or []
        # 判据②：快照必须**前缀延长**（用户在这条之后又说过话）。
        if (not t_turns or len(t_turns) >= len(done_turns)
                or done_turns[: len(t_turns)] != t_turns):
            continue
        out.append({**t, "stage": "superseded"})
        # 🔴 **取代要留痕**（"显式"那一半）：说清谁被谁取代、依据是什么 ——
        #    判据可审计（"判断必须可审计"：依据"没有"也要显式写）。
        trace.note(
            f"🔁 {t['id']}（{t['skill']}）的待答**已被新任务 {done['id']} 取代** —— "
            f"依据：用户在这条之后又说过话（原话快照 {len(t_turns)} → {len(done_turns)} 轮），"
            f"且新任务已完成；旧问题不再摆出（**显式取代，不是静默丢**）",
            kind="supersede",
        )
    return out
