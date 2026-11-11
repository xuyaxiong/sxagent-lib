"""交接的共享词汇表：段名、来源标注、状态词、出口分类、排版、跳声明。

    __init__.py   ← 本文件：词汇表，所有交接点共用这一份
    briefing.py   ← 委托单（平台 → 执行器）
    receipt.py    ← 回执（执行器 → 主 agent）
    ask.py · nudge.py · ledger.py · decision.py  ← 另外四跳，schema 与模板同址

交接单上每一段叫什么、来源怎么标、状态怎么命名、出口怎么分类、每一跳渲染哪几段，全库只在
这里定义一次。渲染器和记录层都从这里取，所以两边不会各抄一份再悄悄失配。

## 不变量

1. 一个语义只定义一处。加一个段就登记进 `SECTION` / `HOPS`，别处不许就地写字符串。
2. 不反解析。结构一旦渲染成文本就回不去。正解是先有对象，注入那一刻才渲染，对象搭同一趟车
   （`additional_kwargs`）；没有对象就如实显示成一段话。
3. 来源标注属于结构，不是措辞装饰。记录层切段时靠 `SOURCE_NOTE` 原样剥掉，不用"行首是
   `（…）` 就剥"这种启发式，它会误剥 `（无）` 这种真内容。
4. 排版一条规矩：段名 0 列、来源 2 列、段体 2 列，每深一层加 2（实现在 `BODY`）。
5. 一个交接点一个模块，schema 和模板同址。
6. 一处内部一致：段体可称「你」，来源标注一律第三人称（它是元信息，不随读的人变）。同一个
   语义一个词：委派（不写派发）、产物是名词而产出只做动词、梗概（不写摘要）。"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

# ──────────────────────────── 段名 ────────────────────────────

#: 段名。渲染和切分都从这里取，一律 `【…】` 形式、独占行首。
SECTION: dict[str, str] = {
    # ── 委托单（平台 → 执行器，`briefing.py`）──
    "task": "【任务】",
    "user_turns": "【用户原话】",
    "answered": "【问答经过】",
    "already_produced": "【已经产出的产物】",
    "readable": "【可读产物】",
    "resume": "【续跑提示】",
    # ── 回执（执行器 → 主 agent，`receipt.py`）──
    "deliver": "【交付】",
    #: 它申报要的那句原话。塞进【交付】就混进平台的框架话里了。
    "asked": "【它要的信息】",
    "produced": "【产出的产物】",
    "report": "【执行器说】",
    "agent_answer": "【你补的信息】",
    "next": "【下一步】",
    #: 平台故障：这一轮根本没跑成，与"执行器没干成"是两回事。
    "fault": "【平台故障】",
    # ── 派发被拒（平台 → 主 agent，不建 Task）──
    "not_dispatched": "【这次没有委派】",
    "available_skills": "【可用的技能】",
    "available_artifacts": "【本会话现有产物】",
    # ── 台账（平台 → 主 agent，`ledger.py`）──
    "ledger": "【平台提示】",
    # ── 试答（平台 → 主 agent，`ask.py`）──
    "ask_context": "【你的上下文】",
    "ask_decide": "【请判断】",
    # ── 推回去（平台 → 执行器，`nudge.py`）──
    "nudge": "【平台要求】",
    # ── 待答透传（平台 → 用户，`graph.render_relay`）──
    #: 技能名包在 `「」` 里，别借段名的方括号。
    "relay": "【待答】",
    # ── 任务文本（平台 → 执行器，`executor.run_task`）──
    #: `known` 的段体是整份交接书，嵌套靠 `body_lines` 缩出来。
    "skill_instructions": "【技能指令】",
    "known": "【已知信息】",
    # ── 决策（LLM 调用 → 记录，`decision.py`）──
    #: 四格混着两方的话：前提 / 问题 / 下一步是平台的话；结果 = 它输出的，要标来源。
    "dec_given": "【前提】",
    "dec_q": "【问题】",
    "dec_result": "【结果】",
    # ── 判定（平台按结构定 → 记录，`decision.py`）──
    #: 【结果】有两种作者：模型输出的（标来源），和平台按结构定的结论（不标）。
    "verdict_result": "【平台判定】",
}

#: 来源标注 —— "这句是谁说的"。记录层切段时原样剥掉它。
#: 不带外层括号：带括号像标题的附注，还会跟真内容撞。
SOURCE_NOTE: dict[str, str] = {
    # ── 委托单（`briefing.py`）──
    "task": "**主 agent 交代的** —— 它根据对话写的，可能补了用户没说的东西",
    "user_turns": "**用户说的** —— 平台直接搬，未经转述",
    #: 段体混着三个来源，段级填不了，只能逐条标。
    "answered": "**逐条标了来源** —— 每一项自己带着「谁说的」（见段体里的粗体）",
    # ── 回执（`receipt.py`）──
    "asked": "**它申报的** —— `ask_user` 的参数原话，未经转述",
    "report": "**它说的** —— 原样，未经改写",
    # ── 决策（`decision.py`）──
    #: 只有【结果】要标（LLM 输出的）；另外三格是平台的框架话，按同一条规矩不标。
    "dec_result": "**它说的** —— LLM 的输出，原样",
    # ── 试答（`ask.py`）──
    #: `ask_context` 故意不在这里：【你的上下文】是平台列的，没有来源就不写来源行。
    "agent_answer": "**你说的** —— 已成对记进 Task，下次重入还会带上",
}

#: 来源那一行的前缀 —— `来源：` 后面接 `SOURCE_NOTE[key]`。
SOURCE_LABEL = "来源："


# ──────────────────────────── 状态词汇 ────────────────────────────

#: 执行器三态 —— `ExecOutcome.status` 用的就是它。由平台按结构算，不是某次 LLM 调用的
#: 结论。
OutcomeStatus = Literal["completed", "waiting_for_input", "incomplete"]

#: Task 五态 —— 台账 / 展示用的任务状态机。与三态前三个同名，唯一映射是
#: `incomplete → failed`；另两个不在三态里：`running`（刚建、还没跑完）、
#: `superseded`（同技能被重新委派）。
TaskStage = Literal["running", "completed", "waiting_for_input", "failed", "superseded"]

#: 平台故障的分类。按结构取（`status_code` / 墙钟），不从 `fault` 文本反解析。
#: - `transient`：网络 / 超时 / 限流 / 网关抖动，原样再委派；
#: - `rejected`：4xx（除 408 / 429），换措辞或改内容再委派；
#: - `timeout`：超了 `executor.MAX_WALL_S`，原样重派可以，老超时说明活该拆小；
#: - `""`：没有故障。
FaultKind = Literal["", "transient", "rejected", "timeout"]


@runtime_checkable
class Outcome(Protocol):
    """执行器一趟的结果，平台据此决定任务状态 —— 只列消费方真的会读的那几样。

    `executor.ExecOutcome` 是它的实现。契约住在这里，是因为 `handoff` 不能 import
    `executor`（成环），而 briefing / receipt / ask 都要拿它当参数类型。两边脱节由测试盯着。
    """

    status: OutcomeStatus
    artifacts: list[str]
    report: str
    asked: str
    fault: str
    fault_kind: FaultKind
    rounds: int
    wall_s: float


# ──────────────────────────── 排版 ────────────────────────────

#: 段体缩进。段名在 0 列，段体 2 列，每深一层加 2。放这里是因为所有交接点共用同一套排版。
BODY = "  "


def section_head(key: str) -> list[str]:
    """段名一行，有来源的话来源另起一行。

    段名 0 列 → 来源 2 列 → 段体 2 列，段体由调用方用 `body_lines` 接。
    """
    out = [SECTION[key]]
    note = SOURCE_NOTE.get(key)
    if note:
        out.append(f"{BODY}{SOURCE_LABEL}{note}")
    return out


def body_lines(text: object, pad: str = BODY) -> list[str]:
    """整块缩进。空行也留 `pad`，否则 markdown 里块会被空行切断。"""
    raw = str(text or "")
    if not raw.strip():
        return []
    return [f"{pad}{ln}" if ln.strip() else pad for ln in raw.splitlines()]


def bullet_lines(text: object, pad: str = BODY) -> list[str]:
    """一个可能多行的列表项 —— 首行带 `- `，续行对齐到文字，不顶回 0 列。"""
    ls = str(text or "").splitlines() or [""]
    return [f"{pad}- {ls[0]}", *(f"{pad}  {ln}" for ln in ls[1:])]


# ──────────────────────────── 平台注入的标记 ────────────────────────────

#: 谁把这条消息塞进来的，挂在 `additional_kwargs["injected_by"]` 上。
INJECTED: dict[str, str] = {
    "platform.finalize": "平台**透传**（执行器的话，原样转达给用户 —— **不是主 agent 说的**）",
}


# ──────────────────────────── 出口分类 ────────────────────────────

#: 【交付】那一行的四种头，`{}` 是变量。挑哪条看 `Receipt.kind`，不从渲染结果反推。
DELIVERY: dict[str, str] = {
    "completed": "它交付了 {} 个产物",
    "waiting": "它在**要信息**（调了 `ask_user`）",
    "failed": "它既没交付产物、也没说要什么信息（推了 {} 轮）",
    #: 单起一条：故障时"推了 N 轮"是假话，一轮都没跑完。
    "fault": "⚠️ 没跑成 —— **平台故障**",
}

#: 故障里被上游拒绝的那档。处置和抖动不同：原样重发会再被拒。
#: 不能并进 `DELIVERY`，那个字典的键就是出口分类。
DELIVERY_FAULT_REJECTED = "⚠️ 没跑成 —— **上游拒绝了这次请求**（4xx，原样重发会再被拒）"

#: 超时那档。`{}` 是跑了多久，说不清就留空。
DELIVERY_FAULT_TIMEOUT = "⚠️ 没跑成 —— **时间用完了**{}"

#: 词汇表外的故障兜底，不宣称任何动作和事实。它不该被走到。
DELIVERY_FAULT_UNCLASSIFIED = "⚠️ 没跑成 —— **平台侧没跑成**（原因未分类）"


# ──────────────────────────── 每一跳用哪几段 ────────────────────────────

#: 每一跳渲染哪几段。校验用 ⊆（渲染出来的都在声明里），因为回执的段有条件。
HOPS: dict[str, tuple[str, ...]] = {
    "briefing": (
        "task",
        "user_turns",
        "answered",
        "already_produced",
        "readable",
        "resume",
    ),
    "receipt": (
        # `asked` 紧跟【交付】：要信息的时候，那句问题就是这条回执的主角。
        "deliver",
        "asked",
        "produced",
        "report",
        "agent_answer",
        "next",
        "fault",
    ),
    "receipt_rejected": (
        "not_dispatched",
        "available_skills",
        "available_artifacts",
    ),
    "ledger": ("ledger",),
    #: 试答。顺序就是读的人理解的顺序：要什么 → 派了什么 → 它说了什么 → 你的上下文 → 请判断。
    "ask": ("asked", "task", "report", "ask_context", "ask_decide"),
    "nudge": ("nudge",),
    "relay": ("relay",),
    #: 任务文本：执行器打开就看的那一页 —— 技能指令 ＋ 已知信息（交接书）。
    "task_text": ("skill_instructions", "known"),
    #: 决策：一次 LLM 调用的两端 ＋ 平台的两句描述。
    "decision": ("dec_given", "dec_q", "dec_result", "next"),
    #: 判定。和 `decision` 共用三个段名，只有结论那格分开写。
    "verdict": ("dec_given", "dec_q", "verdict_result", "next"),
}


def render_hop(hop: str, sections: dict[str, str]) -> str:
    """所有交接单的唯一渲染入口。段名、来源、顺序、缩进都来自词汇表。

    `sections` 里没给的段跳过不渲染；`hop` 没声明过直接 `KeyError`，不静默降级。
    """
    lines: list[str] = []
    for key in HOPS[hop]:
        text = str(sections.get(key) or "").strip()
        if not text:
            continue
        lines += section_head(key) + body_lines(text, BODY)
    return "\n".join(lines)
