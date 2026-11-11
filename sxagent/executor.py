"""执行器（Sub Agent）：可选的执行者，用时才创建，一次 `run_task` 跑一趟就结束。

主 agent 用 `delegate` 派活，平台在这里起一个隔离子 agent，把技能指令和委托单交给它，跑完
把 `ExecOutcome` 交回来。

## 不变量

1. 不绑技能。技能指令随任务传进来，加一个技能不用动平台。
2. 用完即弃：不跨轮、不挂起、不与用户交互。状态在 `Task` 上，所以"重新起一个"是无害的。
3. 出口三态由结构算（`_structural_status`）。平台只看三件自己看得见的事：调过 `ask_user`、
   记过 `write_artifact`、两样都没有。不读正文、不判质量。推满 `MAX_REACT_ROUNDS` 仍没出口，
   就交回主 agent。
4. 判据的输入一样不能少（`_verdict` 的【前提】）：三件结构事实，"没有"也要显式写。
5. 状态是结构不是文本：三态用枚举，要信息走 `ask_user`，故障记 `fault_kind`，不从 `fault`
   那段文字反解析。
6. 闭包按 run 归属（`created` / `asked` / `_user_turn_reader`），不用模块级全局：两个 delegate
   并行时会互相覆盖。"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from sxagent import stream, trace
from sxagent.artifacts import (
    make_write_artifact,
    read_artifact,
)
from sxagent.config import get_model
from sxagent.handoff import FaultKind, OutcomeStatus, render_hop
from sxagent.handoff.briefing import Turn  # 用户原话快照的形状

#                ↑ 状态词汇（唯一一份）· 段名/渲染（唯一一份）
from sxagent.handoff.decision import emit_decision  # 决策事件的发射器（归领域包）
from sxagent.handoff.nudge import nudge_text  # 推回去的模板
from sxagent.skill_pack import (  # 技能包文件（磁盘）＋ 提示块（单源）
    read_skill_file,
    skill_file_prompt,
)
from sxagent.skills import Skill  # 技能领域

if TYPE_CHECKING:  # 只给标注用 —— 运行时不 import
    from langgraph.graph.state import CompiledStateGraph

#: 平台约定 —— 尽量少。技能作者写什么都不会破坏它，因为它不要求作者做任何事。
_PLATFORM_RULE = (
    "你是任务执行者。按下面的技能指令完成任务。\n"
    "要交付成文件的内容，调用 write_artifact 写入产物库：**名称、一句话梗概、正文**三样都给，\n"
    "正文写全（梗概是对外露出的那一层，别人靠它决定要不要读全文）。\n"
    "任务里列出的【可读产物】只是一份**清单**（名称与梗概）——需要全文时用 read_artifact 读，\n"
    "不要凭标题或梗概猜内容。\n"
    "\n"
    "如果缺少完成任务所必需的信息：\n"
    "- **明确说明你需要什么**（缺哪个参数、要什么确认）——你的请求会被转给能提供它的一方；\n"
    "- **不要编造**，也不要用「已完成」之类的话收尾。\n"
    "\n"
    "⚠️ **【用户答复】里不一定真的回答了你的问题**（用户可能换了个话题，或答非所问）。\n"
    "拿到答复后先核对：**你缺的那项信息，它给了吗？**\n"
    "- 给了 → 正常干活；\n"
    "- **没给 → 再问一次，不要自己挑一个默认值硬做。**\n"
    "  你猜的那个值会变成一份**基于错误前提的产物**，而用户不知道 —— 那比多问一句糟得多。\n"
    "  也不要顺手把答复里提到的**别的**需求做掉：那不是你的任务。\n"
)

#: 一个任务最多让它 react 几轮（平台自保，不约束 skill 作者）
MAX_REACT_ROUNDS = 3

#: 一整趟的墙钟上限（秒）。单次调用由 `MODEL_TIMEOUT_S` 兜，但那是一次调用的界，叠起来最坏
#: 能拖到几十分钟。它在轮次边界查，拦不住一次阻塞调用，两条一起才拦得住。
MAX_WALL_S = 600

#: 每次 `run_task` 一个递增编号，日志里用来分清哪一行属于哪一次。
_RUN_SEQ = itertools.count(1)


@dataclass
class ExecOutcome:
    """执行器的结果，平台据此决定任务状态。

    状态是枚举不是两个布尔值：布尔有四种组合、只有三种合法，"矛盾组合"只能靠约定排除。
    """

    #: 三态之一 —— 与 `Task.stage` 前三个同名（`completed` / `waiting_for_input`），
    #: 唯一映射是 `incomplete → failed`。没有第四种。
    status: OutcomeStatus = "incomplete"
    artifacts: list[str] = field(default_factory=list)
    report: str = ""
    #: 实际 react 了几轮
    rounds: int = 0
    #: 它申报要的信息（`ask_user` 的原话；空 = 没申报）。结构：平台直接看得见。
    asked: str = ""
    #: 平台故障（非空 = 这一轮根本没跑成，不是执行器干不了）。不能塞进 `report`，
    #: 故障时 `status` 仍是三态里的值。
    fault: str = ""
    #: 这一趟实际跑了多久（秒）：`run_end` 印着它，超时那条交付行也要说"跑到 X 分钟还没交"。
    wall_s: float = 0.0
    #: 平台故障的"类"（`FaultKind`）—— 四处引导句按它渲染，不许从 `fault` 反解析。
    #: 与 `fault` 同生共死：每次 `run_task` 重新赋值、`settle` 覆盖写（跑完清空）。
    fault_kind: FaultKind = ""


def make_ask_user(asked: dict[str, str]):
    """造 `ask_user` 工具 —— 执行器要信息的唯一出口。

    它一调，平台直接看见两件事：它卡住了，它要什么（`question` 的原话）。之后平台让主
    agent 先试答，答不上才把原话转给用户。这一轮到此结束，它调完工具接着说的那半截话
    不进结论。闭包按 run 归属，不用模块级全局。
    """

    @tool
    async def ask_user(question: str) -> str:
        """你**没法往下干**了，需要一条你自己拿不到的信息 —— 用这个问。

        Args:
            question: **具体、能直接回答**的一句话（如"这次要按哪个范围做？"）。
                一次问一件；**能自己判断的别问**（会打扰用户）。

        ⚠️ **要信息必须用它**：写在回复正文里**平台看不到**（平台不读正文）。
        """
        asked["question"] = str(question or "").strip()
        # 说的是「这一次登记了什么」，不是流程说明：先说刚发生的那件事，再用「接下来」
        # 交代后手。这行只是记录 —— `run_task` 看到 `asked` 就 break，它回不到执行器眼里。
        return (
            "**已登记为待答** —— 这一轮到此结束。"
            "接下来：平台先让主 agent 试答（它上下文里可能就有），"
            "答不上才把你这句**原话**转给用户。"
        )

    return ask_user


def _user_turn_reader(user_turns: list[Turn]):
    """造一个只认这次交接里那几轮的读取工具。

    用闭包不用模块级全局：并行 delegate 时两个执行器会互相覆盖。
    """
    by_n = {int(t["n"]): t for t in user_turns}

    @tool
    async def read_user_turn(turn: int) -> str:
        """读某一轮**用户原话的完整上下文**：原话全文 ＋ **它在回答的那一句**。

        交接书【用户原话】那一段列了每一轮的开头；**确定这一轮确实用得上再读** ——
        别把每一轮都拉进上下文。

        ⚠️ 像「A」这种**短应答**，脱离它回答的问题就没有意义（选项列表在那一句里）——
        所以这里给的是**整轮**，不只是那几十个字。
        """
        if turn not in by_n:
            return f"没有第 {turn} 轮。现有：{'、'.join(str(n) for n in sorted(by_n))}"
        t = by_n[turn]
        # 不用 `【…】`：它是工具返回的头，不是段名 —— 这套系统里 `【…】` 独占行首就是段名。
        out = [f"**用户原话 · 第 {turn} 轮**", str(t.get("text") or "")]
        after = str(t.get("after") or "").strip()
        if after:
            out += [
                "",
                "—— **它在回答的上一句**（可能是主 agent 说的，也可能是平台把执行器"
                "自己的问题透传给用户的那条）：",
                after,
            ]
        return "\n".join(out)

    return read_user_turn


def _at(run_id: str, n: int) -> str:
    """这一轮在哪儿 —— `R01 · 内部第 2 轮`（两个编号各自带着名字）。

    `R01` = 第几次 `run_task`（进程内递增，投递重跑 / 试答重跑都会新开一次）；
    `内部第 N 轮` = 这一次 run 里 react 循环第几圈。
    """
    return f"{run_id} · 内部第 {n} 轮"


@dataclass
class _Run:
    """一次 `run_task` 的运行态，只在 executor 内部传。

    `created` / `asked` 是工具闭包写的，所以必须和 agent 同生共死（`_open_run` 一起造）。
    """

    skill: Skill
    run_id: str
    t0: float
    #: 每轮重入的输入 —— 推回去时往它后面追加（见 `_decide_exit`）
    messages: list[BaseMessage]
    outcome: ExecOutcome
    #: 这次 run 写了哪些产物 —— 由工具自己记（闭包，按 run 归属）
    created: list[str]
    #: `ask_user` 申报的内容（闭包，同 `created`）
    asked: dict[str, str]


def _build_executor(user_turns: list[Turn] | None, created: list[str], asked: dict[str, str]):
    """建执行器那个子 agent：工具表加平台规则。

    `created` / `asked` 是工具闭包，两者得绑在同一份闭包上。
    """
    tools = [
        make_write_artifact(created),
        read_artifact,
        read_skill_file,
        make_ask_user(asked),
    ]
    if user_turns:
        tools.append(_user_turn_reader(user_turns))
    return create_agent(
        get_model(),
        tools=tools,
        system_prompt=_PLATFORM_RULE,
        name="executor",
    )


def _open_run(
    skill: Skill, briefing: str, user_turns: list[Turn] | None
) -> tuple[_Run, CompiledStateGraph]:
    """准备：建 agent、渲染这一趟的输入、写开跑前那几条日志。返回 `(运行态, agent)`。"""
    created: list[str] = []
    asked: dict[str, str] = {}
    agent = _build_executor(user_turns, created, asked)

    briefing_text = briefing.strip() or "（暂无 —— 你需要的任何信息都要先问，不要猜）"
    # 技能自带文件时把清单（路径 + 字数）一起给执行器，正文按需 `read_skill_file` 读。
    # 与 `read_skill` 工具返回同一份（单源 `skill_file_prompt`），没有自带文件就返回空串。
    refs = skill_file_prompt(skill.name)
    # 走唯一那条渲染路：交接书整块缩在【已知信息】下面，哪几段属于它一眼看得出。
    task_text = render_hop(
        "task_text",
        {
            "skill_instructions": f"{skill.instructions}{refs}",
            "known": briefing_text,
        },
    )

    run_id = f"R{next(_RUN_SEQ):02d}"
    trace.section(
        f"run_task · {run_id} · skill={skill.name}（通用执行器，隔离上下文）", kind="run_begin"
    )
    # 技能指令按名引用、不逐字复述：原文是静态资产、另有落盘文件，日志留的是交接内容。
    trace.handoff("交接书", "平台", "执行器", briefing_text, structured="Task → render_briefing")
    trace.block(
        "发给执行器的内容（技能指令按名引用）",
        # 走同一个 `render_hop`：两处拼形状必然漂。`skill_instructions` 是简写，`known` 与
        # 真发出去的逐字相同；简写里不带段名 —— 段名只有 `render_hop` 一处加。
        render_hop(
            "task_text",
            {
                "skill_instructions": f"{skill.name}"
                f"（instructions {len(skill.instructions)} 字，"
                f"原文见该技能包的 SKILL.md）",
                "known": briefing_text,
            },
        ),
        kind="executor_briefing",
    )
    return _Run(
        skill=skill,
        run_id=run_id,
        t0=time.time(),
        messages=[HumanMessage(task_text)],
        outcome=ExecOutcome(),
        created=created,
        asked=asked,
    ), agent


def _record_fault(run: _Run, rounds: int, fault: str, kind: FaultKind, note: str) -> None:
    """记一次「这一轮没跑成」，三处（墙钟 / 流式抛异常 / 流一步都没输出）共用。

    `fault` / `fault_kind` / `rounds` 三个字段同生共死，漏写一个就是"有原因但没有类"，
    四处引导句会退回默认说法——把"换措辞"说成"原样再派"。所以 `kind` 是必填位置参数。
    """
    outcome = run.outcome
    outcome.fault = fault
    outcome.fault_kind = kind
    outcome.rounds = rounds
    trace.note(note, kind="run_fault")


async def _run_round(run: _Run, agent, round_i: int) -> dict[str, object] | None:
    """跑一轮：墙钟、流式跑、把结果吸收进 `outcome`。

    返回这一轮的 `state`；没跑成就返回 `None`，故障已经记进 `outcome`，调用方看到
    `None` 直接停。
    """
    outcome = run.outcome
    # 整趟的墙钟在轮次边界查（拦不住单次阻塞调用，那个归 `MODEL_TIMEOUT_S`）：
    # 超了就不再开新一轮，按故障交回主 agent。它是故障不是"执行器干不了" ——
    # 它一直在干活，只是没干完（`timeout` 那档）。
    if time.time() - run.t0 > MAX_WALL_S:
        _record_fault(
            run,
            round_i,
            f"整趟超过墙钟上限（{MAX_WALL_S}s）",
            "timeout",
            f"⚠️ {run.run_id} **时间用完了**（已跑 {time.time() - run.t0:.0f}s > 上限 {MAX_WALL_S}s）"
            f" → 不再开新一轮，交回主 agent（原样再委派可以；老超时就拆小）",
        )
        return None
    # 不传 config —— 传了执行器会变成独立根图。`stream_agent` 边跑边录，一次 LLM 调用那
    # 几十秒不再黑屏。`who` 带技能名：多个执行器并行时流式行会交错，得能分辨是谁。
    resp = {}
    try:
        async for chunk in stream.stream_agent(
            agent, {"messages": run.messages}, None, f"执行器·{run.skill.name}", role=trace.EXECUTOR
        ):
            resp = chunk
    except Exception as e:  # noqa: BLE001
        # 一次抖动不至于让整轮失败：`stream_agent` 内部已重试过一次，两次都失败才走到这。这一轮记成
        # 平台故障（不是执行器干不了），决定权交回主 agent。不在这里重跑：一轮 60–120 秒。
        # 故障分类从结构取：4xx（除 408/429）是确定性拒绝，据此说"换措辞"而不是"原样再派"。
        _kind: FaultKind = "rejected" if stream.upstream_rejected(e) else "transient"
        _note = (
            f"⚠️ {_at(run.run_id, round_i + 1)}**上游拒绝了这次请求**（4xx，未重试 ——"
            f" 重发同一份内容会再被拒）：{type(e).__name__}: {e}"
            f" —— 这一轮没跑成，交回主 agent 换措辞/改内容"
            if _kind == "rejected"
            else f"⚠️ {_at(run.run_id, round_i + 1)}**平台故障**（已重试过一次仍失败）："
            f"{type(e).__name__}: {e} —— 这一轮没跑成，交回主 agent"
        )
        _record_fault(run, round_i + 1, f"{type(e).__name__}: {e}", _kind, _note)
        return None
    if not resp:
        # 流一步都没输出 → 按同一套故障处置：这一轮没跑成，交回主 agent
        # （主 agent 那边有同类兜底）。
        _record_fault(
            run,
            round_i + 1,
            "流式一步都没输出（stream 为空）",
            "transient",
            f"⚠️ {_at(run.run_id, round_i + 1)}**流式一步都没输出** → 这一轮没跑成，"
            f"交回主 agent（同平台故障处置）",
        )
        return None
    trace.dump_trace(_at(run.run_id, round_i + 1), resp["messages"], role=trace.EXECUTOR)

    outcome.artifacts = list(run.created)  # 记账，不扫文本
    outcome.report = _last_ai_text(resp)
    outcome.rounds = round_i + 1
    return resp


def _decide_exit(run: _Run, resp: dict[str, object], round_i: int) -> str:
    """判出口：这一轮跑完了，按结构看下一步怎么走。

    返回 `"stop"`（有出口：申报要信息 / 交了产物）或 `"again"`（推回去接着跑，或已到
    轮次上限）。只看平台看得见的三件事，不读正文、不判质量。
    """
    outcome = run.outcome
    # ① 它申报了要信息（调了 `ask_user`）→ 这一轮到此结束，后面那半截话不进结论。
    if run.asked.get("question"):
        outcome.asked = run.asked["question"]
        # 「问」是一条链，这一跳是链头：执行器 → 平台。问题原话由上面那条 TOOL 行给全。
        trace.handoff(
            "要信息",
            "执行器",
            "平台",
            outcome.asked,
            structured="`ask_user(question=…)` 的参数原话"
            "（**全文见上面那条 `🔧 TOOL`**，这里不重贴）",
        )
        trace.note(f"{_at(run.run_id, round_i + 1)}：**申报要信息** → 这一轮结束", kind="ask")
        return "stop"

    # ② 它交了产物 → 收工。平台只说"交了东西"这个事实，"够不够、对不对"由主 agent 判。
    if outcome.artifacts:
        trace.note(
            f"{_at(run.run_id, round_i + 1)}：交付 {len(outcome.artifacts)} 个产物 → 收工",
            kind="delivered",
        )
        return "stop"

    # ③ 既没交、也没申报 → 推回去（照结构事实说，不编"还缺什么"）。没干完活是它的问题，
    #    不该转嫁给用户。
    if round_i + 1 < MAX_REACT_ROUNDS:
        nudge = nudge_text()
        trace.note(
            f"⚠️ {_at(run.run_id, round_i + 1)}**既没交付、也没申报要信息**"
            f" → 推它继续（提示词见下）",
            kind="nudge",
        )
        trace.handoff(
            "推回去",
            "平台",
            "执行器",
            nudge,
            structured="结构事实：本轮无产物 ＋ 没调 `ask_user`",
            full=True,
        )
        run.messages = list(resp["messages"]) + [HumanMessage(nudge)]
    else:
        # 推满了，不再推 → 下面按结构定成 incomplete
        trace.note(
            f"⚠️ {_at(run.run_id, round_i + 1)}既没交付、也没申报，且已到轮次上限"
            f" → 不再推，交给主 agent"
        )
    return "again"


def _close_run(run: _Run) -> ExecOutcome:
    """收尾：算三态 → 记判定 → 写结束那行。"""
    outcome = run.outcome
    outcome.wall_s = time.time() - run.t0
    outcome.status = _structural_status(outcome)

    # 判定记录：那次 LLM 调用已经退休，但这个判断本身留着，正是用户要审计的那一处。
    emit_decision(**_verdict(outcome, run.run_id))

    # 这行是「这次 run 的账」，不是复述：两个编号各自带名字，问题不重贴（上面已有三处）。
    # 故障那档的说法按 `fault_kind` 取（表在模块里，守卫盯着键集合）。
    _fault_label = FAULT_LABEL.get(outcome.fault_kind, FAULT_LABEL["transient"])
    trace.note(
        f"{run.run_id} · {run.skill.name} **结束**：status={outcome.status}"
        f"（内部 {outcome.rounds} 轮，产物 {len(outcome.artifacts)} 个，"
        f"{outcome.wall_s:.1f}s）"
        + ("｜它要的那句见上面 `ask_user`" if outcome.asked else "")
        + (f"｜**{_fault_label}**：{outcome.fault}" if outcome.fault else ""),
        kind="run_end",
    )
    return outcome


async def run_task(
    skill: Skill, briefing: str = "", user_turns: list[Turn] | None = None
) -> ExecOutcome:
    """通用执行器：干活、判定出口、没出口就推它继续。

    `briefing` 是整份交接而不是增量——执行器用完即弃，每次都是重入。平台不解析、不抽取，
    原样带过去。`user_turns` 只给 `read_user_turn` 用，交接书里只渲染清单，全文按需取。

    它自己只做编排：准备 → `for 轮次: 跑一轮 → 判出口` → 收尾。
    """
    run, agent = _open_run(skill, briefing, user_turns)
    for round_i in range(MAX_REACT_ROUNDS):
        resp = await _run_round(run, agent, round_i)
        if resp is None:  # 没跑成（墙钟 / 异常 / 空流）—— 故障已经记进 outcome
            break
        if _decide_exit(run, resp, round_i) == "stop":
            break
    return _close_run(run)


def _structural_status(outcome: ExecOutcome) -> str:
    """三态由结构定，只看平台看得见的三件事。

    申报了要信息 → `waiting_for_input`；交了产物 → `completed`；都没有 → `incomplete`。
    顺序不能反：申报优先，它可能在交了一部分之后卡住。`completed` 只说"交了东西"这个
    事实，够不够由主 agent 判。
    """
    if outcome.asked:
        return "waiting_for_input"
    if outcome.artifacts:
        return "completed"
    return "incomplete"


#: 三档故障在判定记录里的说法：依据（`why`）与下一步（`nxt`）。
#: `{detail}` 是原始原因、`{wall}` 是墙钟上限，只在 `why` 里插。键 = `FaultKind`（`""` 除外）——
#: 与 `handoff.receipt` 那几张表同形，加档要一起补。
FAULT_VERDICT: dict[str, tuple[str, str]] = {
    "transient": (
        "**平台故障**（{detail}）—— 这一轮**根本没跑成**，不是执行器干不了",
        "**交回主 agent** —— 原样再 `delegate` 一次即可（不用改措辞、不用改参数、不用问用户）",
    ),
    "rejected": (
        "**上游拒绝了这次请求**（4xx，不是抖动）：{detail} —— 这一轮**根本没跑成**，"
        "不是执行器干不了",
        "**交回主 agent** —— **原样重发同一份内容会再被拒**：换措辞/改内容再 `delegate` 一次；"
        "改不动就向用户说明这一段被拦了",
    ),
    "timeout": (
        "**时间用完了**（跑满 {wall}s 墙钟）：{detail} —— 它**一直在干活**，只是这一趟没干完，"
        "不是它干不了",
        "**交回主 agent** —— 原样再 `delegate` 一次可以；**老超时就说明活太大**，拆小一点再委派",
    ),
}

#: 三档故障在 `run_end` 那行里的短标签（`…｜**平台故障**：原因`）。
FAULT_LABEL: dict[str, str] = {
    "transient": "平台故障",
    "rejected": "上游拒绝了这次请求",
    "timeout": "时间用完了",
}


def _verdict_reason(outcome: ExecOutcome) -> tuple[str, str]:
    """判定的依据（`why`）与下一步（`nxt`），两格文案的唯一来源。

    故障盖在上面：故障时 `status` 仍是三态里的值，但"这一轮根本没跑成"与"执行器干不了"
    是两回事，依据和下一步都得说故障。
    """
    if outcome.asked:
        why = "它**申报了要信息**（顺序：申报优先 → 交了产物 → 都没有）"
        nxt = "登记待答 → 平台强制**主 agent 试答**（答得上就带回去重跑，答不上**原样**转给用户）"
    elif outcome.artifacts:
        why = "它**交了产物**（同一顺序；本轮**没申报**）"
        nxt = (
            "**回执给主 agent** —— 由它判「够不够」（平台不判质量；"
            "判据以**技能声明的交付标准**为准）"
        )
    else:
        why = "**既没申报、也没交产物**（顺序里没有命中的那一条）"
        nxt = "**交回主 agent**（推满轮次仍没出口）—— 它看回执定：换个说法**重新委派** / 向用户说明"
    if outcome.fault:
        # 故障时 status 仍是三态里的值，但这一轮根本没跑成 —— 与"执行器干不了"是两回事。
        # 三档的说法在 `FAULT_VERDICT` 表里，跟 `handoff.receipt` 那几张同形（守卫盯着键集合）。
        why, nxt = FAULT_VERDICT.get(outcome.fault_kind, FAULT_VERDICT["transient"])
        why = why.format(detail=outcome.fault, wall=MAX_WALL_S)
    return why, nxt


def _verdict(outcome: ExecOutcome, run_id: str) -> dict[str, object]:
    """「任务完成没」这条判定记录的四个字段（纯函数，可单测）。

    前提是平台看得见的三件结构事实（申报、产物账、轮次，包括"没有"），结果是三态加命中
    的那条。它没有 LLM 参与，却用和决策记录同一套形状——四格问的是同一件事。
    """
    why, nxt = _verdict_reason(outcome)
    return {
        "who": f"任务完成没（平台按结构定）· {_at(run_id, outcome.rounds)}",
        # 这一格的作者是平台（不是 LLM）—— 它决定那格用【平台判定】而不是【结果】。
        # 标错了就是假话。
        "result_by": "platform",
        "q": "这个任务完成没？（三态**由结构定** —— 平台只看上面那三件事实，"
        "**不读正文、不判质量**）",
        "given": [
            "**申报要信息**（`ask_user`）："
            + (
                "**有** —— 原话见上面那条 `ask_user` 的参数"
                if outcome.asked
                else "**没有**（本轮没调它）"
            ),
            "**交付产物**（`write_artifact` 记的账）："
            + (
                f"**{len(outcome.artifacts)} 个** —— {'、'.join(outcome.artifacts)}"
                if outcome.artifacts
                else "**没有**（本轮没调它）"
            ),
            f"**轮次**：内部第 {outcome.rounds} 轮（上限 {MAX_REACT_ROUNDS}）",
        ],
        "result": [f"status = **{outcome.status}**", f"依据：{why}"],
        "nxt": nxt,
    }


def _last_ai_text(resp: dict[str, object]) -> str:
    """执行器的报告，取最后一条有内容的 AI 消息。

    只用于给主 agent 的摘要，不参与任何判断——平台只看结构，而这句原样进回执（标着
    "它说的"）。
    """
    for m in reversed(resp["messages"]):
        if isinstance(m, AIMessage) and str(m.content).strip():
            return str(m.content)
    return ""
