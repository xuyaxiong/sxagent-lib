"""主 agent：一个完整的 agent，有自己的工具和循环，不是只做转发的路由器。

每轮走哪条路由它自己决定：直接答、自己写一段、或者 `delegate` 派活。只有第三条会建 Task。
`delegate` 里四件事由平台代码保证，模型绕不过去：取代同技能的旧 Task、执行器在隔离上下文
里干活、主 agent 先用自己上下文试答、答不上才登记待答。

## 不变量

1. 不反解析。回执对象随 `ToolMessage.additional_kwargs` 走，记录层读对象；`_StopWhenPending`
   读 `handoff.kind`，不读文本、不猜语义。
2. 交接原样搬。用户原话由平台搬（`build_user_turns`），执行器要什么取 `ask_user` 的原话
   （`asked`）——转述既会丢，也会加。
3. 状态迁移只有一处入口：`dispatch` 里的 `settle()`。手写 `rec.update(stage=…)` 是无条件
   覆盖，会把"这一轮没产出就不抹掉上一轮产物"当场抵消。
4. 一次决策只留一条记录：前提、问题、结果、下一步。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # 只给标注用 —— 运行时不 import
    from langgraph.graph.state import CompiledStateGraph

from sxagent import (
    executor,  # 走模块属性，见 delegate 的注释
    trace,
)
from sxagent.artifacts import (  # 产物领域
    DEFAULT_STORE,
    ArtifactStore,
    artifact_index_lines,
    make_list_artifacts,
    make_read_artifact,
    make_write_artifact,
)
from sxagent.config import get_model
from sxagent.handoff import Outcome  # 执行器结果的契约（实现在 executor，由测试盯着）
from sxagent.handoff.ask import build_prompt as ask_prompt  # 试答
from sxagent.handoff.briefing import (  # 委托单
    Task,
    Turn,
    absorb_outcome,
    moot_older_waiting,
    render_briefing,
    settle,
)
from sxagent.handoff.decision import emit_decision  # 决策事件的发射器（归领域包）
from sxagent.handoff.receipt import (  # 回执
    NOTE_BY_FAULT,
    NOTE_FAULT_UNCLASSIFIED,
    build_receipt,
    build_rejection,
    next_note,
    render_receipt,
)
from sxagent.llm import safe_structured
from sxagent.skill_pack import (  # 技能来源：库自带的默认实现 ＋ 两个读工具的工厂
    DEFAULT_SOURCE,
    make_read_skill,
    make_read_skill_file,
)
from sxagent.skills import Skill, SkillSource, render_catalog  # 技能领域 ＋ 来源接口

#: 主 agent 的 system prompt。它是平台约定，不约束 skill 作者。
_MAIN_PROMPT = (
    "你是一个能自己干活的助手，**不必把所有事都委派出去**。\n"
    "\n"
    "两种做法：\n"
    "1. **直接答**：闲聊、追问、解释、澄清、简单修改——不需要专业技能的事，你自己说。\n"
    "2. **delegate 委派**：需要下面列出的**专业技能**时，**一律委派**，不要自己凭感觉做。\n"
    "   执行器在独立上下文里按该技能的指令干活，只把产物和摘要带回来。\n"
    "\n"
    "为什么必须委派：专业技能的产出可能很长，自己干会把你的上下文撑爆。\n"
    "**你在委派前无法判断产物有多长**，所以不要猜——需要专业技能就委派。\n"
    "\n"
    "委派回来如果显示「它在**要信息**」——**什么都别做，尤其不要自己补一个答案**：\n"
    "平台已经替你查过一次上下文了（`delegate` 内部做的）；确实没有，\n"
    "平台会把**它申报的那句原话**贴给用户，用户答复后也会**投回给那个执行器**。\n"
    "🔴 **你这条回复只说「委派了什么」就行 —— 问题原文由平台贴，你不要写。**\n"
    "（问题内容在台账那行「**它要的信息**」里，你看得到；看得到是为了**别重复问**，\n"
    "不是让你转述 —— 转述会和平台那条**重复**，用户会看到同一个问题两遍。）\n"
    "\n"
    "🔴 **绝不要替用户定一个答案再委派一次。** 用户没说过的，就是没有 ——\n"
    "你猜的那个值会变成一份基于错误前提的产物，而用户不知道。\n"
    "\n"
    "## 信息不全时，**不要一律问** —— 先判断能不能推进\n"
    "\n"
    "- 能推进就**先推进**（真的缺东西时，执行器会告诉你）；\n"
    "- 确实推进不了的，平台会问；你不用抢着问，也不用挤牙膏；\n"
    "- **哪些参数是必答的、哪些有默认值，以技能指令为准** —— 平台不替你猜，你也别猜。\n"
    "\n"
    "## 🔴 产出范围**以用户明确说的为准**，不要自行扩大\n"
    "\n"
    "用户说「做 A」就**只做 A**；不要说「我顺便把 B 也做了」。\n"
    "**用户没要的，不要直接做掉**——那会浪费他的时间和成本。\n"
    "\n"
    "想提供的话，**在回复末尾提一句**就行（如「需要的话我也可以做 B」），\n"
    "让用户自己决定要不要。\n"
    "\n"
    "同理，用户说了「A、B、C」就是**三样**，不要多也不要少。\n"
    "\n"
    "## 🔴 前序产物怎么交给下一个任务\n"
    "\n"
    "**传 ID，不要转述正文，也不用先读它。** 需要前面交付的产物当输入时：\n"
    '`delegate(skill_name="...", task="...", artifacts=["art_001"])`\n'
    "执行器会拿到它的名称与梗概，需要全文**它自己读**。\n"
    "\n"
    "- **别读**：转交之前先 `read_artifact` 读一遍，等于把那几千字**永久背进你的上下文**\n"
    "  （读一次就一直在），隔离就白做了 —— 而你并不需要它。\n"
    "- **别抄**：`task` 只写「这次要什么」（课题、学情、用途、约束），\n"
    "  **不要写「内容是什么」** —— 内容由执行器从产物里读，你复述一遍既失真又多余。\n"
    "- 不确定有哪些产物 → `list_artifacts()`（只有 ID、名称、字数、梗概，很便宜）。\n"
    "- 不确定某个技能**交付什么、按什么标准交付** → `read_skill(name)` 读原文核对，\n"
    "  **不要凭一句话描述猜**：你猜出来的要求会压过技能自己的交付标准。\n"
    "- 传了不存在的产物 ID 会被当场拒绝，并告诉你现有的清单。\n"
    "\n"
    "## 🔴 用户原话由平台**原样转交**，你不用复述\n"
    "\n"
    "你委派时**不用**复述用户说过的话，也**不用**标注「这条需求出自哪一轮」——\n"
    "平台会把**每一轮用户原话**都放进执行器手里的索引（序号 · 字数 · 开头 · 它在回答什么），\n"
    "需要哪一轮的全文，执行器自己按需读。**比你转述可靠**：你写的是理解，原文是事实。\n"
    "\n"
    "所以 `task` 只写「这次要什么」—— 不要写「内容是什么」。\n"
    "\n"
    "## 🔴 回复用户时的硬要求\n"
    "\n"
    "**产物正文不要搬进你的回复。** 委派完成后，只告诉用户这四样：\n"
    "交付了哪几样、各自的产物 ID、**文件在哪**、以及执行器给的**梗概**。\n"
    "\n"
    "- 🔴 **文件路径一定要给** —— `delegate` 的返回里带着它（形如"
    "「位置：art_001 → <路径>」）。\n"
    "  用户是拿着这个路径去**打开文件**的：**只说 ID，他找不到东西**。\n"
    "  （实测：路径已经给到你了，但你有时会漏掉它、只报 ID 和摘要 —— 别漏。）\n"
    "\n"
    "- **不要**用 `read_artifact` 把产物正文读出来贴进回复——产物是给用户看的卡片/文件，\n"
    "  正文贴进来会**撑爆你的上下文**，隔离就白做了。\n"
    "- **不要**自己用 `write_artifact` 重新写一遍执行器已经交付的内容。\n"
    "- `read_artifact` 是**按需**能力（渐进式披露第二层）：用户明确说「给我看看内容」\n"
    "  「改一下某处」、或你自己确实要核对内容时才读。**转交给下一个任务时不要读** ——\n"
    "  传 ID 就够，执行器自己会读。\n"
    "\n"
    "`write_artifact` 只用于**你自己产出**的短内容（比如一段说明、一份对比表）；\n"
    "你自己产出时同样要给**名称、一句话梗概、文件类型、正文**四样，\n"
    "交付后照上面那条把**文件路径**告诉用户。"
)


# ──────────────────────────── 主 agent 试答 ────────────────────────────

#: 这一步的名字，业务侧唯一一份。展示层直接用 `ev["title"]`，不自己再拼一个。
_ASK_STEP = "试答（LLM）· 平台强制的一步"


class _AgentAnswer(BaseModel):
    can_answer: bool = Field(description="你的上下文里有没有它缺的信息")
    answer: str = Field(default="", description="can_answer=true 时给出补充内容，原话照搬不要改写")


async def _ask_main(
    messages: list[BaseMessage],
    skill_name: str,
    task: str,
    outcome: Outcome,
    model: BaseChatModel | None,
) -> str:
    """平台强制的一步：让主 agent 用它自己的上下文，试着补上执行器缺的那条信息。

    "它要什么"必须给全——那是执行器自己申报的原话，不用从它说的话里挑。
    """
    # 不截断：这是独立的结构化调用，只看这个提示词。信息落在长消息后半段它答不上来，只能去
    # 打扰用户。别退回"最近 N 条"：这里问的就是"你的上下文里有没有"。
    msgs = messages or []
    prompt = ask_prompt(skill_name, task, outcome, msgs)
    # 这一步把主 agent 的上下文又发了一遍：这行是清单（多少字 ＋ 什么构成），正文默认
    # 不逐字记（`SXAGENT_LOG_PROMPT`），所以别在这里写"逐字见下"。
    trace.handoff(
        "试答", "平台", "主 agent", prompt, structured=f"ExecOutcome ＋ 全部 {len(msgs)} 条消息"
    )
    # 调用失败时保守当作「答不上」—— 交给用户，而不是编一个答案
    r = await safe_structured(
        _AgentAnswer,
        prompt,
        _AgentAnswer(can_answer=False),
        model=model,
        tag=_ASK_STEP,
        kind="ask_main",
        # 返回不原样再记：下面那条决策记录里已有排好版的「结果」，再来一条 repr 就是两副
        # 面孔。
        log_response=False,
    )
    # 一次决策 = 一条记录：前提 → 问题 → 结果 → 下一步。前提的全文是上面那条 PROMPT，
    # 这里只说清"它看的是什么"。
    ans = r.answer.strip()
    emit_decision(
        _ASK_STEP,
        q="**主 agent 的上下文里**，有没有执行器要的那个信息？（有就不打扰用户）",
        given=[
            # 它要什么 = `ask_user` 的原话；在这儿给全，否则"答不上"这条结论没有指涉对象。
            f"**它要的那句**（执行器 → 平台，`ask_user` 原话，{len(outcome.asked)} 字）："
            f"{outcome.asked}",
            f"它最后说的那段（{len(outcome.report)} 字，可能有上下文、也可能只是过场）",
            # 指向清单和开关，不指"上面那条 PROMPT"：正文默认不逐字记，那句会变成假话。
            f"主 agent 的全部 {len(msgs)} 条消息 —— **全文、不截**"
            f"（逐条已各自记录；拼好的整份置 `SXAGENT_LOG_PROMPT=1` 可看）",
        ],
        result=[
            f"can_answer = **{r.can_answer}**",
            f"补充：{ans or '（无）'}",
        ],
        # 两个出口都写清"谁 → 谁"：只写"答不上 → 登记待答"，读的人得自己补主语。
        nxt=(
            "**主 agent 答得上** → 成对记进 `Task.qa`（`by=主 agent`）→ 带回去重跑执行器，"
            "**不打扰用户**"
            if r.can_answer
            else "**主 agent 答不上 → 转给用户**：登记待答，由**平台**把执行器原话**透传**给用户"
        ),
    )
    return ans if r.can_answer else ""


async def try_answer_then_rerun(
    rec: Task,
    skill: Skill,
    outcome: Outcome,
    *,
    messages: list[BaseMessage],
    artifacts: ArtifactStore = DEFAULT_STORE,
    skills: SkillSource = DEFAULT_SOURCE,
    model: BaseChatModel | None = None,
) -> tuple[Outcome, str]:
    """执行器说「缺信息」→ 主 agent 先试答 → 答得上就带回去重跑一遍。

    首次派发和投递重跑共用这一套。返回 `(最终 outcome, 试答)`；试答为空表示答不上，由
    调用方登记待答。
    """
    answer = await _ask_main(messages, skill.name, rec.get("task") or "", outcome, model)
    if not answer:
        return outcome, ""
    # 成对记：只记答案的话，下次重入时 `report` 已被新问题覆盖，这条答案就没了指涉。
    # 「问」取 `outcome.asked`（`ask_user` 的原话）而不是 `report`（混着过场的那段话）。
    rec["qa"] = [*(rec.get("qa") or []), {"by": "主 agent", "q": outcome.asked, "a": answer}]
    # 试答后重跑也是重入：走同一份交接书（带上刚才那一轮的进度、它问过什么）
    o2 = await executor.run_task(
        skill,
        render_briefing(rec, artifacts),
        rec.get("user_turns"),
        artifacts=artifacts,
        skills=skills,
        model=model,
    )
    return o2, answer


async def dispatch(
    skill: Skill,
    rec: Task,
    tasks: list[Task],
    *,
    trigger: str,
    messages: list[BaseMessage],
    artifacts: ArtifactStore = DEFAULT_STORE,
    skills: SkillSource = DEFAULT_SOURCE,
    model: BaseChatModel | None = None,
) -> tuple[Outcome, str, list[Task]]:
    """派发编排，`delegate` 与 `graph._feed_waiting` 共用的那一段。

    顺序不能调：跑执行器 → 落账 → 记 `run_by`（台账据此说清"全文在哪"）→ 要信息就
    试答重跑 → `settle`（状态迁移唯一入口）→ 完成了就取代更早的哑待答。顺序写两遍
    就一定会漂，所以收在一处。

    只做编排，不管出口措辞。返回 `(最终 outcome, 试答, 被取代的 Task)`。
    """
    outcome = await executor.run_task(
        skill,
        render_briefing(rec, artifacts),
        rec.get("user_turns"),
        artifacts=artifacts,
        skills=skills,
        model=model,
    )
    absorb_outcome(rec, outcome)
    rec["run_by"] = trigger
    answer = ""
    if outcome.status == "waiting_for_input":
        outcome, answer = await try_answer_then_rerun(
            rec, skill, outcome, messages=messages, artifacts=artifacts, skills=skills, model=model
        )
    settle(rec, outcome)
    mooted = moot_older_waiting(rec, tasks) if outcome.status == "completed" else []
    return outcome, answer, mooted


def _cmd(
    runtime: ToolRuntime,
    recs: list[Task] | None,
    msg: str,
    *,
    handoff: dict[str, object] | None = None,
) -> Command:
    """返回这几条 Task，合并交给 state 的 reducer。

    `handoff` 是回执对象，挂在 `additional_kwargs` 上搭同一趟车：模型只看 `content`，
    记录和审计层读对象。
    """
    update: dict[str, object] = {
        "messages": [
            ToolMessage(
                content=msg,
                tool_call_id=runtime.tool_call_id,
                additional_kwargs=({"handoff": handoff} if handoff else {}),
            )
        ]
    }
    if recs:
        update["tasks"] = recs
    return Command(update=update)


def _reject(runtime: ToolRuntime, kind: str, detail: str, available: str) -> Command:
    """派发被拒（不建 Task）→ 回执对象 ＋ 渲染，与正常回执走同一条路。"""
    r = build_rejection(kind, detail, available)
    return _cmd(runtime, None, render_receipt(r), handoff=r)


def _supersede_old(runtime: ToolRuntime, skill_name: str) -> list[Task]:
    """把同 skill 下已经死掉的旧 Task 标成 `superseded`。

    只取代 `failed` 的。`waiting_for_input` 一律不碰：重派不是解除待答的路径，碰它只会
    静默抹掉一个用户从没回答过的问题。`completed` 的是历史，不能抹。
    返回完整 Task 字典（不是 patch），reducer 按 id 整体覆盖。
    """
    out: list[Task] = []
    for t in runtime.state.get("tasks") or []:
        if t.get("skill") == skill_name and t.get("stage") == "failed":
            out.append({**t, "stage": "superseded"})
    return out


# ──────────────────────────── 用户原话（平台硬搬） ────────────────────────────


def build_user_turns(messages: list[BaseMessage]) -> list[dict[str, object]]:
    """把对话历史里的用户原话抽出来，按顺序编号，平台硬搬不经转述。

    只认 `HumanMessage`：主图里注入的台账是 `SystemMessage`、透传是 `AIMessage`，所以
    "HumanMessage 就是用户本人说的"在主图里成立（执行器自己的图里不成立，别复用）。
    存全文，交接书里只渲染开头几个字，全文按需 `read_user_turn(n)` 取。
    """
    out: list[dict[str, object]] = []
    last_ai = ""
    for m in messages:
        if isinstance(m, AIMessage):
            text = str(m.content).strip()
            if text:
                last_ai = text
            continue
        if not isinstance(m, HumanMessage):
            continue
        text = str(m.content).strip()
        if text:
            out.append({"n": len(out) + 1, "text": text, "after": last_ai})
            # 一个"上文"只配一轮：连着两条用户消息时，第二条的语境是第一条，
            # 不该再挂同一条 AI 消息（挂了就是错的上下文，比没有更糟）。
            last_ai = ""
    return out


# ──────────────────────────── 委派 ────────────────────────────


def _new_task(skill_name: str, task: str, tid: str, refs: list[str], turns: list[Turn]) -> Task:
    """建委托单 —— 这次委派的全部已知（schema 见 `handoff.briefing.Task`）。

    `total=False` 是如实：注入的 Task 可能只有部分字段。
    """
    return {
        "id": tid,
        "skill": skill_name,
        "task": task,
        "stage": "running",
        "artifacts": [],
        "report": "",
        # 它申报要的那条信息（`ask_user` 的原话），每次 `settle` 覆盖写
        "asked": "",
        # 平台故障的原因（空 = 没故障）与它的类（`FaultKind`），同样覆盖写
        "fault": "",
        "fault_kind": "",
        # `artifacts` 是产出（产物 ID），`refs` 是输入引用 —— 两个方向，别混。待答投递要
        # 按 refs 原样重建可读产物清单。
        "refs": refs,
        # 历次问答对，累积在 Task 上：执行器用完即弃，每次重入都要重放全部已知。`q` 取
        # `asked` 而不是 `report`：平台不解析散文。
        "qa": [],
        #: 这轮是谁起的（台账据此说清"全文在哪"）：`delegate` 这一趟的全文在回执里。
        "run_by": "delegate",
        # 用户原话快照（不是渲染时现取）：答复投递那条路也调 render_briefing，快照在 Task
        # 上它自动跟着走，而且与 `qa` 天然去重。
        "user_turns": turns,
    }


def _out(
    runtime: ToolRuntime,
    rec: Task,
    superseded: list[Task],
    outcome: Outcome,
    *,
    agent_answer: str = "",
    note: str = "",
    mooted: list[Task] | None = None,
    artifacts: ArtifactStore = DEFAULT_STORE,
) -> Command:
    """统一的返回口，总是带上被取代的旧 Task。

    一次做完两件事：构造回执对象、渲染成文本。文本进 `content`（模型只看这个），对象挂
    `additional_kwargs`（记录层读它）。
    """
    r = build_receipt(outcome, agent_answer=agent_answer, note=note, store=artifacts)
    text = render_receipt(r)
    # 交接关键日志（复盘用）：这一跳交了多少、有没有结构化对象。
    # 正文不在这里 dump：主 agent 的逐步骤 dump 里已经有它（`graph.node_agent`）。
    trace.handoff(
        "回执",
        "执行器",
        "主 agent",
        text,
        structured=f"Receipt(kind={r['kind']}，{len(r['sections'])} 段)",
    )
    return _cmd(runtime, [rec, *superseded, *(mooted or [])], text, handoff=r)


def _reply(
    runtime: ToolRuntime,
    rec: Task,
    superseded: list[Task],
    tid: str,
    skill: Skill,
    outcome: Outcome,
    *,
    answer: str = "",
    mooted: list[Task] | None = None,
    artifacts: ArtifactStore = DEFAULT_STORE,
) -> Command:
    """把这一趟的结果说回主 agent：按三态枚举分派，穷举，没有第四种。"""
    # ① 完成了 —— 平台不下"完成了"的结论：三态是结构算的，"够不够"由主 agent 看回执判。
    if outcome.status == "completed":
        return _out(
            runtime,
            rec,
            superseded,
            outcome,
            agent_answer=answer,
            note=next_note("completed", skill=skill.name),
            mooted=mooted,
            artifacts=artifacts,
        )

    # ② 在要信息 —— 试答已在 `dispatch` 里走过；走到这里 = 答不上（或答了仍缺）→
    #    登记待答，执行器问的那句由平台原样转给用户。
    if outcome.status == "waiting_for_input":
        trace.note(
            f"delegate · {tid}（{skill.name}）**登记待答** —— 执行器问的那句"
            f"由平台原样转给用户（不由主 agent 代问）",
            kind="waiting",
        )
        return _out(
            runtime, rec, superseded, outcome, note=next_note("waiting"), artifacts=artifacts
        )

    # ③ 只剩 incomplete：推了 N 轮，既没交产物也没申报要信息，交给主 agent 决定。平台不猜
    #    它为什么没交（不读正文）。平台故障 ≠ 执行器干不了：故障时"推了 N 轮"是假话，
    #    主 agent 该做的事也不同（原样重派 vs 换说法重派）。
    if outcome.fault:
        # 三档故障三种说法：措辞按 `fault_kind` 从对象取；表住 `handoff.receipt`，不在这儿。
        trace.note(
            f"delegate · {tid} " + NOTE_BY_FAULT.get(outcome.fault_kind, NOTE_FAULT_UNCLASSIFIED),
            kind="delegate_failed",
        )
        return _out(
            runtime,
            rec,
            superseded,
            outcome,
            note=next_note("fault", fault_kind=outcome.fault_kind),
            artifacts=artifacts,
        )
    trace.note(
        f"delegate · {tid} status=incomplete（推了 {outcome.rounds} 轮）→ failed",
        kind="delegate_failed",
    )
    return _out(runtime, rec, superseded, outcome, note=next_note("failed"), artifacts=artifacts)


def make_delegate(
    store: ArtifactStore = DEFAULT_STORE,
    source: SkillSource = DEFAULT_SOURCE,
    model: BaseChatModel | None = None,
):
    """造一个 `delegate` 工具，落在给定的产物库与技能来源上。

    准入要按来源查技能在不在、按产物库查产物在不在，回执要按产物库列清单。
    """

    @tool
    async def delegate(
        skill_name: str,
        task: str,
        runtime: ToolRuntime,
        artifacts: list[str] | None = None,
    ) -> Command:
        """把**一件具体工作**委派给隔离的执行器（它在独立上下文里干，只把产物和**梗概**带回来）。

        **需要专业技能时一律用它**；闲聊、追问、澄清自己直接答。
        执行器在独立上下文里干活，产出不会撑爆你的上下文。

        ⚠️ **用户原话由平台原样转交**：每一轮都在执行器手里的索引里（序号 · 字数 · 开头 ·
        它在回答什么），需要哪一轮的全文它自己按需读。所以你**不用**在 `task` 里复述用户的话，
        也不用标注依据哪一轮。

        Args:
            skill_name: 技能名。
            task: 这件工作的具体要求。**不要把前序产物的正文抄进来** —— 需要它作为输入时，
                把产物 ID 放进 `artifacts`，执行器会自己读全文（转述既失真、又白占你的上下文）。
            artifacts: 这次工作要参考的产物 ID（如 `["art_001"]`）。不确定有哪些产物先
                `list_artifacts()`；不确定某个技能交付什么先 `read_skill(name)`。

        ⚠️ **它自己只做四步**：**准入 → 立案 → 执行 → 回话**。
        后三步各在一个私有件里（`_new_task` / `dispatch` / `_reply`）。
        """
        # ① 准入 —— 技能不存在 / 引用了不存在的产物，当场拒：编出来的 ID 放行的话，执行器
        #    读不到会自己找补，错误被埋进产物里。
        skill = source.get(skill_name)
        if skill is None:
            return _reject(
                runtime,
                "rejected_skill",
                f"❌ 没有技能「{skill_name}」",
                "、".join(s.name for s in source.values()),
            )
        refs = [a for a in (artifacts or []) if a]
        unknown = [a for a in refs if not store.has(a)]
        if unknown:
            # 用换行拼，不用"；"：清单每项可能两行（梗概 ＋ 路径），分号拼会挤成一大段。
            have = "\n".join(artifact_index_lines(store)) or "（本会话还没有产物）"
            return _reject(
                runtime, "rejected_artifact", f"❌ 产物 {'、'.join(unknown)} 不存在", have
            )

        # ② 立案 —— 这次委派记成一条 Task。用户原话平台硬搬（快照进 Task），全文按需取。
        turns = build_user_turns(runtime.state.get("messages") or [])
        # id 用 tool_call_id 派生：并行调用时不会撞车。必须取尾部：前几位所有调用都一样。
        tid = f"t{(runtime.tool_call_id or 'x')[-8:]}"
        # 同 skill 下还没了结的旧 Task → 标记 superseded，免得台账里挂着过期项
        superseded = _supersede_old(runtime, skill_name)
        rec = _new_task(skill_name, task, tid, refs, turns)
        # kind：语义标签，展示层（订阅事件的那层）按它归角色 —— 不靠子串认行
        trace.section(f"delegate · {skill_name} · {tid}", kind="dispatch")
        if superseded:
            trace.note(
                f"取代了 {len(superseded)} 个同技能的未了结 Task：{[t['id'] for t in superseded]}",
                kind="supersede",
            )

        # ③ 执行 —— 编排（run_task → 落账 → 试答 → 落账 → 取代）走两条路共用的那个函数；
        #    `run_task` 内部已含出口限制：既没完成也没提问会被推回去继续。
        outcome, answer, mooted = await dispatch(
            skill,
            rec,
            runtime.state.get("tasks") or [],
            trigger="delegate",
            messages=runtime.state.get("messages") or [],
            artifacts=store,
            skills=source,
            model=model,
        )

        # ④ 回话 —— 按三态说回主 agent（穷举，没有第四种）。
        return _reply(
            runtime,
            rec,
            superseded,
            tid,
            skill,
            outcome,
            answer=answer,
            mooted=mooted,
            artifacts=store,
        )

    return delegate


# ──────────────────────────── 装配 ────────────────────────────

#: 执行路径开关。跑批时想验某条链路，得能把它固定住。
#: `inline` 是全硬的（工具根本不存在），`delegate` 是半硬的（拿掉原料，但挡不住它直接写）。
EXEC_PATH = "auto"


def _tools_for(
    exec_path: str, store: ArtifactStore, source: SkillSource, model: BaseChatModel | None
) -> list[BaseTool]:
    """这条执行路径给哪些工具。改这里时想清楚：拿掉某个工具会不会顺手拿掉一条合法能力。

    按装配现造（不缓存）：产物那几个工具闭包住注入的产物库，技能那两个闭包住注入的来源。
    """
    return {
        "auto": [
            make_delegate(store, source, model),
            make_write_artifact(store=store),
            make_read_artifact(store),
            make_list_artifacts(store),
            make_read_skill(source),
            make_read_skill_file(source),
        ],
        "delegate": [make_delegate(store, source, model), make_list_artifacts(store)],
        "inline": [
            make_write_artifact(store=store),
            make_read_artifact(store),
            make_list_artifacts(store),
            make_read_skill(source),
            make_read_skill_file(source),
        ],
    }[exec_path]


def tool_names() -> set[str]:
    """主 agent 可能调到的全部工具名 —— 三种执行路径的并集。

    审计层按它列表来源（`app.runtime.TOOL_SOURCE`）：加了工具忘了补表，记录里那条返回的
    来源会静默回落成"平台"。测试拿这个函数断覆盖性。
    """
    return {
        tool.name
        for mode in ("auto", "delegate", "inline")
        for tool in _tools_for(mode, DEFAULT_STORE, DEFAULT_SOURCE, None)
    }


#: 非默认模式下往系统提示里补一句 —— 工具集变了，得让它知道为什么。
_MODE_NOTE: dict[str, str] = {
    "delegate": (
        "\n\n🔴 **本轮强制委派**：专业技能的活**一律**用 `delegate` 委派出去，"
        "**不要自己在回复里写产物** —— 你手边既没有产出工具，也没有技能原文。"
    ),
    "inline": (
        "\n\n🔴 **本轮不委派**：你手边**没有 `delegate` 工具**，"
        "专业技能的活自己按技能清单和技能原文做。"
    ),
}


class _StopWhenPending(AgentMiddleware):
    """执行器一申报待答，主 agent 这一拍就不生成 —— "待答时复述问题"的解法。

    那一刻要决定的是「结果回给执行器，还是问用户」，不是对话轮次；ReAct 的默认规则
    "工具返回了 → 再生成一拍"把它当普通轮次，于是它只能复述问题。正解不是拦住输出，
    是让那一拍不发生：中间件在模型调用之前就跳走（不是生成完再压住）。
    触发条件是结构事实：读回执的 `handoff.kind`，只有 `"waiting"` 才停，其余照常发言。
    """

    @hook_config(can_jump_to=["end"])
    # 这两个参数跟着 langchain 的中间件签名，标不出更细的（状态类型住在 `graph.py`，import 会成环）
    def before_model(self, state: Any, runtime: Any) -> dict[str, object] | None:
        """这一轮平台要贴一条新问题出去 → 结束这一轮，模型不会被调用。

        两种形状都要认，少认一种就是"拦一半"：

        形状一「本批回执」：主 agent 自己调的 `delegate`。要看整批，不能只看最后一条 ——
        模型会并行调工具，把 `delegate` 和 `read_skill` 放在同一条消息里时，待答回执不排在
        最后，只看末尾那条会让机制静默失效。"这一批" = 末尾连续的那串 `ToolMessage`。

        形状二「投递后又问」（`state["fed_asking"]`）：这条路上根本没有 `delegate` 调用，
        消息末尾是 `HumanMessage`，上面那轮扫描扫不到任何工具返回。事实由平台写在
        `AgentState.fed_asking` 上（本轮信号，每轮重写）。
        """
        # ── 形状一：本批回执 ──
        for m in reversed(state.get("messages") or []):
            if not isinstance(m, ToolMessage):
                break
            handoff = (m.additional_kwargs or {}).get("handoff") or {}
            if handoff.get("kind") == "waiting":
                return self._stop("这批工具返回里有待答回执")
        # ── 形状二：投递后又问 ──
        if state.get("fed_asking"):
            return self._stop("投递之后执行器又问了一个")
        return None

    @staticmethod
    def _stop(why: str) -> dict[str, object]:
        """单独报一行：读日志的人得看得出"主 agent 这一轮被平台收了话" ——
        不然只会以为它没说话。`why` 写清是哪条路拦下的，复盘要看。
        """
        trace.note(
            f"主 agent **这一轮不发言**（{why}，问题由平台原样贴给用户） → `before_model` 跳 end",
            kind="",
        )
        return {"jump_to": "end"}


def _check_tool_names(tools: Sequence[BaseTool]) -> None:
    """工具名不许撞。

    撞了模型叫哪个都说不清；而且提示词里写着"用 `write_artifact` 交付"这类话，两个同名的工具
    会让那句话变成假话。宿主的领域工具和库的机制工具都不许重名。
    """
    dupes = sorted(name for name, n in Counter(t.name for t in tools).items() if n > 1)
    if dupes:
        raise ValueError(
            f"工具名撞了：{'、'.join(dupes)} —— 库的机制工具和宿主的领域工具不许重名，"
            "改宿主那个工具的名字（或者拿掉它，用库的同名工具）"
        )


def build_agent(
    state_schema: type,
    *,
    artifacts: ArtifactStore = DEFAULT_STORE,
    skills: SkillSource = DEFAULT_SOURCE,
    model: BaseChatModel | None = None,
    extra_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
    """主 agent：一个完整的 agent（工具 + 循环）。

    工具集 = `EXEC_PATH` 定的那组机制工具（`delegate` / `write_artifact` / …）**加上**宿主的
    领域工具；重名当场报错。机制工具不可摘：摘了机制就不完整，提示词里那些话也成了假话。
    `artifacts` 是宿主注入的产物库，`skills` 是技能来源（磁盘 / OSS / 数据库），`model` 是
    模型实例 —— 三者不传就用库自带的。
    """
    tools = [*_tools_for(EXEC_PATH, artifacts, skills, model), *extra_tools]
    _check_tool_names(tools)
    return create_agent(
        model or get_model(),
        tools=tools,
        system_prompt=_MAIN_PROMPT
        + _MODE_NOTE.get(EXEC_PATH, "")
        + "\n\n可用技能：\n"
        + render_catalog(skills.values()),
        state_schema=state_schema,
        # 待答时不发言 —— 见 `_StopWhenPending`
        middleware=[_StopWhenPending()],
        name="main_agent",
    )
