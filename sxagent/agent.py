"""主 agent —— **一个完整的 agent**（有自己的工具与循环），不是纯路由器。

## 这个文件是什么

**三种出路由它自己决定**：

  1. 直接答              —— 闲聊、追问、澄清（**不建 Task**）
  2. 自己产出短内容       —— write_artifact（**不建 Task**）
  3. `delegate` 委派      —— 需要专业技能时（**建 Task**）

**`delegate` 内部由平台保证四件事**（LLM 跳不过）：

    ⓪ **取代旧的**：同 skill 下**已经失败的**旧 Task → 标记 `superseded`（免得台账挂过期项）。
       ⚠️ **`waiting_for_input` 一个字不碰** —— 那是用户还没答过的问题，取代它＝**静默弄丢**
    ① **执行器干活**（`run_task`，隔离上下文；内部含"出口限制"，最多 react 3 轮）
    ② **先让主 agent 试答**：用它自己的完整上下文；答得上就不打扰用户
    ③ **登记待答**：答不上才 `stage = waiting_for_input`，交给主 agent 去问用户

**Task 只在决定委派时才建。** 不委派就没有 Task。

四块拼起来：system prompt（`_MAIN_PROMPT`）／试答（`_ask_main`）／派发（`delegate`
＋ 两条路共用的 `dispatch`）／装配（`EXEC_PATH` 选工具集 ＋ `_StopWhenPending`）。

## 设计原则（不变量）

1. **不许反解析**：回执对象搭 `ToolMessage.additional_kwargs` 走（实测**不会漏给
   模型**），记录 / 审计层**读对象**；`_StopWhenPending` 读的是 `handoff.kind` ——
   **不读文本、不猜语义**。
2. **交接必须硬拼**：用户原话由平台搬（`_user_turns`），"它要什么"取 `ask_user` 的参数原话
   （`asked`）—— 转述不只丢、还会加，所以平台不转述。
3. **状态迁移只有一处**：`settle()`（在 `dispatch` 里）。手写 `rec.update(stage=…)` 是
   **无条件覆盖** —— 会把"这一轮没产出就不抹掉上一轮产物"的保护当场抵消。
4. **一次决策 = 一条记录**：前提 → 问题 → 结果 → 下一步；同一个事实不写两遍。

## 关键取舍

**① 待答那一刻是决策点，不是对话轮次** —— 正解不是"拦住它的输出"，是**让那一拍不发生**：
中间件在模型调用之前跳走，那次调用根本没发生（`_StopWhenPending`）。

**② 两条路共用一段编排**（`dispatch`）：核心（试答 / 落账 / 取代）早就统一了，但**它们之间的
顺序**原来两条路各写一遍 —— 于是"重入前要先落账"只在一条路上成立。顺序写两遍就一定会漂。

**③ 试答只做一步、不决定出口**：completed / 仍待答 / failed 由调用方按最终 outcome 分派
（三个出口的措辞各不相同，混进来就又长出一个上帝函数）。

**④ 模式切换靠"拿掉工具"，不靠提示词求它**：`inline` 是**全硬**的（`delegate` 工具根本不存在，
它调不了）；`delegate` 档是**半硬**的（拿掉原料，但**挡不住它在回复里直接写**）—— 所以那一档
还要配 `_MODE_NOTE`，跑批时用数量断言盯着。

## 遗留问题

- `_user_turns` 的 `after` 现在是**次级线索**（取到的是主 agent 自己那条回复，可能压根没提问题）
  —— 「答不能脱离问」靠【问答经过】兜（`qa[].q`）。⚠️ **待实测确认够不够**：不够的话，正解是让
  `_user_turns` 在"这一轮是答复"时直接取 `Task.asked`（而不是从 `messages` 里找前一条 AI
  消息）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # 只给标注用 —— 运行时不 import
    from langgraph.graph.state import CompiledStateGraph

from sxagent import trace
from sxagent.config import get_model
from sxagent import executor  # ⚠️ 走模块属性，见 delegate 的注释
from sxagent.handoff.ask import build_prompt as ask_prompt  # 试答
from sxagent.handoff.decision import emit_decision  # 决策事件的发射器（归领域包）
from sxagent.handoff.briefing import (  # 委托单
    Task,
    absorb_outcome,
    moot_older_waiting,
    render_briefing,
    settle,
)
from sxagent.handoff.receipt import (  # 回执
    NOTE_BY_FAULT,
    NOTE_FAULT_UNCLASSIFIED,
    build_receipt,
    build_rejection,
    next_note,
    render_receipt,
)
from sxagent.llm import safe_structured
from sxagent import artifacts as artifacts_mod  # 产物领域（唯一入口是它的读 API）
from sxagent.artifacts import (
    artifact_index_lines,
    list_artifacts,
    read_artifact,
    write_artifact,
)
from sxagent.skill_pack import read_skill, read_skill_file  # 技能包：两个读工具（磁盘）
from sxagent.registry import registry  # 技能注册表（唯一持有者）

#: 主 agent 的 system prompt —— 平台约定，**不约束 skill 作者**
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
    "用户说「帮我备课」就**只出教案**；不要说「我顺便把导学单和小测也做了」。\n"
    "**用户没要的，不要直接做掉**——那会浪费他的时间和成本。\n"
    "\n"
    "想提供的话，**在回复末尾提一句**就行（如「需要的话我也可以出导学单和随堂小测」），\n"
    "让用户自己决定要不要。\n"
    "\n"
    "同理，用户说「备课、导学单、小测」就是**三样**，不要多也不要少。\n"
    "\n"
    "## 🔴 前序产物怎么交给下一个任务\n"
    "\n"
    "**传 ID，不要转述正文，也不用先读它。** 需要前面交付的产物当输入时：\n"
    "`delegate(skill_name=\"...\", task=\"...\", artifacts=[\"art_001\"])`\n"
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
    "「文件：art_001 → <路径>」）。\n"
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

#: 这一步**叫什么** —— 业务侧的**唯一一份**（`section` 的标题和决策记录的 `who` 都用它）。
#:
#: 🔴 展示层**直接用 `ev["title"]`**（`chat._on_event`），**不许就地再编一个标题** ——
#: 原来三份各不相同，屏幕上印出两个 `🧭 DECIDE` 块，读的人以为是两次决策。
_ASK_STEP = "试答（LLM）· 平台强制的一步"


class _AgentAnswer(BaseModel):
    can_answer: bool = Field(description="你的上下文里有没有它缺的信息")
    answer: str = Field(default="", description="can_answer=true 时给出补充内容，原话照搬不要改写")


async def _ask_main(messages: list, skill_name: str, task: str, outcome: Any) -> str:
    """**平台强制的一步**：让主 agent 用自己的上下文试着补上缺的信息。

    ⚠️ **必须把"它要什么"给全**（信息交接原则）：那是**执行器自己申报的原话**
    （`ask_user` 参数 ✓ 结构 ✓）—— 不用从它说的话里挑哪句是问题。

    ⚠️ 入参是**消息列表**而不是 `ToolRuntime`：这一步**两条路都要用** —— `delegate` 有
    runtime、`graph._feed_waiting` 只有 state，而这里真正要的只是 `messages`。
    """
    # 🔴 **不截断**（「交接三问」之二）：这是个**独立的结构化
    # 调用**，它**只看得见这个提示词里的东西** —— 信息落在长消息的后半段，它就答不上来、
    # 只能去打扰用户，而"能不打扰用户就不打扰"正是这一步存在的全部意义。
    # 主 agent 的上下文本来就小（隔离生效），整份给**全文**是便宜的。
    # ⚠️ **别退回"最近 N 条"**：这一步问的就是"**你的上下文里**有没有" —— 砍掉前文等于让它
    # 答不上来，那是个**无声**的范围限制，比截断更难发现。
    msgs = messages or []
    prompt = ask_prompt(skill_name, task, outcome, msgs)
    # 交接关键日志：这一步把**主 agent 的全部上下文**又发了一遍（膨胀代价看得见）
    # ⚠️ **这行就是清单**（多少字 ＋ 什么构成）—— 正文默认不逐字记（`SXAGENT_LOG_PROMPT`），
    #    所以这里**不许再写"逐字见下"**：那是句会变成假话的文案（下面那条 note 说在哪取）。
    trace.handoff("试答", "平台", "主 agent", prompt,
                  structured=f"ExecOutcome ＋ 全部 {len(msgs)} 条消息")
    # 调用失败时保守当作「答不上」—— 交给用户，而不是编一个答案
    r = await safe_structured(
        _AgentAnswer, prompt, _AgentAnswer(can_answer=False),
        tag=_ASK_STEP,
        kind="ask_main",
        # ⚠️ 返回**不原样再记一遍**：下面那条决策记录里已经有排好版的「结果」，
        # 再来一条转义 repr 就是同一件事两副面孔（**不要搞两套**）。
        log_response=False,
    )
    # 🔴 **一次决策 = 一条记录**：前提 → 问题 → 结果 → 下一步。
    # 前提的**全文**就是上面那条 PROMPT（主 agent 的整个上下文，不截）——
    # 这里只说清"它看的是什么"，不重复贴一遍。
    ans = r.answer.strip()
    emit_decision(
        _ASK_STEP,
        q="**主 agent 的上下文里**，有没有执行器要的那个信息？（有就不打扰用户）",
        given=[
            # 🔴 **它要什么 = `ask_user` 参数里的原话**（结构 ✓）—— 判定退休后没有别的
            #    来源了（那两样"已做到 / 还缺"本来是判定吐的）。
            # ⚠️ **原话在这儿给全**（便宜）：不给全的话，"试答答不上"这条结论就没有
            #    指涉对象了。**不重贴的只有结尾那行 run 账**（它是摘要）。
            f"**它要的那句**（执行器 → 平台，`ask_user` 原话，{len(outcome.asked)} 字）："
            f"{outcome.asked}",
            f"它最后说的那段（{len(outcome.report)} 字，可能有上下文、也可能只是过场）",
            # ⚠️ 指向**清单 ＋ 开关**，不指"上面那条 PROMPT" —— 正文默认不逐字记，
            #    那句会变成假话。逐条的内容各自在进上下文那一刻记过（见 `llm` 那条 note）。
            f"主 agent 的全部 {len(msgs)} 条消息 —— **全文、不截**"
            f"（逐条已各自记录；拼好的整份置 `SXAGENT_LOG_PROMPT=1` 可看）",
        ],
        result=[
            f"can_answer = **{r.can_answer}**",
            f"补充：{ans or '（无）'}",
        ],
        # 🔴 **两个出口都写清「谁 → 谁」** —— 原来只写"答不上 → 登记待答"，
        #    **谁答不上、转给谁**都得靠读的人自己补。
        nxt=(
            "**主 agent 答得上** → 成对记进 `Task.qa`（`by=主 agent`）→ 带回去重跑执行器，"
            "**不打扰用户**"
            if r.can_answer
            else "**主 agent 答不上 → 转给用户**：登记待答，由**平台**把执行器原话"
                 "**透传**给用户"
        ),
    )
    return ans if r.can_answer else ""


async def try_answer_then_rerun(
    rec: Task, skill: Any, outcome: Any, *, messages: list
) -> tuple[Any, str]:
    """执行器说「缺信息」→ **平台强制主 agent 先用自己上下文试答** → 答得上就重跑一遍。

    🔴 **两条路共用**：首次派发（`delegate`）和**投递重跑**（`graph._feed_waiting`）走
    **同一套**处置 —— 原来投递那条路直接又登记待答，用户刚答完一轮就**又被问一遍**。

    返回 `(最终 outcome, 主 agent 的试答)`：
      - 试答为空 → 原样返回（调用方去登记待答）
      - 试答非空 → **成对**记进 `qa`（`q` 是它问的那句原话）→ 重跑 → 返回新 outcome

    ⚠️ **只做这一步，不决定出口** —— completed / 仍待答 / failed 由调用方按最终 outcome
    分派（三个出口的措辞各不相同，混进来就又长出一个上帝函数）。
    """
    answer = await _ask_main(messages, skill.name, rec.get("task") or "", outcome)
    if not answer:
        return outcome, ""
    rec["agent_tried"] = True
    # 🔴 **成对记**（`by="主 agent"`）：只记答案的话，下一次重入时 `report` 已经被
    # 新问题覆盖，这条答案就没了指涉 —— 同 `qa` 那段注释。
    # 🔴 **「问」取 `outcome.asked`（`ask_user` 的参数原话），不取 `report`**：
    # `report` 是它最后说的**那段话**（混着过场），拿它当「问」配对就失去指涉。
    rec["qa"] = [*(rec.get("qa") or []),
                 {"by": "主 agent", "q": outcome.asked, "a": answer}]
    # 试答后重跑也是**重入**：走同一份交接书（带上刚才那一轮的进度 / 它问过什么）
    o2 = await executor.run_task(skill, render_briefing(rec), rec.get("user_turns"))
    return o2, answer


async def dispatch(skill: Any, rec: Task, tasks: list[Task], *, trigger: str,
                   messages: list) -> tuple[Any, str, list[Task]]:
    """**派发编排** —— 两条路（`delegate` / `graph._feed_waiting`）共用的那一段。

    🔴 **为什么抽出来**：核心早就统一了（`try_answer_then_rerun` / `settle` /
    `moot_older_waiting` 三样共用），但**它们之间的顺序**原来是两条路各写一遍 ——
    于是"重入前要先落账"这条**只在一条路上成立**（投递那条路的试答重跑拿到的是
    **上一轮**的 `report`）。顺序这种东西，**写两遍就一定会漂**；收进一处，漂不动了。

    顺序（每一步都有理由，别调）：
      ① `run_task` —— 执行器干活（独立上下文，用完即弃）
      ② `absorb_outcome` —— **重入前先落账**：下面 `try_answer_then_rerun` 里的重跑
         就是一次重入，它的交接书要看到**这一轮**的 report / asked
      ③ `run_by = trigger` —— 台账据此说清"全文在哪"（回执 / 没有回执）
      ④ 它要信息 → `try_answer_then_rerun`（主 agent 先试答，答得上就重跑）
      ⑤ `settle` —— **状态迁移唯一入口**（`absorb` 幂等，这里再落一次换"不可能漏写"）
      ⑥ 完成了 → `moot_older_waiting`（同技能更早的哑待答**显式取代**）

    ⚠️ **只做编排，不做出口措辞** —— 回执（`delegate`）/ 日志（`_feed_waiting`）
    各渲染各的（"视图可以有很多个，事实源只能有一个"）；这里返回**事实**：
    `(最终 outcome, 主 agent 的试答, 被取代的 Task)`。
    """
    outcome = await executor.run_task(skill, render_briefing(rec), rec.get("user_turns"))
    absorb_outcome(rec, outcome)
    rec["run_by"] = trigger
    answer = ""
    if outcome.status == "waiting_for_input":
        outcome, answer = await try_answer_then_rerun(rec, skill, outcome, messages=messages)
    settle(rec, outcome)
    mooted = moot_older_waiting(rec, tasks) if outcome.status == "completed" else []
    return outcome, answer, mooted


def _cmd(
    runtime: ToolRuntime, recs: list[dict] | None, msg: str, *, handoff: dict | None = None
) -> Command:
    """返回**这几条** Task —— 合并交给 state 的 reducer。

    并行调用时各写各的，reducer 按 id 合并，不会互相覆盖。

    🔴 `handoff` = **回执对象**，挂在 ToolMessage 的 `additional_kwargs` 上**搭同一趟车**：
    模型只看 `content`（渲染出来的文本），**记录/审计层读这个对象** ——
    实测 `additional_kwargs` **不会漏给模型**（`_convert_message_to_dict` 只取
    content / role / tool_call_id）。这样记录层永远不需要**从文本反推结构**
    （见 `sxagent/handoff/receipt.py` 的 docstring）。
    """
    update: dict[str, Any] = {
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
    """**派发被拒**（不建 Task）→ 回执对象 ＋ 渲染 —— 和正常回执走**同一条**路。"""
    r = build_rejection(kind, detail, available)
    return _cmd(runtime, None, render_receipt(r), handoff=r)


def _supersede_old(runtime: ToolRuntime, skill_name: str) -> list[dict]:
    """把**同 skill 下已经死掉的**旧 Task 标记为 `superseded`。

    ⚠️ **只取代 `failed` 的。`waiting_for_input` 一律不碰** —— 待答由平台负责投递和透传，
    **重派不是解除待答的路径**：碰它只会**静默抹掉一个用户从没回答过的问题**。

    ⚠️ 解除待答的**另一条**路是 `moot_older_waiting`（新任务完成 ＋ 用户已另起话题）——
    **建的时候**（这里）`waiting` 仍然一个字不碰。

    - `completed` 的是**历史记录**，不能被抹掉（用户可能还要看产物）
    - `running` 的不会出现在旧 state 里（并行调用各读各的旧快照）

    返回的是**完整的 Task 字典**（不是 patch）—— reducer 按 id 整体覆盖。
    """
    out: list[dict] = []
    for t in runtime.state.get("tasks") or []:
        if t.get("skill") == skill_name and t.get("stage") == "failed":
            out.append({**t, "stage": "superseded"})
    return out


# ──────────────────────────── 用户原话（平台硬搬） ────────────────────────────


def _user_turns(messages: list) -> list[dict]:
    """把对话历史里的**用户原话**抽出来，按顺序编号 —— 平台硬搬，不经任何转述。

    🔴 **为什么要搬**：首次派发时执行器看到的"任务"**100% 是主 agent 的转述**，而转述
    **不只丢、还会加** —— 加了会变成一份"基于错误前提的成品"，而用户不知道。

    ⚠️ **只认 `HumanMessage`**：主图里注入的台账是 `SystemMessage`、透传是 `AIMessage`，
    所以"HumanMessage = 用户本人说的"在**主图**里成立（执行器自己的图里不成立，别复用）。

    ⚠️ 这里存的是**全文**（可能很长，用户会粘长文）—— 交接书里只渲染**开头几个字**，
    全文由执行器按需用 `read_user_turn(n)` 取（渐进式披露，见 `render_briefing` 那段）。

    🔴 **每轮还要带上"它在回答的那一句"（`after`）** —— 光一个「A」，执行器既解读不了、
    也核对不了主 agent 的转述。

    ⚠️ `after` 取**紧邻的前一条 AI 消息**（纯位置关系，平台搬，不判断语义）；**它是次级
    线索**（取到的是主 agent 自己的回复）—— 见模块头「遗留问题」。
    """
    out: list[dict] = []
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
            # 不该再挂同一条 AI 消息（挂了就是**错的**上下文，比没有更糟）。
            last_ai = ""
    return out


# ──────────────────────────── 委派 ────────────────────────────


def _new_task(skill_name: str, task: str, tid: str, refs: list[str],
              turns: list[dict]) -> Task:
    """建**委托单** —— 这次委派的全部已知（schema 见 `handoff.briefing.Task`）。

    ⚠️ `total=False` 是**如实**：注入的 Task 可能只有部分字段。
    """
    return {
        "id": tid, "skill": skill_name, "task": task,
        "stage": "running", "artifacts": [], "report": "", "agent_tried": False,
        # 它申报要的那条信息（`ask_user` 的原话）—— 新建的 Task 还没申报；
        # 每次 `settle` 由 `absorb_outcome` **覆盖写**（见 `Task.asked`）。
        "asked": "",
        # 平台故障的原因（空 = 没故障）—— 新建的 Task 当然还没故障；
        # 每次 `settle` 由 `absorb_outcome` **覆盖写**（见 `Task.fault`）。
        "fault": "",
        # 故障的"类"（`FaultKind`）—— 与 `fault` 同生共死，同样覆盖写。
        "fault_kind": "",
        # ⚠️ `artifacts` 是**产出**（产物 ID 列表），`refs` 是**输入引用** —— 两个方向，别混。
        # 待答投递（`graph._feed_waiting`）要按 refs 原样重建可读产物清单。
        "refs": refs,
        # 🔴 **历次问答对，累积在 Task 上**：`[{"by": "用户"|"主 agent", "q": 它问的原话, "a": 答复}]`
        # —— 执行器用完即弃，每次重入都要重放全部已知；`report` / `asked` 都是**覆盖**写的，
        # 答案累积、问题覆盖的话，第二次重入之后旧答案就没有指涉了。一个列表 ＋ 一个 `by`
        # 字段就够（分 `answers` / `agent_answer` 两个字段是**按来源分叉**）。
        #
        # 🔴 **`q` 取 `asked`（`ask_user` 的参数原话），不取 `report`**：平台**不解析散文**，
        # 所以"哪句才是问题"它答不了 —— 正解不是去抽，是**取那个本来就是问题的字段**。
        "qa": [],
        #: 这轮是谁起的（台账据此说清"全文在哪"）—— `delegate` 这一趟的全文在**回执**里，
        #: 而投递重跑那一趟没有回执（见 `briefing.Task.run_by`）。⚠️ 试答后重跑仍在
        #: **同一次** delegate 调用里，回执也还是这一份 —— 所以这里恒为 "delegate"。
        "run_by": "delegate",
        # 用户原话**快照**（不是渲染交接书时现取），两个理由：
        # ① 答复投递那条路（`graph._feed_waiting`）也调 `render_briefing`，快照在 Task 上它自动跟着走
        # ② 与 `qa` **天然去重** —— 快照是"派发**之前**用户说的"，
        #    `qa` 是"派发**之后**用户答的"，时间上不重叠
        "user_turns": turns,
    }


def _out(runtime: ToolRuntime, rec: Task, superseded: list[Task], outcome: Any, *,
         agent_answer: str = "", note: str = "", mooted: list[Task] | None = None) -> Command:
    """统一的返回口 —— **总是带上被取代的旧 Task**。

    🔴 它做两件事，**一次做完**：构造**回执对象**、渲染成文本 ——
    文本进 `content`（模型只看这个），对象挂 `additional_kwargs`（模型看不到，
    记录层读它）。见 `sxagent/handoff/receipt.py` 的 docstring。

    ⚠️ `mooted` 由 `dispatch` 算好传进来（**取代的判据只有一处**，见它的 docstring）。
    """
    r = build_receipt(outcome, agent_answer=agent_answer, note=note)
    text = render_receipt(r)
    # 交接关键日志（复盘用）：这一跳交了多少、有没有结构化对象 —— 见 `trace.handoff`。
    # 正文**不在这里 dump**：主 agent 的逐步骤 dump 里已经有它（`graph.node_agent`）。
    trace.handoff("回执", "执行器", "主 agent", text,
                  structured=f"Receipt(kind={r['kind']}，{len(r['sections'])} 段)")
    return _cmd(runtime, [rec, *superseded, *(mooted or [])], text, handoff=r)


def _reply(runtime: ToolRuntime, rec: Task, superseded: list[Task], tid: str,
           skill: Any, outcome: Any, *, answer: str = "",
           mooted: list[Task] | None = None) -> Command:
    """**回话**：把这一趟的结果说回主 agent —— 按**三态枚举**分派，穷举，没有第四种。

    ⚠️ 状态迁移一律走 `settle()`（**唯一一处**，在 `dispatch` 里）—— 手写
    `rec.update(stage=…)` 是**无条件覆盖**，会把"这一轮没产出就不抹掉上一轮产物"
    的保护当场抵消。
    """
    # ① **完成了** —— 🔴 **平台不下"完成了"的结论**：三态是**结构算的**
    #    （交没交产物 / 报没报要信息）—— "**够不够**"由主 agent 判：
    #    回执把交付账摆给它，它自己看 ✓。
    # ⚠️ **文案从 `receipt.next_note` 取**（不就地写）：一处定义，所有渲染方从那儿取。
    # ⚠️ `agent_answer`：试答补上的那次要把"你补的信息"一起带上 —— `dispatch` 收口后
    #    outcome 已经是**最终**三态，这里一并带（原来在 waiting 分支里分过一次）。
    if outcome.status == "completed":
        return _out(runtime, rec, superseded, outcome, agent_answer=answer,
                    note=next_note("completed", skill=skill.name), mooted=mooted)

    # ② **在要信息** —— **试答那一步已经在 `dispatch` 里走过了**（两条路共用同一个函数）。
    #    走到这里 = 试答答不上（或答了仍缺）→ 登记待答：执行器问的那句，**由平台原样转给用户**。
    # ⚠️ 「**主 agent 答不上**」这半句**不写在这儿**：它就在上面那条 `🧭 DECIDE` 的
    #    【结果】里，而展示层上屏时**已经加了**「分支：主 agent 答不上 → 」前缀
    #    （`chat._on_event`）—— 再写一遍就是同一个意思印两回。
    if outcome.status == "waiting_for_input":
        trace.note(
            f"delegate · {tid}（{skill.name}）**登记待答** —— 执行器问的那句"
            f"由平台原样转给用户（不由主 agent 代问）",
            kind="waiting",
        )
        return _out(runtime, rec, superseded, outcome, note=next_note("waiting"))

    # ③ 只剩 `incomplete`（含"试答补了信息之后反而变成 incomplete"那条）——
    #    推了 N 轮，**既没交产物、也没申报要信息**。
    #    按「情况 A：执行失败」→ **交给主 agent 决定**，而不是拿去烦用户。
    #    ⚠️ 平台**不猜**它为什么没交（不读正文 ✓）：可能它真干不了，也可能它把答案
    #    **写在了回复里**（正文交付是反模式，平台不认）—— 两种都由主 agent 看回执决定 ✓。
    # 🔴 **平台故障 ≠ 执行器干不了**：故障时那句"推了 N 轮"是假话（一轮都没跑完），
    #    而主 agent 该做的事也完全不同 —— 换说法重派 vs **原样重派**。
    if outcome.fault:
        # ⚠️ **三档故障三种说法**：`fault_kind` 一起带进【下一步】，措辞按它选 ——
        #    从对象取，不在这里现编。
        # 🔴 **表住 `handoff.receipt`，不在这儿**：散在 `agent.py` 里的话，
        #    覆盖守卫够不着它 —— 那正是"四处里只有一处有守卫"的成因。
        trace.note(f"delegate · {tid} "
                   + NOTE_BY_FAULT.get(outcome.fault_kind, NOTE_FAULT_UNCLASSIFIED),
                   kind="delegate_failed")
        return _out(runtime, rec, superseded, outcome,
                    note=next_note("fault", fault_kind=outcome.fault_kind))
    trace.note(f"delegate · {tid} status=incomplete（推了 {outcome.rounds} 轮）→ failed",
               kind="delegate_failed")
    return _out(runtime, rec, superseded, outcome, note=next_note("failed"))


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
    后三步各在一个私有件里（`_new_task` / `dispatch` / `_reply`）——
    原来是一个 159 行的函数，把"校验输入""建单""委派""按三态回话"缠在一起。
    """
    # ① **准入** —— 技能不存在 / 引用了不存在的产物，**当场拒**（编出来的 ID 放行的话，
    #    执行器读不到会自己找补，错误被埋进产物里）。
    skill = registry.get(skill_name)
    if skill is None:
        return _reject(runtime, "rejected_skill", f"❌ 没有技能「{skill_name}」",
                       "、".join(registry.names()))
    refs = [a for a in (artifacts or []) if a]
    unknown = [a for a in refs if not artifacts_mod.has(a)]
    if unknown:
        # ⚠️ 用**换行**拼，不用"；" —— 清单每项可能两行（梗概 ＋ 路径），
        #    分号拼会把它们挤成一大坨（`body_lines` 那边会整块缩进，本来就是列表）。
        have = "\n".join(artifact_index_lines()) or "（本会话还没有产物）"
        return _reject(runtime, "rejected_artifact",
                       f"❌ 产物 {'、'.join(unknown)} 不存在", have)

    # ② **立案** —— 这次委派记成一条 Task。
    # 用户原话**平台硬搬**（快照进 Task）—— 见 `_user_turns` 的 docstring。
    # ⚠️ 平台**不挑**哪几轮给全文（原来由主 agent 点名，那个字段已删，见
    #    `briefing.turn_lines`）：全给索引、全文按需取。
    turns = _user_turns(runtime.state.get("messages") or [])
    # id 用 tool_call_id 派生 —— **并行调用时不会撞车**（用序号会撞，实测踩过）。
    # ⚠️ 必须取**尾部**：tool_call_id 形如 `call_abc123...`，前几位所有调用都一样。
    tid = f"t{(runtime.tool_call_id or 'x')[-8:]}"
    # 同 skill 下还没了结的旧 Task → 标记 superseded，免得台账里挂着过期项
    superseded = _supersede_old(runtime, skill_name)
    rec = _new_task(skill_name, task, tid, refs, turns)
    # kind：语义标签，展示层（订阅事件的那层）按它归角色 —— 不靠子串认行
    trace.section(f"delegate · {skill_name} · {tid}", kind="dispatch")
    if superseded:
        trace.note(f"取代了 {len(superseded)} 个同技能的未了结 Task：{[t['id'] for t in superseded]}",
                   kind="supersede")

    # ③ **执行** —— 编排（run_task → 落账 → 试答 → 落账 → 取代）走**两条路共用**的那个函数；
    #    `run_task` 内部已含「出口限制」：既没完成也没提问会被推回去继续。
    outcome, answer, mooted = await dispatch(
        skill, rec, runtime.state.get("tasks") or [],
        trigger="delegate", messages=runtime.state.get("messages") or [],
    )

    # ④ **回话** —— 按三态说回主 agent（穷举，没有第四种）。
    return _reply(runtime, rec, superseded, tid, skill, outcome,
                  answer=answer, mooted=mooted)


# ──────────────────────────── 装配 ────────────────────────────

#: **执行路径开关** —— 主 agent 到底"自己干"还是"派出去"。
#:
#:   - `"auto"`（**默认，产品行为**）：两条路都开，由它自己选
#:   - `"delegate"` **强制委派**：裁掉"自己干"的工具（`write_artifact` /
#:     `read_skill` / `read_skill_file`）→ 它**拿不到原料**，只能派出去
#:   - `"inline"` **强制主 agent 自己干**：**不给 `delegate`** → 工具不存在，它调不了
#:
#: 🔴 **为什么要这个开关**：跑批时想验某条链路，只能**碰运气**碰上一个模型愿意走那条路的
#: case —— 而"走哪条路"是**模型的自由选择**，那条断言必然脆。有了开关，**测哪条就开哪条**。
#:
#: ⚠️ **两档的硬度不一样**，别当成一回事：
#:   - `"inline"` 是**全硬**的 —— `delegate` 工具根本不存在，它调不了。
#:   - `"delegate"` 是**半硬**的 —— 裁掉工具能拿掉它的原料（读不到技能、写不了产物），
#:     但**挡不住它在回复里直接写**。所以这一档还要配一句系统提示（`_MODE_NOTE`），
#:     并且跑批时用数量断言盯着"到底派没派"。
EXEC_PATH = "auto"

#: 三种模式各给哪些工具。⚠️ 改这里时想清楚**拿掉某个工具会不会顺手拿掉一条合法能力**。
_TOOLS: dict[str, list] = {
    "auto": [delegate, write_artifact, read_artifact, list_artifacts, read_skill, read_skill_file],
    "delegate": [delegate, list_artifacts],
    "inline": [write_artifact, read_artifact, list_artifacts, read_skill, read_skill_file],
}

#: 非默认模式下，往系统提示里补一句 —— 工具集变了，得让它知道**为什么**。
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
    """**执行器一申报待答，主 agent 这一拍就不生成**（"待答时复述问题"的收口）。

    🔴 **那一刻是个决策点，不是对话轮次**（要决定的是「结果回给执行器，还是问用户」）——
    ReAct 的默认规则"工具返回了 → 模型再生成一拍"把它当成了普通轮次，主 agent 就"发言"了，
    而它唯一能说的就是复述那个问题。**正解不是"拦住它的输出"，是"让那一拍不发生"。**

    ⚠️ **"拦截 / 禁言 / 收话"是错的框法** —— 中间件在**模型调用之前**就跳走了，那次调用
    根本没发生，不是"生成了再压住"。⚠️ `before_model` 是**公开 API**，框架自带的
    `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` / `PIIMiddleware` 用的就是
    这同一个模式（`@hook_config(can_jump_to=["end"])`）。

    🔴 **触发条件是结构事实，不是模型判断**：读的是回执对象上的 `handoff.kind`
    （`Receipt.kind`）—— `"waiting"` 才停；`completed` / `failed` / `fault` **照常发言**
    （它要报交付、要处置失败）。**不读文本、不猜语义**。

    **谁把问题交到用户手上**：平台。两条路，都不经过 LLM —— CLI 的 `next_pending` 提示符、
    前端的交互卡片；外加 `finalize` 注入的那条 `AIMessage`（转写里那条）。
    """

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """这一轮平台要贴一条新问题出去 → 结束这一轮，模型不会被调用。

        **两种形状都要认**（少认一种就是"拦一半"）：

        **形状一 —— 「本批回执」**（主 agent 自己调的 `delegate`）：
        ⚠️ **要看整批，不能只看最后一条**：模型会**并行调工具** —— 它把 `delegate` 和
        `read_skill` 放在**同一条消息**里时，待答回执**不排在最后**，只看末尾那条会让机制
        **静默失效**。"这一批" = 末尾**连续的那串 `ToolMessage`**（同一条 `AIMessage` 的
        `tool_calls`）。

        **形状二 —— 「投递后又问」**（`state["fed_asking"]`，平台投递后执行器又问了一个）：
        这条路上**根本没有 `delegate` 调用** —— 主 agent 的消息末尾是 `HumanMessage`
        （用户刚答的那句），上面那轮扫描**扫不到任何工具返回**。事实由平台写在
        `AgentState.fed_asking` 上（**本轮信号**，每轮重写）。
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
    def _stop(why: str) -> dict[str, Any]:
        """⚠️ **单独报一行**：读日志的人得看得出"主 agent 这一轮被平台收了话" ——
        不然只会以为它没说话（那是另一回事）。`why` 写清是**哪条路**拦下的（复盘要看）。
        """
        trace.note(f"主 agent **这一轮不发言**（{why}，问题由平台原样贴给用户）"
                   " → `before_model` 跳 end", kind="")
        return {"jump_to": "end"}


def build_agent(state_schema: type) -> CompiledStateGraph:
    """主 agent：一个完整的 agent（工具 + 循环）。

    ⚠️ **工具集由 `EXEC_PATH` 决定**（默认 `"auto"` = 两条路都开）。
    """
    return create_agent(
        get_model(),
        tools=list(_TOOLS[EXEC_PATH]),
        system_prompt=_MAIN_PROMPT
        + _MODE_NOTE.get(EXEC_PATH, "")
        + "\n\n可用技能：\n"
        + registry.catalog(),
        state_schema=state_schema,
        # 🔴 **待答时不发言** —— 见 `_StopWhenPending` 的 docstring。
        middleware=[_StopWhenPending()],
        name="main_agent",
    )
