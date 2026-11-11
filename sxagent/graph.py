"""主 agent 一次回合的图：feed → prepare → agent → finalize。

`build()` 把四个节点连成一条直线；`State` / `AgentState` 两个 schema 和三个纯函数
（`_feed_waiting` / `render_relay` / `next_pending`）都在这里。

    feed ─────► prepare ─────► agent ─────► finalize
    投递原话     注入台账       主 agent 循环   兜底补问
    （跑执行器
      60–120 秒）

## 不变量

1. 平台只给事实，不替主 agent 做决定。"算不算在回答""要不要重跑"是理解问题，判错了比多问
   一遍更糟。平台硬保证的只有待答不丢。
2. 断言和报告都读 `actions`，不读 `log`。`log` 是渲染出来的文本，改一句文案，读它的地方就
   静默失配。
3. 子图 schema 里不能放带 `operator.add` reducer 的字段：子图会原样回吐，每轮翻一倍。
4. `answer_to` / `fed_asking` 只在这一轮有意义，`node_feed` 只有一条返回路径。
5. 投给谁由通道标（CLI 提示符 / case YAML），没有标识就一条都不投。
6. "取哪句""取哪条"各只有一处定义（`asked_text` / `next_pending`），别处不许另写。"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

if TYPE_CHECKING:  # 只给标注用 —— 运行时不 import
    from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from sxagent import stream, trace
from sxagent.agent import build_agent
from sxagent.agent import dispatch  # 派发编排（两条路共用）
from sxagent.handoff.briefing import Task, asked_text
from sxagent.handoff import render_hop  # 唯一那条渲染路（见 handoff/__init__）
from sxagent.handoff.ledger import render_ledger
# 本模块不直接调执行器：走 `agent.dispatch`，那边用属性访问 `executor.run_task`（晚绑定）。
from sxagent.registry import registry  # 技能注册表（唯一持有者）


def _merge_tasks(old: list[Task] | None, new: list[Task] | None) -> list[Task]:
    """按 id 合并 Task，`tasks` 的 reducer。

    必须用 reducer：模型会并行发多个 `delegate`，每个都读到同一份旧 state，直接覆盖只会
    剩最后一个。
    """
    merged: dict[str, dict] = {t["id"]: t for t in (old or [])}
    for t in new or []:
        merged[t["id"]] = t
    return list(merged.values())


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    #: 只记"主 agent 决定委派出去的工作"；不委派就是空
    tasks: NotRequired[Annotated[list[Task], _merge_tasks]]
    #: 平台这一轮干了什么 —— 给人看的渲染记录。不许拿它做判据，改一句文案就静默失配。
    log: NotRequired[Annotated[list[str], operator.add]]
    #: 平台动作的结构化事实（谁做的 / 做了什么 / 涉及哪些 Task / 多少字）。断言与报告读它。
    #: 不许进 `AgentState` —— 它的 reducer 是 `operator.add`，子图回吐每轮翻一倍。
    actions: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    #: 这条用户消息在回答哪个待答任务，由通道填。有它 → 只投给那一个；没有 → 一条都不投。
    #: 用后即清（写回 `""`）：不清会跟 checkpointer 活到下一轮。也不许进 `AgentState`。
    answer_to: NotRequired[str]
    #: 这一轮投递之后，执行器又问了一个（见 `agent._StopWhenPending`）。两个 schema 都要声明：
    #: `State` 少了它 LangGraph 会把键悄悄丢掉，`AgentState` 少了它子图里的中间件读不到。
    fed_asking: NotRequired[bool]


class AgentState(TypedDict):
    """主 agent 子图的状态，故意不含 `log`。

    子图会把声明过的字段原样回吐给父图，而父图 `log` 的 reducer 是 `operator.add`，父图
    自己的 log 会被再加一遍，每轮翻一倍。`tasks` 不受影响（按 id 合并，幂等）。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    tasks: NotRequired[Annotated[list[Task], _merge_tasks]]
    #: 这一轮投递之后，执行器又问了一个 —— `agent._StopWhenPending` 要读它。「本批回执」那种
    #: 待答扫工具回执就能认，「投递后又问」这轮手里没东西，只能由平台把事实写在这儿。
    fed_asking: NotRequired[bool]


# ────────────────── feed：投递用户原话（会跑执行器）／prepare：注入台账 ──────────────────


def _last_human(state: State) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage) and str(m.content).strip():
            return str(m.content)
    return ""


async def _deliver(state: State, t: dict, skill: Any, user: str) -> tuple[dict, list[str], list[dict]]:
    """投一条待答。返回投递后的 Task、一行日志，以及被取代的旧 Task。

    历次答复累积在 `Task.qa` 上：执行器每次都是全新的 `run_task`，只带最新一条的话，用户
    答第二个就把第一个冲掉了。
    """
    rec = {**t, "qa": [*(t.get("qa") or []),
                       {"by": "用户", "q": t.get("asked") or "", "a": user}]}
    trace.section(f"feed · 投递用户原话 → {t['id']} · {t['skill']}", kind="feed")
    trace.block("投过去的原话", user)
    trace.note(f"累计问答 {len(rec['qa'])} 对（历次一起带上，否则执行器会重问）")
    # 谁答复了谁的什么问题：答复的是用户，回答的是执行器问的那句，平台只是转达。
    trace.handoff("答复投递", "用户", "执行器", user,
                  structured=f"平台转达｜在回答它问的那句：{asked_text(t)}"
                             f"｜Task.qa 追加 1 对（累计 {len(rec['qa'])} 对）")
    # refs 要一起带上，否则执行器续跑时看不见前序产物。编排走 `dispatch`（两条路共用）：
    # run_task → 落账 → run_by → 试答重跑 → settle → 取代哑待答，顺序写两遍一定会漂。
    # trigger="feed" = 这轮是平台起的，没有回执，全文不在主 agent 上下文里。
    outcome, answer, mooted = await dispatch(
        skill, rec, state.get("tasks") or [],
        trigger="feed", messages=state.get("messages") or [],
    )
    logs: list[str] = []
    if answer:
        logs.append(f"feed: 🔁 {t['id']} 仍缺信息 → 主 agent 试答（{len(answer)} 字）后重跑")
    # 状态迁移走 `settle()`（唯一一处，在 `dispatch` 里）；三条分支只决定日志怎么说。
    if outcome.status == "completed":
        # 完成了 → 同技能更早的待答成了哑问题，由 `dispatch` 算好带出来（快照没延长就不碰）。
        logs.append(
            f"feed: ✅ {t['id']}（{t['skill']}）答复后**完成**"
            f"　产物={outcome.artifacts or '（在回复里）'}"
        )
    elif outcome.status == "waiting_for_input":
        logs.append(f"feed: ⏳ {t['id']}（{t['skill']}）答复后**还是缺信息** → 仍待答")
    else:
        logs.append(f"feed: ❌ {t['id']}（{t['skill']}）答复后仍未完成 → failed")
    return rec, logs, mooted


async def _feed_waiting(state: State) -> tuple[list[dict], list[str]]:
    """把用户这条消息的原话投给它标明的那一个待答任务（`answer_to`）。

    不做任何判断——平台不替用户判断他的话算不算回答，那是语义判断，判错比多问一遍更糟。
    它只挑目标、逐条投。这是投递这条路上唯一的解除路径。
    """
    user = _last_human(state)
    all_waiting = [
        t for t in (state.get("tasks") or []) if t.get("stage") == "waiting_for_input"
    ]
    # 定向投递：只投 `answer_to` 指的那一条，标识由通道填、平台不猜。
    target = str(state.get("answer_to") or "").strip()
    waiting = [t for t in all_waiting if t.get("id") == target] if target else []
    if not user or not waiting:
        if target and not waiting:
            trace.note(
                f"⚠️ `answer_to={target}` 指的那条**不在待答**（已被取代 / 已完成 / 不存在）"
                f" → 不投递（当前待答 {len(all_waiting)} 条）"
            )
        return [], []

    out: list[dict] = []
    logs = [f"feed: 定向投递 → {target}（`answer_to` 指的那条；其余待答不动）"]
    for t in waiting:
        skill = registry.get(t["skill"])
        if skill is None:
            logs.append(f"feed: ⚠️ {t['id']} 的技能「{t['skill']}」不存在，跳过")
            continue
        rec, more, mooted = await _deliver(state, t, skill, user)
        logs += more
        out.extend(mooted)
        out.append(rec)
    return out, logs


async def node_feed(state: State) -> dict:
    """① 把用户原话投给它标明的那一个待答任务，见 `_feed_waiting`。

    只有一条返回路径：`answer_to` / `fed_asking` 是本轮信号，多一条路就多一处可漏。
    """
    updated, logs = await _feed_waiting(state)

    # 投递之后执行器又问了吗 —— 判据是结构事实：投递过的那条落成了 `waiting_for_input`。
    fed_asking = any(t.get("stage") == "waiting_for_input" for t in updated)

    fed = [t["id"] for t in updated
           if t.get("run_by") == "feed" and t.get("stage") != "superseded"]
    actions = (
        [{"by": "platform.feed", "did": "feed", "tasks": fed,
          "chars": len(_last_human(state))}]
        if fed else []
    )
    return {
        "tasks": updated,           # reducer 按 id 合并
        "log": logs,                # `feed: …` —— reducer 是 operator.add
        "actions": actions,
        # 本轮信号必须在这唯一一条返回路径上写回
        "answer_to": "",
        "fed_asking": fed_asking,
    }


async def node_prepare(state: State) -> dict:
    """② 把任务台账注入主 agent 上下文。只给事实，不替它做决定。

    投递已在 `feed` 里做完，这里读到的 `state["tasks"]` 已是最新。不注入的话，主 agent
    只看得到对话历史里那几条工具返回，不知道已经派过什么。
    """
    tasks = state.get("tasks") or []
    if not tasks:
        # 闲聊轮次：没有台账可注入，一个字都不加。
        return {}

    # 台账怎么渲染在 `sxagent/handoff/ledger.py`（schema 与模板放一起）
    waiting = [t for t in tasks if t.get("stage") == "waiting_for_input"]
    text = render_ledger(tasks)
    # 注入什么就记什么：这一行是平台每轮塞给主 agent 的东西，复盘时最该看到的就是它。
    trace.handoff("台账", "平台", "主 agent", text,
                  structured=f"Task × {len(tasks)} → render_ledger", full=True)
    return {
        "messages": [SystemMessage(content=text)],
        "log": [f"prepare: 注入台账（{len(tasks)} 项，其中待答 {len(waiting)} 条）"],
    }


# ──────────────────────────── finalize：兜底补问 ────────────────────────────


def render_relay(waiting: list[dict]) -> str:
    """待答转成贴给用户的那段文本。

    贴的是 `Task.asked`（它申报的那句原话），不是 `report`——用户要答的是问题，不是那段
    自述。`asked` 缺失时显式说清，不静默降级。
    """
    def _one(t: Task) -> str:
        who = t.get("skill") or "?"
        # "取哪句、取不到怎么办"只有一处定义（`asked_text`），CLI 的提示符也用它
        text = asked_text(t)
        if "\n" not in text:
            return f"- 「{who}」问：{text}"
        # 续行自己在里面缩 2 格（`render_hop` 会再缩 2 格）；不然多行那段会顶到 `- ` 那一列
        return f"- 「{who}」问：\n" + "\n".join(f"  {ln}" for ln in text.splitlines())

    return render_hop("relay", {"relay": "\n".join(_one(t) for t in waiting)})


async def node_finalize(state: State) -> dict:
    """兜底：待答原样贴进回复，保证问题一定到用户手上。

    保证归代码，礼貌归提示词：主 agent 没照做也只是重复一句，不影响不丢。不靠 LLM 判断
    要不要贴——那层判断两头都会错。
    """
    waiting = [t for t in (state.get("tasks") or []) if t.get("stage") == "waiting_for_input"]
    if not waiting:
        return {}

    block = render_relay(waiting)
    # 只发内容：`finalize: ` 前缀属于日志文件格式，归记录器；"谁 → 谁"展示层已写着。
    trace.note(f"透传 {len(waiting)} 条待答（平台直接贴，不判断主 agent 问没问）",
               kind="relay")
    trace.handoff("待答透传", "平台", "用户", block,
                  structured=f"Task.asked × {len(waiting)} → AIMessage(injected_by=platform.finalize)",
                  full=True)
    return {
    # 打 injected_by 标记，挂 `additional_kwargs`：不打的话记录层会把这条标成主 agent
    # 说的，而不是去看"内容像不像执行器的话"。
        "messages": [AIMessage(content=block, additional_kwargs={"injected_by": "platform.finalize"})],
        "log": [f"finalize: 透传 {len(waiting)} 条待答（原样贴执行器的话，无条件）"],
        # 同一刻记下结构化事实（断言读它，不读上面那句人话）
        "actions": [{
            "by": "platform.finalize", "did": "relay",
            "tasks": [t["id"] for t in waiting], "chars": len(block),
        }],
    }


# ──────────────────────────── 待答：一次摆一个 ────────────────────────────


def next_pending(state: State, skill: str = "") -> dict | None:
    """当前该摆到用户面前的那一个待答，返回那条 Task（`None` = 没有）。

    一次只摆一个，通道决定归属。`skill` 给跑批用（并行派发时"第一个"不确定），CLI 不点名。
    """
    for t in state.get("tasks") or []:
        if t.get("stage") == "waiting_for_input" and (not skill or t.get("skill") == skill):
            return t
    return None


# ──────────────────────────── 装配 ────────────────────────────


def _make_node_agent(main_agent):
    """造主 agent 那个节点，让它每走一步都露出来。

    用 `stream` 不用 `invoke`：后者要整个循环跑完才返回，那一段全程静默。行为不变，返回
    的仍是最终 state。是工厂不是节点，`main_agent` 要靠闭包带进去。
    """
    async def node_agent(state: State, config: RunnableConfig) -> dict:
        out: dict = {}
        prev_n = len(state.get("messages") or [])
        logs: list[str] = []
        step = 0
        # `stream_agent` 顺带把每个字也写进日志（攒到换行写一行），那几十秒不再黑屏。
        async for chunk in stream.stream_agent(main_agent, state, config, "主 agent",
                                              role=trace.AGENT):
            out = chunk
            msgs = chunk.get("messages") or []
            new = msgs[prev_n:]
            if not new:
                continue  # 第一个 chunk 是输入 state 本身，没有新消息
            prev_n = len(msgs)
            step += 1
            trace.dump_trace(f"主 agent 第 {step} 步", new, role=trace.AGENT)
            # 同一批调用也进 state 的 log —— 逐轮审计文件读的是它，只写 trace 就得两头对着看
            for name, args in trace.tool_calls_of(new):
                line = f"[工具] 主 agent → {name}（{args}）"
                trace.note(line)
                logs.append(line)
        if not out:  # 理论上不会（stream 至少吐一次最终 state）—— 兜一下，别把状态弄丢
    # 这条兜底必须说话：走到这里等于这一趟全程没有流式日志，正是要治的黑屏。
            trace.note(
                "⚠️ 流式**一步都没吐** → 退回 `invoke` 兜底：**这一趟没有流式日志**，"
                "跑完才回来。正常不该发生 —— 见到这条要查 `stream` 为什么是空的。",
                kind="stream_empty",
            )
            return await main_agent.ainvoke(state, config)
        return {**out, "log": logs} if logs else out

    return node_agent


def build() -> CompiledStateGraph:
    """把父图拼起来 —— 四个节点、一条直线，`InMemorySaver` 做 checkpointer。"""
    g = StateGraph(State)
    # 顺序由边保证：先投递（里面同步跑执行器），后注入台账
    g.add_node("feed", node_feed)
    g.add_node("prepare", node_prepare)
    # 主 agent：完整的 agent（工具 + 循环）
    g.add_node("agent", _make_node_agent(build_agent(AgentState)))
    g.add_node("finalize", node_finalize)
    g.add_edge(START, "feed")
    g.add_edge("feed", "prepare")
    g.add_edge("prepare", "agent")
    g.add_edge("agent", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())
