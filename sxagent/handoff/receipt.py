"""回执（执行器 → 主 agent）：Receipt 的结构与渲染模板，`briefing.py` 的镜像。

平台把执行结果结构化成 `Receipt`（`build_receipt` / `build_rejection`），注入那一刻才渲染成
文本（`render_receipt` → `render_hop`）；同一个对象搭同一趟车进记录层和审计层。

## 不变量

1. 对象和文本是同一份东西的两种形态：文本只是给模型看的视图，事实源是对象。

       ExecOutcome ──build_receipt()──▶ Receipt ──┬─ render_receipt() ──▶ content（给模型看）
                                                  └─ additional_kwargs["handoff"]

2. 不反解析：记录层没有也不许有"把回执文本解析回段"的代码。
3. 交付内容一个字都不截（用户原话："你截断了我怎么看"）。
4. 段名、来源标注、顺序全从词汇表取（`SECTION` / `SOURCE_NOTE` / `HOPS` / `BODY`），本文件
   不写一个字面量。
5. 【下一步】的文案住在这里，不写在调用点。
6. 三类故障在四处各渲染一句：交付行、【下一步】、【平台故障】段、`delegate` 的 trace
   note。四处都按 `FaultKind` 分档，键必须一致，漏一档就是静默走兜底。"""

from __future__ import annotations

from typing import TypedDict

from sxagent.artifacts import DEFAULT_STORE, ArtifactStore
from sxagent.handoff import (
    DELIVERED_DESPITE_FAULT,
    DELIVERY,
    DELIVERY_FAULT_REJECTED,
    DELIVERY_FAULT_TIMEOUT,
    DELIVERY_FAULT_UNCLASSIFIED,
    NUDGED_WITHOUT_DELIVERY,
    Outcome,
    render_hop,
)

# ──────────────────────────── schema：回执上有什么 ────────────────────────────


class Receipt(TypedDict, total=False):
    """回执：一次委派的结果，平台要交给主 agent 的全部内容。

    `sections` 的键就是 `SECTION` 的键，值的顺序由 `HOPS` 定，所以"渲染出来的段"和
    "记录层显示的段"结构上必然一致。
    """

    kind: str  # 出口分类：DELIVERY 的键 + rejected_skill / rejected_artifact
    sections: dict[str, str]  # 段名 key → 该段的内容原文（不含段名、不含来源标注）


# ──────────────────────────── 【下一步】：各出口的处置提示 ────────────────────────────

#: 三个不带变量的出口的【下一步】文案 —— 见 `next_note`。
_NEXT: dict[str, str] = {
    "waiting": (
        "平台已登记为待答，**并会负责把这个问题转达给用户**，"
        "用户的答复也会由平台投回给它 —— 你**不用**重复问，"
        "更不要自己替用户定一个答案。"
    ),
    "failed": (
        "它**推了几轮都没出口**：既没交产物、也没说要什么信息。两种可能，你看回执定：\n"
        "- 它**真的干不了** → 换个说法再 `delegate` 一次，或向用户说明这次没做成；\n"
        "- 它其实**把答案写在了回复里**（上面【执行器说】）→ 平台不认正文交付，"
        "但你可以**原样转达**给用户，或者让它**交付成产物**再来一次。"
    ),
    "fault": (
        "这次是**平台故障**，**不是任务本身的问题** —— 原样再 `delegate` 一次即可"
        "（不用改措辞、不用改参数、不用问用户）。"
    ),
}

#: 三类故障的交付行。`transient` 直接借词汇表那条，别另写一份。
_DELIVER_BY_FAULT: dict[str, str] = {
    "transient": DELIVERY["fault"],
    "rejected": DELIVERY_FAULT_REJECTED,
    "timeout": DELIVERY_FAULT_TIMEOUT,
}

#: 【下一步】里故障那档的三类处置。`transient` 借 `_NEXT["fault"]`，不分叉。
_NEXT_BY_FAULT: dict[str, str] = {
    "transient": _NEXT["fault"],
    "rejected": (
        "这次是**上游拒绝了请求**（4xx）—— **不是任务本身的问题，也不是抖动**："
        "**原样重发同一份内容会再被拒**。\n"
        "**换措辞 / 改内容**再 `delegate` 一次；改不动就向用户说明这一段被拦了。"
    ),
    "timeout": (
        "这次是**时间用完了**（不是它干不了，也不是内容有问题）—— "
        "**可以原样再 `delegate` 一次**。\n"
        "⚠️ 要是**老是**超时，说明这件活对一轮来说太大：**把它拆小**"
        "（分成两次委派）。"
    ),
}

#: 未分类故障的【下一步】：不知道是哪一档，就别替它说该怎么办。
_NEXT_FAULT_UNCLASSIFIED = (
    "**平台侧没跑成**，而且这一档**平台没分类** —— 先看上面那段里的原始原因，"
    "再决定是原样再委派一次还是换措辞。"
)

#: 故障那趟**交过产物**时的【下一步】。处置和"什么都没交"那三档不一样：已交的部分别丢，
#: 先看够不够；要补就**别原样重发**（原样重发会把已经交的整段重做一遍）——改小或换措辞。
#: 不分档：原因上面那段已经说了，这里只说该怎么办。
_NEXT_FAULT_DELIVERED = (
    "**先看它交的够不够**：够 → 接着往下做（把产物 ID 当输入传下去，别让它重做）；"
    "不够 → 补一次委派 —— **别原样重发**（已经交的会整段重做一遍），**改小或换措辞**。"
)

#: 【平台故障】段的三类写法：**只讲原因，不讲处置**（处置住【下一步】，一处定义）。
#: `{detail}` 是原始原因。头一句按档不同：抖动和被拒是"根本没跑成"，超时是"根本没跑完"，
#: 所以整段进表、不共用开头。
_FAULT_SECTION: dict[str, str] = {
    "transient": (
        "这一轮**根本没跑成**：{detail}\n"
        "平台侧的基础设施抖动 —— 不是任务本身的问题，也不是执行器没干成。"
    ),
    "rejected": (
        "这一轮**根本没跑成**：{detail}\n"
        "**上游拒绝了这次请求**（4xx）—— 不是基础设施抖动，也不是执行器没干成；"
        "**原样重发同一份内容会再被拒**。"
    ),
    "timeout": (
        "这一轮**根本没跑完**：{detail}\n"
        "**时间用完了**（整趟跑超了平台的上限）—— 不是它干不了，也不是内容有问题；"
        "这一趟没干完。"
    ),
}

#: 词汇表外的故障档的【平台故障】段 —— 把原始原因摆出来，不替它编处置。
_FAULT_SECTION_UNCLASSIFIED = (
    "这一轮**根本没跑成**：{detail}\n"
    "**平台侧没跑成**，而且这一档**平台没分类** —— 上面那句就是原始原因；"
    "**别按「重试过 / 没重试」去推断**，看它是什么再决定怎么重新委派。"
)

#: `delegate` 那条 trace note 的三类写法，住这里是为了和上面几处同表。
NOTE_BY_FAULT: dict[str, str] = {
    "transient": "**平台故障** → 没跑成",
    "rejected": "**上游拒绝了这次请求**（4xx）→ 没跑成（原样重发会再被拒）",
    "timeout": "**时间用完了**（整趟超墙钟）→ 没跑完（原样再委派可以；老超时就拆小）",
}

NOTE_FAULT_UNCLASSIFIED = "**平台侧没跑成**（原因未分类）"


def next_note(kind: str, *, skill: str = "", fault_kind: str = "", delivered: int = 0) -> str:
    """出口 → 回执的【下一步】那段话（`kind` ∈ `Receipt.kind`）。

    不认识的 `kind` 当场 `KeyError`，不静默给一句错的处置。`skill` 只有 `completed`
    那档用得上，`fault_kind` / `delivered` 只有 `fault` 那档用得上。
    """
    if kind == "completed":
        # 判据要说清：平台不替它判，只把判据指出来 —— 技能指令里就写着交付格式，
        # 够不够由它拿这个标准自己看。
        return (
            "**你判**：这次交付够不够？（不够就换个说法重新委派，或向用户说明）\n"
            f"判据以**技能「{skill}」声明的交付标准**为准"
            f"（不确定标准是什么就用 `read_skill('{skill}')` 读原文）——"
            "交付账就在上面【产出的产物】里。\n"
            "要核对内容就用 `read_artifact('<产物 ID>')` 读全文 —— 正文不在你的上下文里。\n"
            "⚠️ 它说的话在上面【执行器说】里，**过一眼** —— 它可能在话里说了什么"
            "（这次它没调 `ask_user`，平台不替它转达）。"
        )
    if kind == "fault":
        # 三档故障三种处置。走表不走 if/elif：加第五档忘了补表要能被检查出来，
        # 而不是静默走兜底去说一句假话。
        # 交过产物的那趟处置不一样（先看够不够、别原样重发）—— 三档那套默认"什么都没交"。
        if delivered:
            return DELIVERED_DESPITE_FAULT.format(delivered) + "\n" + _NEXT_FAULT_DELIVERED
        return _NEXT_BY_FAULT.get(fault_kind, _NEXT_FAULT_UNCLASSIFIED)
    return _NEXT[kind]


# ──────────────────────────── 构造：结果 → 对象 ────────────────────────────


def _kind_of(outcome: Outcome) -> str:
    """出口分类。故障先判：这一轮根本没跑成，与"执行器干不了"是两回事。"""
    if outcome.fault:
        return "fault"
    if outcome.status == "completed":
        return "completed"
    if outcome.status == "waiting_for_input":
        return "waiting"
    return "failed"


def _deliver_line(outcome: Outcome, kind: str) -> str:
    """① 交付行：一行说清成没成、交付了几样。"""
    if kind == "fault":
        # 三档故障三条交付行，各有各的处置：`rejected` 那档尤其不许提"重试"（根本没重试）
        # （根本没重试）。跑了多久说不清就不说（没有 `wall_s`），留空比编一个数强；
        # `max(1, …)` 是为了别说出"跑到 0 分钟"（<60s 时会四舍五入到 0）。
        mins = (
            f"（跑到 {max(1, round(outcome.wall_s / 60))} 分钟还没交）"
            if getattr(outcome, "wall_s", 0)
            else ""
        )
        template = _DELIVER_BY_FAULT.get(outcome.fault_kind, DELIVERY_FAULT_UNCLASSIFIED)
        # 三张模板都不带别的占位符（只有 timeout 那个 `{}`），所以统一 format 是安全的。
        line = template.format(mins)
        # 交过产物就得说：故障话术默认"什么都没交"，账如实之后这个假设不成立。
        if outcome.artifacts:
            line += "。" + DELIVERED_DESPITE_FAULT.format(len(outcome.artifacts))
        return line
    if kind == "completed":
        return DELIVERY["completed"].format(len(outcome.artifacts))
    if kind == "waiting":
        return DELIVERY["waiting"]
    return DELIVERY["failed"].format(outcome.rounds)


def _produced_block(aids: list[str], store: ArtifactStore) -> str | None:
    """② 产出的产物：逐条列，带真实位置。缺失的显式写出来，不静默跳过。"""
    produced = [a for a in aids if a]
    if not produced:
        return None
    block: list[str] = []
    for aid in produced:
        a = store.get(aid)
        if not a:
            block.append(f"- {aid}（**产物库里没有这一条**）")
            continue
        block.append(
            f"- {aid}《{a['title']}》（{len(a['content'])} 字）"
            f"—— {a.get('summary') or '（无梗概）'}"
        )
        if a.get("path"):
            block.append(f"  位置：{a['path']}")
    block.append(
        # 不许写死 ID：用占位形式，和【下一步】那句同一套写法 —— 指到"存在但内容不对"的
        # 出处，比指不到更坏。
        "（正文在**产物库**里，你的上下文里没有 —— 要看就用 `read_artifact('<产物 ID>')` 读，"
        "不要凭标题猜。回复里报 ID ＋ 路径就够了。）"
    )
    return "\n".join(block)


def build_receipt(
    outcome: Outcome,
    *,
    agent_answer: str = "",
    note: str = "",
    store: ArtifactStore = DEFAULT_STORE,
) -> Receipt:
    """执行器的结果 ＋ 处置提示 → 回执对象。各段有条件的写在这里，`HOPS` 声明的是上限。"""
    kind = _kind_of(outcome)
    sections: dict[str, str] = {"deliver": _deliver_line(outcome, kind)}

    # ①b 它要的信息：`ask_user` 的参数原话（结构，不是从它的话里挑出来的）。
    # 回执里必须有它：主 agent 的回复写在 `finalize` 注入问题之前 —— 不给它这一句，
    # 它写"派了什么、还差什么"时只能编。
    # 不塞进【交付】（那是平台的框架话，混进执行器的原话就串了来源标注），也不是
    # 【执行器说】（那段是它最后说的整段话，混着过场 —— 用户要答的是问题，不是独白）。
    # 缺失时不许静默降级：说清"这一条没留下问题原话"，让下面的【执行器说】把话摆出来。
    if kind == "waiting":
        asked = str(getattr(outcome, "asked", "") or "").strip()
        sections["asked"] = asked or (
            "（**这一条没留下问题原话** —— 下面是它最后说的那段，可能混着过场）"
        )

    # ② 产出的产物
    produced = _produced_block(outcome.artifacts or [], store)
    if produced:
        sections["produced"] = produced

    # ③ 执行器说：全文，一个字都不截（这是"它交付了什么"）。
    # 已知膨胀点：它交代得长的时候，这一段会整段进主 agent 的上下文 —— 代价不是 bug。
    report = str(outcome.report or "").strip()
    if report:
        sections["report"] = report

    # ④ 主 agent 自己试答补的信息（成对记进了 `qa`，这里回显一下）
    if agent_answer.strip():
        sections["agent_answer"] = agent_answer.strip()

    # ⑤ 下一步：处置提示（各出口不同）
    if note:
        sections["next"] = note

    # ⑥ 平台故障：说清是谁的锅 + 该怎么办。单起一段、不塞进【执行器说】—— 那句话的来源
    #    标注是"它说的"，而这段是平台说的。
    if outcome.fault:
        # 三档故障三段话，整段进表（头一句就不一样）。兜底不宣称任何事实 ——
        # 对 `rejected` / `timeout` 提"重试过"就是假话 —— 重试只发生在流的第一步之前。
        section = _FAULT_SECTION.get(outcome.fault_kind, _FAULT_SECTION_UNCLASSIFIED)
        text = section.format(detail=outcome.fault)
        # 交过产物就得说清（有没有交 / 交得完不完整）—— 上面的故障话术默认"什么都没交"。
        if outcome.artifacts:
            text += "\n" + DELIVERED_DESPITE_FAULT.format(len(outcome.artifacts))
        elif outcome.rounds > 1:
            # 一个产物都没交、却推了好几轮：这不是"被时间切停"，是"推了又推都没出口"。
            text += "\n" + NUDGED_WITHOUT_DELIVERY.format(outcome.rounds)
        sections["fault"] = text

    return {"kind": kind, "sections": sections}


def build_rejection(kind: str, detail: str, available: str) -> Receipt:
    """派发被拒（不建 Task）→ 回执对象。

    `kind` ∈ `rejected_skill` / `rejected_artifact`；`available` 是平台列的"现在有什么"。
    """
    return {
        "kind": kind,
        "sections": {"not_dispatched": detail, available_key(kind): available},
    }


#: 拒绝口 → 它列出的"现在有什么"那一段的段名 key。
_REJECTED_AVAILABLE: dict[str, str] = {
    "rejected_skill": "available_skills",
    "rejected_artifact": "available_artifacts",
}


def available_key(kind: str) -> str:
    """拒绝口 → 它列出的那一段的段名 key（一处映射，别在两处各写一份）。"""
    return _REJECTED_AVAILABLE[kind]


def is_rejection(kind: str) -> bool:
    """这个 kind 是不是派发被拒 —— 上面那张表的键就是全部。

    别处要问"是不是拒绝口"就调它，不要按 `rejected_` 前缀近似：前缀认得出表外的 kind，
    认出来了也是错的。
    """
    return kind in _REJECTED_AVAILABLE


# ──────────────────────────── 渲染：注入前那一瞬才拼 ────────────────────────────


def render_receipt(receipt: Receipt) -> str:
    """回执对象 → 给主 agent 看的文本。段名、来源、排布全从词汇表取。"""
    # 渲染收敛到 `handoff.render_hop`（所有交接单共用一条路），这里只负责挑"这一跳是哪
    # 一组段"（派发被拒那条走 `receipt_rejected`）。排版是共用规矩 —— 顶在 0 列时，
    # 执行器的原话和平台框架话左边界一模一样。
    hop = "receipt_rejected" if is_rejection(str(receipt.get("kind") or "")) else "receipt"
    return render_hop(hop, receipt.get("sections") or {})
