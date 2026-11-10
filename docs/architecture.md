# 架构（现状说明书）

> 描述**现在是什么样**；实现改了就改它。字段级明细以代码为准。

## 一句话

**Skill 是专业能力，Task 是具体业务工作，Sub Agent 是可选执行者 —— 三者正交。**

subagent 的边界画在「**参数齐了之后的执行段**」：上下文隔离保留，
"重复问用户 / 状态间歇丢失 / 提问写进正文"三类问题**结构性消失**。

## 模块地图（`sxagent/` —— 库本体）

| 文件 | 负责 | **不**负责 |
|---|---|---|
| `config.py` | `app_root()`（冻结后 = exe 目录）· `RESOURCE_ROOT` · `.env` 加载 · **三流 UTF-8** · `get_model()` | 不碰业务 |
| `llm.py` | `safe_structured()`（重试 1 次 → 保守默认） | 不知道谁在调它 |
| `registry.py` | 技能表的**唯一持有者**（`get/all/replace/extend/reset`） | 不定义 Skill 语义；**对外不给 dict** |
| `skills.py` | `Skill`（`name`/`description`/`instructions` 三个字段）· 内置表 · `read_skill` | **不做磁盘 I/O** |
| `skill_pack.py` | 文件夹形态 → `Skill`（`SKILL.md` + frontmatter）；自带文件的**列与读** | 不定义语义 |
| `artifacts.py` | 产物库（私有表 + 读 API）· ID 发号器 · 类型→扩展名派发 · 当场落盘 | 不管技能 |
| `executor.py` | 执行器：**三态由结构定** · `ask_user` 申报 · 出口限制循环 · `run_task()` · `read_user_turn` | 不持有状态、不跨轮、**不直接问用户** |
| `agent.py` | 主 agent：`delegate` 工具 · **交接书** `render_briefing` · 用户原话快照 · 试答 | **不解析、不抽取参数** |
| `graph.py` | 图：`feed` / `prepare` / `agent` / `finalize` + `build()` | 不做业务判断 |
| `trace.py` | **事件总线** + 全量日志（逐 super-step + 逐字流式） | 不截断正文（记录层不砍） |
| `stream.py` | 传输层（从 `trace` 拆出） | 不做总线 |
| `handoff/` | **一个交接点 = 一个模块 = schema + 模板**（briefing / receipt / ledger / nudge / ask / decision） | 不持有状态 |
| `aio.py` / `session.py` / `recorder.py` | 异步支撑 / 会话标识 / 日志落盘 | — |
| `__main__.py` | 裸 CLI（内置示例技能，显示层最简） | — |

`app/`（**验证 CLI，不是库本体**）：`runtime.py`（装配 + 记录层）· `chat.py`（交互式验证入口）。
`build_cli.py`：把验证 CLI 打成 exe。

## 依赖方向（单向，无环）

```
config ──► trace ──► (所有模块都在写日志)
  │
  ├─► llm ──► executor ──► agent ──► graph
  │              ▲            ▲          │
  ├─► skills ────┘            │          │
  ├─► skill_pack ─────────────┘          │
  └─► artifacts ─────────────────────────┘
```

两条边界纪律（踩出来的）：**`executor` 与 `agent` 互相不 import**（公共 LLM 调用放 `llm.py`）；
**`skills` / `artifacts` 是两个领域**（技能规范变了 vs 交付机制变了）。
**技能表只有一个家：`registry.py`** —— 三方都只认 `registry.get()`，换技能只走 `replace/extend/reset`。

## 三个领域对象 + 一条不变量

| 对象 | 在哪 | 是什么 |
|---|---|---|
| **Skill** | `skills.py` | 静态技能包：`name` + `description` + `instructions`（只有这三个字段） |
| **Task** | `graph.State["tasks"]` | 一次具体业务工作：跨轮存活（字段定义见 `handoff/briefing.py`） |
| **Artifact** | `artifacts.py` | 运行期交付物（字段定义见该模块） |

> 🔑 **一条不变量**：`read_skill_file` **只认 `list_skill_files` 给得出来的东西**——
> 路径穿越、二进制、超大文件全被这一条挡住，不需要第二套路径检查。

## 一次 `invoke` 的时序

```
用户 ─► ① feed（纯代码）
          定向投递：答复只投给 answer_to 指的那条待答（没有标识就不投）
          → 渲染台账注入 messages（顺序不能反：先投递，台账才反映新状态）
       ② agent（主 agent 子图 —— 完整的 agent：自己的模型 + 工具 + 循环）
          🔴 待答时不发言：那一刻是**决策点**，不是对话轮次
          三条出路自己选：直接答 / write_artifact（都不建 Task）/ delegate（建 Task）
       ③ delegate 工具内部（平台代码，LLM 跳不过）
          校验：技能名不存在 → 拒绝；产物 ID 不存在 → 拒绝（都不建 Task）
          → run_task()（执行器，隔离上下文）：
             申报 ask_user = 要信息；交产物 = 交付
          → 三态**由结构定**（不是 LLM 判的），穷举分派：
             waiting_for_input → 主 agent 强制试答 → 答不上才登记待答
             completed         → 交付（产物 ID + 路径 + 摘要）
             incomplete        → failed，交给主 agent
       ④ finalize（纯代码）
          有待答 → **无条件**把它申报的原话贴进回复（透传）
```

## 状态机

```
delegate 建 Task → running ─┬─ completed（交了产物）
                            ├─ waiting_for_input（调了 ask_user）
                            └─ failed（推满 3 轮仍 incomplete）
   另有 superseded：同 skill 的 failed 旧 Task 被重新委派时自动标记
```

**四条防错区分**（实现时最容易混）：

| 不要混 | 说明 |
|---|---|
| `waiting_for_input` ≠ 图失败 | 前者是**正常状态** |
| `completed` ≠ 任务完成 | 只表示"**交了产物**"这个事实；"够不够"由**主 agent 判** |
| `failed` ≠ 模型报错 | 定义是"推满 3 轮仍既没交产物、也没申报要信息" |
| 执行器"跑完了" ≠ Task 完成 | 所以平台只报结构事实 |

## 交接（这套系统的行为几乎全由它决定）

三条硬规矩：

1. **用户原话平台硬搬**（不许 LLM 转述——转述不只丢，**还会加料**）；
   长的走**渐进式披露**（清单常驻 + `read_user_turn` 按需取全文）
2. **切了必须标** —— "摘要"和"截断"是两回事；悄悄切一刀不算
3. **每段标来源**（【用户原话】是事实、【任务】是要求）—— 平台不拦主 agent 加料，
   但必须让下游知道每句话是谁说的

## 软判断 / 硬判断

**判据**：违反会导致"状态不一致"的 → **硬**（程序强制）；只是"效果差一点"的 → **软**（LLM 决策）。

- 运行期**平台主动发起**的 LLM 调用**只剩一处**：主 agent 试答（输入缺一不可——
  缺一样它就凭空猜）。其余软判断是两个 agent 自己的循环。
- 几条代表性硬判断（都在代码里）：产物必须**真实存在且是这次新写的** ·
  状态字段由代码写 · 三态穷举分派（没有第四种）· 待答不丢（无条件透传）·
  产物 ID 用发号器（`len+1` 并行会撞）· 路径只从 `app_root()` 算。

## 可观测性

`trace` 是**事件总线**：记录器与展示层**各自渲染事件**，谁也不读对方的产物——
"不许把渲染出来的文本反解析回结构"是全局判据（改这块前先过这条判据）。
