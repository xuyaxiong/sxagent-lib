# 规范（提交、分支、编码）

> 从 `sx-agent` 的 `03_CODING_RULES.md` 和 `04_AI_WORKFLOW.md` 摘出来、按本仓库裁剪过的一份。
> 那边是 FastAPI 服务端，路由、SSE、数据库、OSS、前端那几章这边用不上，没有搬；
> 日志那条也不一样，本仓库走自己的事件总线，下面单说。

## 提交

格式：

```
<type>(<scope>): <描述>

- <变更详情>
```

type 取值：

| type | 什么时候用 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修 bug |
| `refactor` | 重构，行为不变 |
| `perf` | 性能 |
| `style` | 格式 |
| `docs` | 文档 |
| `test` | 测试 |
| `chore` | 构建、工具链、依赖 |

要求：

- 中文，祈使语气。首字母不大写，末尾不加句号。
- 一次提交只做一件事。一次改动太大就拆成几次提交 —— 每条要能独立看懂，主线不能淹在细节里。
- 变更详情写 3 到 4 条。太细看不出主线，太粗等于没写。
- 变更详情一条一行，不折行 —— 折在哪都是随意的，读的人得自己拼回去。
- 只写改了什么。不写为什么，也不写工作进度。
- 非必要不写数字。改了多少行、加了多少测试、动过几个文件、扫了几处 —— `git diff --stat` 和
  日志里都有，写进来是噪音。数字要留就留得有理由：它得是某个结论的依据（"复核过 N 次真会话，
  一次都没出现"），不是顺手的记数。
- 不写阶段、步骤、优先级标签（`Phase 1`、`Step 2`、`P0`）。
- 代码变更和文档更新放在同一个 commit 里。
- 不写任何署名行，包括 `Co-Authored-By`。
- 提交信息写好后等用户明确说"提交"再 commit。不许自己 push。

提交前跑验证，零新增告警：

```bash
uv run ruff check .
uv run pytest
```

## 分支

只有两条，都在同一条线性历史上，不做 merge 提交。

| 分支 | 角色 | 怎么动 |
|---|---|---|
| `dev` | 开发 | 所有提交落在这里 |
| `main` | 基线 | 只 `merge --ff-only`，不直接提交 |

不开特性分支，不重写历史。提测和发布目前没有独立环境，等有了再加分支，
加的时候把这一节补上。

发布就是把 `dev` 推进到 `main`，再打个 tag：

```bash
git checkout main
git merge dev --ff-only
git tag v0.1
```

`pyproject.toml` 里的 `version` 用 PEP 440 三段式（`0.1.0`），git tag 用 `v0.1`，
两者独立管理。

### 发布清单

发布前逐条过，缺一条都不发：

- `uv run ruff check .` 零告警、`uv run pytest` 全绿；
- 单轮冒烟真跑一次：`uv run python -m sxagent --say "你好"`（走真模型）；
- `uv run python scripts/wheel_check.py` —— 构建、验 wheel 里只有 `sxagent/`、wheel 和 sdist
  各装进一个干净 venv 用公共 API 跑通一轮（要网络，几十秒）；
- CLI 那半：`uv run python build_cli.py`，产物在一台没装 Python 的机器上真跑一遍；
- 换过模型/端点的，先按 `evolution.md` 的「换模型这条是硬的」重跑端到端；
- 打 tag、把 `dev` 推到 `main`（上面那三行）。

## 编码

### 动手之前

- 最小修改：只改相关代码，动手前先评估影响范围。
- 复用优先：新增能力前先搜一遍已有实现，确认没有再造。
- 保持一致性：新代码的风格、命名、模式跟已有的同类代码对齐。
- 不确定的方案先写测试验证，不要凭经验判断。

### 格式

Ruff 格式化，行宽 100，双引号，4 空格缩进。

### 命名

标识符一律 ASCII：函数名、变量名、类名、测试名都用英文。中文只出现在注释、docstring 和文案里。

### 类型

- 每个函数都要有完整类型注解，返回值也算。
- 用 `X | None`，不用 `Optional[X]`。
- 不用 `Any`，也少用裸 `list` / `dict`。注解是给人读的（不跑 mypy 强校验），糊一个 `Any` 等于没写。
- 不用 `# type: ignore`。

### 导入

绝对导入，不用相对导入。顺序是标准库、第三方、本地，区块之间空一行。

### 异步

I/O 密集的走 `asyncio`。不用 `threading`、`multiprocessing`、`concurrent.futures`，
也不用 `time.sleep()`。只有同步 SDK 才允许同步调用，而且要用 `asyncio.to_thread()` 包起来。

记录器、`skill_pack`、`artifacts` 现在还有同步文件 I/O 跑在事件循环里，这是记着的欠账，
不是可以照抄的样板。

### 异常

- 不用裸 `except:`。
- 异常要记下来，不要吞掉。
- 业务异常叫 `XxxException`，系统错误叫 `XxxError`。

### 输出

库代码（`sxagent/`）不许 `print()`，也不自己配 `logging`。要记录什么就发事件到
`sxagent.trace`，往哪落由宿主决定 —— 库不该替宿主决定日志去哪。

命令行入口（`sxagent/__main__.py`、`app/`）和 `scripts/` 可以 `print()`，它们的输出就是界面。

token、密码、隐私数据不许写进任何日志或产物。

### 安全

不用 `eval()`、`exec()`。不硬编码密钥，敏感配置走 `.env`。

### 测试

- pytest 加 pytest-asyncio。
- 一个测试只验证一个行为，函数名说清验证的是什么。
- 目录镜像源码：`sxagent/handoff/briefing.py` 的测试放 `tests/sxagent/handoff/test_briefing.py`。
  镜像出来的每一层放 `__init__.py`（或保证文件名全局唯一），否则 pytest 会报 import file
  mismatch，而且报得晚。
- 单元测试不碰真模型、真数据库，外部服务一律 Mock：代码里已经留好了打桩点（`executor.run_task`
  是模块属性、`trace.install_recorder` 是槽位、`session` 是 contextvar），别为测试往产品代码里
  加钩子。真模型那层是跑批和真会话，不是单测的事。
- 断言要强：断对象、断事件、断异常，不断渲染出来的文本。要断文本就断整段相等，不许
  `assert "关键词" in out`，不许正则 —— 见判据零。
- 测试要能重复跑，不依赖执行顺序。
- 一个函数很难单独测，先别急着写测试：多半是职责过大，或者缺一个可替换点。

- 改完源码立刻跑测试时当心 `.pyc`：同一秒里改过两次、文件大小又没变（改一个字又改回去这种），
  Python 按 mtime ＋ 大小判"缓存还有效"，跑的是上一版。故障注入这个"改一下、跑一遍、改回去"
  的节奏最容易撞上 —— 见到红先确认红的是不是你要验的那处，别把"缓存里的旧代码"当成"我刚改的
  那行起作用了"。

### 依赖

用 `uv add` / `uv remove` / `uv sync`。新增前先评估必要性，锁版本范围，
`uv.lock` 提交进版本控制。

## 禁止事项

1. 跨模块直接引用私有成员
2. 模块间循环依赖
3. 硬编码配置值
4. `Any` 类型注解
5. 库代码里的 `print()`
6. `time.sleep()`
7. 裸 `except:`
8. `eval()` / `exec()`
9. 复制已有代码
