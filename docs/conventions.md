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
- 一次提交只做一件事。
- 只写改了什么。不写为什么，也不写工作进度。
- 不写阶段、步骤、优先级标签（`Phase 1`、`Step 2`、`P0`）。
- 代码变更和文档更新放在同一个 commit 里。
- 不写任何署名行，包括 `Co-Authored-By`。
- 提交信息写好后等用户明确说"提交"再 commit。不许自己 push。

提交前跑验证，零新增告警：

```bash
uv run ruff check .
uv run pytest          # 测试建好之后这条才有约束力
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

## 编码

### 动手之前

- 最小修改：只改相关代码，动手前先评估影响范围。
- 复用优先：新增能力前先搜一遍已有实现，确认没有再造。
- 保持一致性：新代码的风格、命名、模式跟已有的同类代码对齐。
- 不确定的方案先写测试验证，不要凭经验判断。

### 格式

Ruff 格式化，行宽 100，双引号，4 空格缩进。

### 类型

- 每个函数都要有完整类型注解，返回值也算。
- 用 `X | None`，不用 `Optional[X]`。
- 不用 `Any`。
- 不用 `# type: ignore`，除非用户确认过。

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
- 外部服务必须 Mock。
- 测试要能重复跑，不依赖执行顺序。
- 文件放 `tests/`，命名 `test_<模块名>.py`。

测试还没建，这几条是建的时候照着来的。

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
