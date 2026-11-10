# sxagent-lib

> 多 Skill Agent 库：**Skill 是专业能力，Task 是具体业务工作，Sub Agent 是可选执行者 —— 三者正交。**
> 平台管机制（**程序强制**），skill 管业务（**行为建议**）。

抽取自 `sx-agent-cli`@`52d83b1`（2026-11-10）。

## 这个仓库里有什么

| 目录 / 文件 | 是什么 |
|---|---|
| `sxagent/` | **库本体**（运行期全部逻辑） |
| `app/` | 验证 CLI：交互式跑真会话 —— **内部工具，不是交付面** |
| `skills/` | 示例技能包（文件夹形态：一个子目录 = 一个技能，入口是 `SKILL.md`） |
| `docs/` | `architecture.md`（设计现状）· `decisions.md`（关键决策与为什么）· `evolution.md`（往哪走） |
| `build_cli.py` | 把验证 CLI 打包成独立程序（发给没装 Python 的测试者） |

## 跑起来（验证 CLI）

```bash
cp .env.example .env     # 填 API key / 模型名 / 端点
uv sync
uv run python -m app.chat                       # 交互式（你当用户）
uv run python -m sxagent --say "帮我查一下……"     # 单轮冒烟
```

打包：`uv run python build_cli.py` → `dist/sxagent/`。
⚠️ 打出来**必须在一台没装 Python 的机器上真跑一遍**才算数（漏收依赖是常态）。

## 状态（2026-11-10）

- 版本 **0.1.0**：库的**公共 API 尚未冻结**（导出面在整理中）
- 已知边界与演化方向：`docs/evolution.md`
- ⚠️ **没有 fake / 离线模式**：所有 LLM 调用都走真实模型
- ⚠️ **换模型必须重新端到端验证**（有模型在结构化输出上静默失效，见 `docs/evolution.md`）
