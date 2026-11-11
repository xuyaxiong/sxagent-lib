# sxagent-lib

多 Skill Agent 库。三个概念分开看：Skill 是专业能力，Task 是一次具体业务工作，Sub Agent 是
可选来执行的 agent，三者互不重叠。平台负责机制，用程序强制；skill 负责业务，用说明引导。

这个库是从 `sx-agent-cli` 里抽出来的，那个仓库现在只作只读参考。

## 这个仓库里有什么

| 目录 / 文件 | 是什么 |
|---|---|
| `sxagent/` | 库本体，运行期的全部逻辑都在这 |
| `app/` | 验证用的 CLI，交互式跑真会话。内部工具，不是交付面 |
| `skills/` | 示例技能包。一个子目录一个技能，入口是 `SKILL.md` |
| `docs/` | `architecture.md` 讲现在长什么样，`decisions.md` 讲为什么，`evolution.md` 讲往哪走，`principles.md` 讲怎么干活，`conventions.md` 讲提交、分支和编码规范 |
| `build_cli.py` | 把验证 CLI 打包成独立程序，发给没装 Python 的测试者 |

## 跑起来

```bash
cp .env.example .env     # 填 API key、模型名、端点
uv sync
uv run python -m app.chat                       # 交互式，你当用户
uv run python -m sxagent --say "帮我查一下……"     # 单轮冒烟
```

打包用 `uv run python build_cli.py`，产物在 `dist/sxagent/`。

打出来的东西必须在一台没装 Python 的机器上真跑一遍才算数。漏收依赖是常态，本机跑通说明不了什么。

## 状态

- 版本 0.1.0。公共 API 还没冻结；导出面已收进 `sxagent/__init__.py` 的 `__all__`
- 已知边界和演化方向在 `docs/evolution.md`
- 没有 fake 模式，也没有离线模式，所有 LLM 调用都走真模型
- 换模型必须重新做端到端验证。有模型在结构化输出上会静默失效，症状是"看起来成功了"，
  详见 `docs/evolution.md`
