"""多技能 agent 库：主 agent 收参 → 派发隔离执行器 → 缺参数 → 确定性续跑。

宿主只需要一个装配入口（`build`），其余是它要给的配料和要读的词汇：

    from sxagent import build, LocalSkills, MemoryArtifacts, session, trace

## 导出面

- **装配**：`build`（整张图）、`next_pending` / `asked_text`（通道要摆到用户面前的那个待答）
- **配料**（宿主注入 `build`）：`Skill`、`SkillSource` / `LocalSkills` / `DEFAULT_SOURCE` /
  `builtin_skills`、`ArtifactStore` / `MemoryArtifacts` / `DEFAULT_STORE` / `file_ext`
  （声明类型 → 扩展名：实现产物库时记 `ext` 用同一条规则）/ `next_aid`（发产物 ID：
  形状由库定，实现产物库时同走它）
- **接线**：`session`（会话隔离）、`trace`（事件总线，展示层订阅它，不读日志文件）、
  `Task` / `TaskStage` / `Outcome` / `OutcomeStatus` / `FaultKind`（交接的词汇表）、
  `delivered_marks`（工具结果的交付标记：宿主拼、执行器按同名字段入账）

实现件（`agent` / `executor` / `handoff` / `stream` / `recorder` / `llm` / `aio` / `config`）
不从这里出：它们是库的内部结构，公共面变了才算破坏兼容。

import 这个包不在宿主那边留任何东西：不读 `.env`，也不在磁盘上建目录。"""

from sxagent import session, trace
from sxagent.artifacts import DEFAULT_STORE, ArtifactStore, MemoryArtifacts, file_ext, next_aid
from sxagent.graph import build, next_pending
from sxagent.handoff import FaultKind, Outcome, OutcomeStatus, TaskStage, delivered_marks
from sxagent.handoff.briefing import Task, asked_text
from sxagent.skill_pack import DEFAULT_SOURCE, LocalSkills
from sxagent.skills import Skill, SkillSource, builtin_skills

__all__ = [
    "DEFAULT_SOURCE",
    "DEFAULT_STORE",
    "ArtifactStore",
    "FaultKind",
    "LocalSkills",
    "MemoryArtifacts",
    "Outcome",
    "OutcomeStatus",
    "Skill",
    "SkillSource",
    "Task",
    "TaskStage",
    "asked_text",
    "build",
    "builtin_skills",
    "delivered_marks",
    "file_ext",
    "next_aid",
    "next_pending",
    "session",
    "trace",
]
