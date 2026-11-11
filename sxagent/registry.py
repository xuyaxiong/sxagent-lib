"""技能注册表：全进程唯一的一张 `{name: Skill}` 表。

`skills.py` 在 import 时登记内置表，此后谁都不许直接拿表。读走 `get` / `all` / `values` /
`names` / `catalog`，换表走 `registry.testing` 下的三个口子，每次调用查的都是当前那张。

## 不变量

1. 对外只给函数，不给 dict。`from sxagent.registry import SKILLS` 等于又造一个绑定点。
2. 换表的口子在 `testing` 上，不和读 API 同层。"这是装配 / 测试口"由结构说，不靠下划线。
3. 未初始化时 `get()` 直接抛错，不静默返回 None：静默会造出"没有这个技能"这种假事实。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅用于类型标注 —— 运行时不 import，否则 registry ↔ skills 成环
    from sxagent.skills import Skill


class _Registry:
    """一张内置表 + 一张当前表。分开是为了 `replace` 之后不串 case：直接拿当前表当下次
    合并的底，第一个 case 装过的技能会被第二个 case 合并进来。"""

    def __init__(self) -> None:
        #: 内置技能表（`skills.py` 启动时登记）。只读，`replace` 不碰它。
        self._defaults: dict[str, Skill] = {}
        #: 当前生效表。整体替换引用而不是原地 clear+update —— 并发读的人要么看到旧的
        #: 完整一份、要么看到新的完整一份，不会撞见半张空表。
        self._current: dict[str, Skill] = {}
        self._initialized = False
        #: 测试口 —— 换表那三个动作，见 `_Testing`。
        #: 它和读 API 不在一个平面上：读是生产的，换表是装配 / 测试的。
        self.testing = _Testing(self)

    # ──────────────────────────── 写（唯一入口） ────────────────────────────

    def register_defaults(self, skills: Mapping[str, Skill]) -> None:
        """登记内置技能表 —— 由 `skills.py` 在模块级调用一次。同时它就是初始的当前表。"""
        self._defaults = dict(skills)
        self._current = dict(skills)
        self._initialized = True

    # ──────────────────────────── 读 ────────────────────────────

    def get(self, name: str) -> Skill | None:
        """按名取技能；没有就返回 None（"表里没有"是业务事实，由调用方决定怎么回话）。"""
        self._require_init()
        return self._current.get(name)

    def all(self) -> dict[str, Skill]:
        """当前表副本 —— 调用方随便改，改不到注册表。"""
        self._require_init()
        return dict(self._current)

    def values(self) -> list[Skill]:
        self._require_init()
        return list(self._current.values())

    def names(self) -> list[str]:
        self._require_init()
        return list(self._current)

    def catalog(self) -> str:
        """技能清单的唯一排版来源（渐进式披露第一层）。这里改一次，别处都跟着变。"""
        self._require_init()
        return "\n".join(f"- {s.name}：{s.description}" for s in self._current.values())

    def _require_init(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "技能注册表未初始化 —— 没有任何模块 import 过 `sxagent.skills`。"
                "默认表由它登记；用技能之前请先 `import sxagent.skills`。"
            )


class _Testing:
    """换表那三个口子，单独成对象是为了把"这是测试口"从名字升级成结构。

    它们改的是全进程那张表，生产代码里写出 `registry.testing.reset()` 会自己看见不对劲。
    """

    def __init__(self, reg: _Registry) -> None:
        self._reg = reg

    def reset(self) -> None:
        """把当前表恢复成内置表 —— 跑批每个 case 开始前调它。"""
        self._reg._require_init()
        self._reg._current = dict(self._reg._defaults)

    def replace(self, skills: Mapping[str, Skill]) -> None:
        """整体替换当前表（评测的 `skills_mode: replace`）。内置表不受影响。"""
        self._reg._require_init()
        self._reg._current = dict(skills)

    def extend(self, extra: Mapping[str, Skill]) -> None:
        """在当前表上合并。要"干净地加几个"，先 `reset()` 再 `extend()`，两步是两件事。"""
        self._reg._require_init()
        self._reg._current = {**self._reg._current, **extra}


#: 全进程唯一实例。只导入这个对象，不要复制表。
registry = _Registry()
