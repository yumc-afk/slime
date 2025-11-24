"""
SLIME Coordination模块 - 多Agent训练协调

该模块提供了多种训练协调模式，包括：
- Plan/Act: 大模型规划、小模型执行的异构架构
- 未来可扩展其他协调模式
"""

from .plan_act_orchestrator import PlanActOrchestrator
from .plan_act_rollout import plan_act_generate_rollout

__all__ = [
    "PlanActOrchestrator",
    "plan_act_generate_rollout",
]