"""
Plan/Act Orchestrator - 协调两个独立slime job的核心组件
"""
import asyncio
import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

import ray

logger = logging.getLogger(__name__)


@dataclass
class RolloutSession:
    """单个rollout的会话状态"""
    rollout_id: int
    planner_data: Optional[Dict] = None
    actor_data: Optional[Dict] = None
    planner_buffer: Optional[Any] = None  # Buffer reference
    actor_buffer: Optional[Any] = None
    result_future: Optional[asyncio.Future] = None
    executed: bool = False
    created_at: float = 0
    
    def __post_init__(self):
        self.created_at = time.time()
        self.result_future = asyncio.Future()


@ray.remote
class PlanActOrchestrator:
    """
    协调Planner和Actor两个job的核心组件
    使用同步栅栏模式确保两个agent都准备好后再执行
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.sessions: Dict[int, RolloutSession] = {}
        self.max_turns = self.config.get("max_turns", 5)
        self.max_concurrent_acts = self.config.get("max_concurrent_acts", 8)
        self.session_timeout = self.config.get("session_timeout", 3600)  # 1 hour
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # 统计信息
        self.stats = defaultdict(int)
        
        logger.info(f"PlanActOrchestrator initialized with config: {self.config}")
    
    async def execute_rollout(self, rollout_id: int, agent_role: str, data_buffer_ref):
        """
        两个agent都会调用这个方法，但只执行一次主逻辑
        使用同步栅栏模式
        """
        logger.info(f"Rollout {rollout_id}: {agent_role} arrived")
        
        # 定期清理过期session
        self._maybe_cleanup_old_sessions()
        
        # 初始化session
        if rollout_id not in self.sessions:
            self.sessions[rollout_id] = RolloutSession(rollout_id=rollout_id)
        
        session = self.sessions[rollout_id]
        
        # 记录到达的agent
        if agent_role == "planner":
            session.planner_data = {"arrived_at": time.time()}
            session.planner_buffer = data_buffer_ref
        elif agent_role == "actor":
            session.actor_data = {"arrived_at": time.time()}
            session.actor_buffer = data_buffer_ref
        else:
            raise ValueError(f"Unknown agent role: {agent_role}")
        
        # 检查是否两个都到了
        if (session.planner_data and session.actor_data and not session.executed):
            session.executed = True
            logger.info(f"Rollout {rollout_id}: Both agents ready, starting orchestration")
            
            try:
                # 执行主逻辑
                result = await self._orchestrate_plan_act_loop(session)
                session.result_future.set_result(result)
                self.stats["successful_rollouts"] += 1
            except Exception as e:
                logger.error(f"Rollout {rollout_id} failed: {e}")
                session.result_future.set_exception(e)
                self.stats["failed_rollouts"] += 1
        
        # 两个agent都等待同样的结果
        try:
            return await session.result_future
        except Exception as e:
            logger.error(f"Rollout {rollout_id}: {agent_role} got exception: {e}")
            raise
    
    async def _orchestrate_plan_act_loop(self, session: RolloutSession):
        """
        核心的Plan/Act协调逻辑
        """
        rollout_id = session.rollout_id
        all_samples = []
        context = ""
        
        # 从Buffer获取任务
        tasks = await self._get_tasks_from_buffer(session.planner_buffer)
        logger.info(f"Rollout {rollout_id}: Got {len(tasks)} tasks")
        
        for turn in range(self.max_turns):
            logger.info(f"Rollout {rollout_id}: Starting turn {turn}")
            
            # 1. Plan阶段
            plan_samples = await self._generate_plan(
                session.planner_buffer, 
                tasks, 
                context,
                turn
            )
            all_samples.extend(plan_samples)
            
            # 2. 解析并发任务
            concurrent_acts = self._parse_concurrent_acts(plan_samples[-1])
            if not concurrent_acts:
                logger.info(f"Rollout {rollout_id}: No acts to execute, finishing")
                break
            
            # 3. Act阶段（并发执行）
            act_results = await self._execute_concurrent_acts(
                session.actor_buffer,
                concurrent_acts,
                context
            )
            all_samples.extend(act_results)
            
            # 4. 更新上下文
            context = self._update_context(context, plan_samples[-1], act_results)
            
            # 5. 检查完成条件
            if self._is_task_complete(act_results):
                logger.info(f"Rollout {rollout_id}: Task completed")
                break
        
        logger.info(f"Rollout {rollout_id}: Completed with {len(all_samples)} samples")
        
        # 返回给两个agent相同的数据
        return self._prepare_results_for_agents(all_samples)
    
    async def _get_tasks_from_buffer(self, buffer_ref):
        """从Buffer获取任务"""
        # 调用Buffer的get_samples方法
        batch_size = self.config.get("rollout_batch_size", 1)
        tasks = ray.get(buffer_ref.get_samples.remote(batch_size))
        return tasks
    
    async def _generate_plan(self, planner_buffer, tasks, context, turn):
        """调用Planner生成计划"""
        # 这里应该调用planner_buffer的某个方法来生成计划
        # 简化实现：返回模拟数据
        plan_sample = {
            "prompt": f"Plan for tasks: {tasks}\\nContext: {context}",
            "response": f"<plan>\\n<act>Search for information about task</act>\\n<act>Analyze results</act>\\n</plan>",
            "metadata": {"turn": turn, "agent": "planner"}
        }
        return [plan_sample]
    
    def _parse_concurrent_acts(self, plan_sample):
        """从计划中解析出可并发的任务"""
        # 简化实现：解析<act>标签
        import re
        acts = re.findall(r'<act>(.*?)</act>', plan_sample.get("response", ""))
        return [{"task": act, "id": i} for i, act in enumerate(acts)]
    
    async def _execute_concurrent_acts(self, actor_buffer, acts, context):
        """并发执行多个Act"""
        # 使用asyncio.gather并发执行
        act_futures = []
        for act in acts:
            future = self._execute_single_act(actor_buffer, act, context)
            act_futures.append(future)
        
        results = await asyncio.gather(*act_futures, return_exceptions=True)
        
        # 处理结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Act {acts[i]} failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _execute_single_act(self, actor_buffer, act, context):
        """执行单个Act"""
        # 简化实现
        result = {
            "prompt": f"Execute: {act['task']}\\nContext: {context}",
            "response": f"Executed {act['task']} successfully",
            "metadata": {"act_id": act["id"], "agent": "actor"}
        }
        # 模拟执行时间
        await asyncio.sleep(0.1)
        return result
    
    def _update_context(self, context, plan_sample, act_results):
        """更新对话上下文"""
        new_context = context + f"\\nPlan: {plan_sample['response']}"
        for result in act_results:
            new_context += f"\\nAct Result: {result['response']}"
        return new_context
    
    def _is_task_complete(self, act_results):
        """检查任务是否完成"""
        # 简化实现：检查是否有包含"complete"的结果
        for result in act_results:
            if "complete" in result.get("response", "").lower():
                return True
        return False
    
    def _prepare_results_for_agents(self, samples):
        """准备返回给两个agent的结果"""
        # 两个agent需要相同格式但可能不同内容的数据
        # 这里返回所有samples供训练使用
        return samples
    
    def _maybe_cleanup_old_sessions(self):
        """定期清理过期的session"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        expired_sessions = []
        
        for rollout_id, session in self.sessions.items():
            if current_time - session.created_at > self.session_timeout:
                expired_sessions.append(rollout_id)
        
        for rollout_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {rollout_id}")
            del self.sessions[rollout_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_stats(self):
        """获取统计信息"""
        return dict(self.stats)