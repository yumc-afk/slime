"""
Plan/Act自定义rollout函数
通过--rollout-function-path参数指定使用
"""
import os
import time
import logging
import ray

logger = logging.getLogger(__name__)


def plan_act_generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """
    Plan/Act模式的rollout生成函数
    这个函数被Buffer.generate()调用，但实际控制权交给orchestrator
    
    Args:
        args: 训练参数
        rollout_id: 当前rollout的ID
        data_buffer: Buffer实例的引用
        evaluation: 是否是评估模式
    
    Returns:
        生成的样本列表
    """
    if evaluation:
        # 评估模式暂时使用默认行为
        from slime.rollout.sglang_rollout import generate_rollout
        return generate_rollout(args, rollout_id, data_buffer, evaluation)
    
    # 获取agent角色
    agent_role = getattr(args, "agent_role", None)
    if not agent_role:
        raise ValueError("agent_role not specified in args")
    
    logger.info(f"[{agent_role}] Starting Plan/Act rollout {rollout_id}")
    
    # 获取orchestrator
    orchestrator = None
    max_retries = 60  # 最多等待5分钟
    
    for retry in range(max_retries):
        try:
            orchestrator = ray.get_actor("plan_act_orchestrator")
            logger.info(f"[{agent_role}] Connected to orchestrator")
            break
        except ValueError:
            if agent_role == "planner" and retry == 0:
                # Planner应该已经创建了orchestrator
                logger.error("Orchestrator should be created by Planner")
                raise RuntimeError("Orchestrator not found. Check if --enable-plan-act is set.")
            else:
                # Actor等待orchestrator创建
                if retry % 12 == 0:  # 每分钟打印一次
                    logger.info(f"[{agent_role}] Waiting for orchestrator... ({retry}/60)")
                time.sleep(5)
    else:
        raise TimeoutError(f"[{agent_role}] Orchestrator not found after {max_retries*5} seconds")
    
    # 将控制权交给orchestrator
    logger.info(f"[{agent_role}] Delegating control to orchestrator for rollout {rollout_id}")
    
    try:
        # 设置超时，防止永久阻塞
        timeout = getattr(args, "plan_act_timeout", 1800)  # 默认30分钟
        
        result = ray.get(
            orchestrator.execute_rollout.remote(
                rollout_id,
                agent_role,
                data_buffer
            ),
            timeout=timeout
        )
        
        logger.info(f"[{agent_role}] Rollout {rollout_id} completed with {len(result)} samples")
        
        # 转换结果格式以匹配slime的期望
        return _convert_to_slime_format(result, args)
        
    except ray.exceptions.GetTimeoutError:
        logger.error(f"[{agent_role}] Rollout {rollout_id} timed out after {timeout}s")
        # 降级到普通rollout
        if hasattr(args, "fallback_on_timeout") and args.fallback_on_timeout:
            logger.warning(f"[{agent_role}] Falling back to standard rollout")
            from slime.rollout.sglang_rollout import generate_rollout
            return generate_rollout(args, rollout_id, data_buffer, evaluation)
        else:
            raise
    except Exception as e:
        logger.error(f"[{agent_role}] Rollout {rollout_id} failed: {e}")
        raise


def _convert_to_slime_format(orchestrator_results, args):
    """
    将orchestrator返回的结果转换为slime期望的格式
    slime期望: List[Sample] 或 List[List[Sample]]
    """
    from slime.utils.types import Sample
    
    # 如果已经是Sample对象，直接返回
    if orchestrator_results and isinstance(orchestrator_results[0], Sample):
        return orchestrator_results
    
    # 转换字典格式为Sample对象
    samples = []
    for item in orchestrator_results:
        if isinstance(item, dict):
            sample = Sample(
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                reward=item.get("reward", 0.0),
                metadata=item.get("metadata", {})
            )
            samples.append(sample)
        else:
            samples.append(item)
    
    # 根据n_samples_per_prompt分组
    n_samples = getattr(args, "n_samples_per_prompt", 1)
    if n_samples > 1:
        # 分组返回
        grouped_samples = []
        for i in range(0, len(samples), n_samples):
            grouped_samples.append(samples[i:i+n_samples])
        return grouped_samples
    else:
        # 直接返回
        return samples


# 为了方便测试，也提供一个评估版本
def plan_act_eval_rollout(args, rollout_id, data_buffer, evaluation=True):
    """
    评估模式的rollout函数
    """
    # 评估时可能想要不同的行为
    return plan_act_generate_rollout(args, rollout_id, data_buffer, evaluation)