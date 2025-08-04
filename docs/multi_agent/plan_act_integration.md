# Plan/Act 集成指南

## 需要修改的文件

### 1. slime/utils/arguments.py

添加Plan/Act相关参数：

```python
# 在parse_args函数中添加
group = parser.add_argument_group(title="plan-act")
group.add_argument(
    "--agent-role",
    type=str,
    choices=["planner", "actor"],
    default=None,
    help="Agent role for Plan/Act coordination",
)
group.add_argument(
    "--enable-plan-act",
    action="store_true",
    help="Enable Plan/Act training mode",
)
group.add_argument(
    "--plan-act-timeout",
    type=int,
    default=1800,
    help="Timeout for Plan/Act coordination (seconds)",
)
group.add_argument(
    "--fallback-on-timeout",
    action="store_true",
    help="Fall back to standard rollout on timeout",
)
group.add_argument(
    "--master-port-offset",
    type=int,
    default=0,
    help="Port offset to avoid conflicts between planner and actor",
)
```

### 2. train.py

在train函数开始处添加orchestrator启动逻辑：

```python
def train(args):
    # Plan/Act orchestrator启动逻辑
    if args.enable_plan_act:
        if not args.agent_role:
            raise ValueError("--agent-role must be specified when --enable-plan-act is set")
        
        if args.agent_role == "planner":
            # Planner负责创建orchestrator
            try:
                from slime.coordination import PlanActOrchestrator
                orchestrator = PlanActOrchestrator.options(
                    name="plan_act_orchestrator",
                    lifetime="detached",
                    num_cpus=2
                ).remote()
                print(f"Plan/Act Orchestrator created by Planner")
            except Exception as e:
                print(f"Warning: Failed to create orchestrator: {e}")
                if not args.ignore_orchestrator_errors:
                    raise
    
    # 处理端口偏移避免冲突
    if args.master_port_offset and hasattr(args, 'master_port'):
        args.master_port += args.master_port_offset
    
    # 继续原有的训练流程
    pgs = create_placement_groups(args)
    # ...
```

## 最小化修改原则

1. **不改变核心流程**：训练循环保持不变
2. **通过参数控制**：所有Plan/Act功能通过参数启用
3. **利用扩展点**：使用`--rollout-function-path`注入自定义逻辑
4. **向后兼容**：不影响现有用户

## 与现有架构的关系

```
训练流程:
1. train.py 启动
   ↓
2. 创建 RolloutManager 和 ActorModel
   ↓
3. 训练循环:
   for rollout_id in range(num_rollouts):
       # 这里是Plan/Act的注入点
       data = rollout_manager.async_generate(rollout_id)
       #      ↓ 
       #      Buffer.generate()
       #      ↓
       #      plan_act_generate_rollout() ← 通过--rollout-function-path指定
       
       actor_model.async_train(rollout_id, data)
```

## 为什么这样设计

1. **最小侵入**：只在必要的地方添加代码
2. **清晰分离**：协调逻辑都在coordination模块
3. **灵活扩展**：未来可以添加其他协调模式
4. **生产友好**：容错机制和降级策略

## 快速验证

```bash
# 1. 修改arguments.py和train.py
# 2. 启动测试
python train.py --agent-role planner --enable-plan-act --rollout-function-path slime.coordination.plan_act_rollout.plan_act_generate_rollout --dry-run
```