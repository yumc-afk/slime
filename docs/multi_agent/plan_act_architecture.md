# Plan/Act 架构设计文档

## 1. 架构概览

Plan/Act是SLIME的一种**核心训练协调模式**，支持异构模型协作：大模型（Planner）负责规划，小模型（Actor）负责执行。

### 1.1 在SLIME中的位置

```
slime/
├── algorithms/        # 训练算法（PPO、DPO等）
├── backends/         # 后端支持（Megatron、SGLang）
├── coordination/     # 训练协调模式 ← Plan/Act在这里
│   ├── plan_act_orchestrator.py
│   └── plan_act_rollout.py
├── models/          # 模型包装器
├── ray/            # 分布式组件
├── rollout/        # 数据生成
└── utils/          # 工具函数
```

### 1.2 与train.py的关系

Plan/Act通过SLIME的**扩展点机制**与训练循环集成：

```python
# train.py的核心循环
for rollout_id in range(num_rollouts):
    # 1. 生成数据 - 这里是Plan/Act的集成点
    rollout_data = rollout_manager.async_generate(rollout_id)
    #              ↓
    #              Buffer.generate()
    #              ↓
    #              plan_act_generate_rollout() ← 自定义函数
    
    # 2. 训练模型
    actor_model.async_train(rollout_id, rollout_data)
    
    # 3. 更新权重
    actor_model.async_update_weights()
```

## 2. 架构设计

### 2.1 组件关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Ray Cluster                              │
├─────────────────────────┬───────────────────────────────────────┤
│   Planner Job (70B)     │        Actor Job (7B)                 │
│                         │                                        │
│  train.py               │      train.py                         │
│    ↓                    │        ↓                              │
│  创建Orchestrator ←─────┼────→ 连接Orchestrator                 │
│    ↓                    │        ↓                              │
│  RolloutManager         │      RolloutManager                   │
│    ↓                    │        ↓                              │
│  Buffer.generate()      │      Buffer.generate()                │
│    ↓                    │        ↓                              │
│  plan_act_rollout() ────┼────→ plan_act_rollout()              │
│         ↓               │              ↓                        │
│         └───────────────┴───────────────┘                       │
│                         ↓                                        │
│              PlanActOrchestrator                                │
│              (Detached Actor)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **最小侵入**：通过`--rollout-function-path`参数注入，不修改核心训练逻辑
2. **独立生命周期**：Orchestrator作为detached actor，独立于训练job
3. **同步协调**：使用同步栅栏确保两个agent协调执行
4. **统一接口**：返回标准的Sample格式，与SLIME训练流程完全兼容

## 3. 执行时序

```
Planner Job                  Orchestrator              Actor Job
    |                            |                          |
    |------ 启动 -------->       |                          |
    |   创建Orchestrator         |                          |
    |                            |                          |
    |                            |      <------ 启动 -------|
    |                            |        连接Orchestrator  |
    |                            |                          |
    |--- rollout_id=0 --->       |                          |
    |   Buffer.generate()        |                          |
    |   plan_act_rollout() ----> |                          |
    |                            |                          |
    |                            |      <--- rollout_id=0 --|
    |                            |         Buffer.generate()|  
    |                            |      <-- plan_act_rollout|
    |                            |                          |
    |                     同步栅栏等待                      |
    |                            |                          |
    |                    开始协调执行                       |
    |                            |                          |
    |    <--- 获取任务 ---       |                          |
    |    生成计划 --->           |                          |
    |                            |       --- 分发计划 ----> |
    |                            |       <--- 执行结果 ---- |
    |                            |                          |
    |                    整合结果返回                       |
    |                            |                          |
    |    <--- 训练数据 ---       |       --- 训练数据 ---> |
    |                            |                          |
    |   继续训练流程            |         继续训练流程     |
    |                            |                          |
```

## 4. 与train_async.py的关系

### 4.1 兼容性分析

train_async.py使用pipeline并行优化GPU利用率：
```python
# 当前rollout生成时，同时训练上一个rollout
rollout_next_future = rollout_manager.async_generate(rollout_id + 1)
ray.get(actor_model.async_train(rollout_id, current_rollout))
```

Plan/Act的同步栅栏可能影响pipeline效率，但可以通过以下方式优化：

1. **预生成**：Orchestrator可以提前协调下一个rollout
2. **批量处理**：一次协调多个rollout_id
3. **异步变体**：实现异步版本的Plan/Act协调器

### 4.2 使用建议

- **同步训练**（train.py）：Plan/Act工作良好，适合实验和原型
- **异步训练**（train_async.py）：需要额外优化以保持pipeline效率

## 5. 配置和使用

### 5.1 必需的参数

```bash
# 通用参数
--enable-plan-act               # 启用Plan/Act模式
--agent-role {planner,actor}    # 指定角色
--rollout-function-path slime.coordination.plan_act_rollout.plan_act_generate_rollout

# 可选参数
--plan-act-timeout 1800         # 协调超时（秒）
--fallback-on-timeout           # 超时后降级到标准rollout
--master-port-offset 0/1000     # 避免端口冲突
```

### 5.2 启动流程

1. **启动Planner**（自动创建Orchestrator）：
```bash
ray job submit -- python train.py \
    --agent-role planner \
    --enable-plan-act \
    --rollout-function-path slime.coordination.plan_act_rollout.plan_act_generate_rollout \
    # ... 其他70B模型参数
```

2. **启动Actor**（自动连接Orchestrator）：
```bash
ray job submit -- python train.py \
    --agent-role actor \
    --enable-plan-act \
    --rollout-function-path slime.coordination.plan_act_rollout.plan_act_generate_rollout \
    # ... 其他7B模型参数
```

## 6. 扩展性设计

### 6.1 添加新的协调模式

```python
# 未来可以添加其他协调模式
slime/coordination/
├── base_coordinator.py      # 基类（未来）
├── plan_act_orchestrator.py # Plan/Act实现
├── peer_review_orchestrator.py # 同行评审模式（示例）
└── ensemble_orchestrator.py    # 集成学习模式（示例）
```

### 6.2 自定义Plan/Act行为

通过继承和重写方法：
```python
class CustomPlanActOrchestrator(PlanActOrchestrator):
    def _parse_concurrent_acts(self, plan_sample):
        # 自定义计划解析逻辑
        pass
    
    def _execute_concurrent_acts(self, actor_buffer, acts, context):
        # 自定义执行逻辑
        pass
```

## 7. 性能考虑

### 7.1 开销分析

- **通信开销**：跨job的Ray actor调用，通常1-10ms
- **同步等待**：取决于两个job的启动时间差
- **内存使用**：Orchestrator维护session状态，定期清理

### 7.2 优化建议

1. **减少同步点**：批量处理多个rollout
2. **预热缓存**：提前加载常用prompt
3. **监控指标**：添加协调延迟、成功率等指标

## 8. 故障处理

### 8.1 容错机制

- **超时保护**：默认30分钟超时
- **降级策略**：失败时回退到标准rollout
- **Session清理**：自动清理过期session

### 8.2 常见问题

1. **Orchestrator未找到**：检查Planner是否成功启动
2. **超时错误**：增加`--plan-act-timeout`或检查模型推理速度
3. **内存泄漏**：确保Orchestrator定期清理旧session