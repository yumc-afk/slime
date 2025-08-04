# Plan/Act 双代理架构

一个最小化的Plan/Act实现，通过两个独立的slime job实现大模型规划、小模型执行的异构架构。

## 核心文件

- `plan_act_orchestrator.py` - 协调两个job的核心组件
- `plan_act_rollout.py` - 自定义的rollout函数
- `minimal_start.sh` - 快速启动脚本

## 架构概览

```
Planner Job (70B) ──┐
                    ├──→ PlanActOrchestrator ←──→ 协调执行
Actor Job (7B) ─────┘
```

## 快速开始

1. **修改train.py**（参考sub agent的建议添加orchestrator启动逻辑）

2. **启动Planner**:
```bash
ray job submit -- python train.py \
  --agent-role planner \
  --enable-plan-act \
  --rollout-function-path examples.plan_act.plan_act_rollout.plan_act_generate_rollout \
  --model-path /path/to/70B-model
```

3. **启动Actor**:
```bash
ray job submit -- python train.py \
  --agent-role actor \
  --enable-plan-act \
  --rollout-function-path examples.plan_act.plan_act_rollout.plan_act_generate_rollout \
  --model-path /path/to/7B-model
```

## 工作原理

1. Planner启动时创建PlanActOrchestrator（detached actor）
2. 两个job的Buffer.generate()调用plan_act_rollout函数
3. plan_act_rollout将控制权交给orchestrator
4. orchestrator使用同步栅栏等待两个agent都准备好
5. 执行多轮Plan/Act交互，返回训练数据

## 关键特性

- **最小侵入**：只需修改train.py几行代码
- **同步栅栏**：确保两个agent协调执行
- **超时保护**：避免永久阻塞
- **降级机制**：失败时回退到标准rollout

## 后续优化

需要更复杂功能时可以：
- 增强计划解析逻辑
- 添加真实的工具调用
- 实现更智能的并发Act执行
- 添加监控和性能指标