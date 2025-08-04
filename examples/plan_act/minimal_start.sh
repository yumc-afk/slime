#!/bin/bash
# 最简化的Plan/Act启动脚本

# 1. 启动Planner
ray job submit --runtime-env-json '{"env_vars": {"MASTER_PORT_OFFSET": "0"}}' -- \
  python train.py \
  --agent-role planner \
  --enable-plan-act \
  --rollout-function-path examples.plan_act.plan_act_rollout.plan_act_generate_rollout \
  --model-path /path/to/70B-model \
  # ... 其他训练参数

# 2. 启动Actor  
ray job submit --runtime-env-json '{"env_vars": {"MASTER_PORT_OFFSET": "1000"}}' -- \
  python train.py \
  --agent-role actor \
  --enable-plan-act \
  --rollout-function-path examples.plan_act.plan_act_rollout.plan_act_generate_rollout \
  --model-path /path/to/7B-model \
  # ... 其他训练参数