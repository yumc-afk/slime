#!/bin/bash

# Plan/Act架构启动脚本
# 
# 该脚本启动完整的Plan/Act架构，包括Orchestrator、Planner和Actor三个组件
# 基于slime现有架构，通过最小侵入的方式实现Plan/Act功能

set -e

# 配置参数
SHARED_NAMESPACE=${SHARED_NAMESPACE:-"plan_act_shared"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-"/root/Megatron-LM/"}

# 模型和数据路径
PLANNER_MODEL=${PLANNER_MODEL:-"/path/to/planner/model"}
ACTOR_MODEL=${ACTOR_MODEL:-"/path/to/actor/model"}
REF_MODEL=${REF_MODEL:-"/path/to/reference/model"}
TRAIN_DATA=${TRAIN_DATA:-"/path/to/train/data"}

# 训练参数
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
PLAN_BATCH_SIZE=${PLAN_BATCH_SIZE:-64}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-1000}

# GPU配置
ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-8}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.7}

# 超时配置
PLAN_TIMEOUT=${PLAN_TIMEOUT:-300}
FEEDBACK_TIMEOUT=${FEEDBACK_TIMEOUT:-300}

echo "=== Starting Plan/Act Architecture ==="
echo "Shared Namespace: $SHARED_NAMESPACE"
echo "Ray Address: $RAY_ADDRESS"
echo "Rollout Batch Size: $ROLLOUT_BATCH_SIZE"
echo "Plan Batch Size: $PLAN_BATCH_SIZE"
echo "======================================="

# 函数：检查Ray集群状态
check_ray_cluster() {
    echo "Checking Ray cluster status..."
    ray status --address="$RAY_ADDRESS" || {
        echo "Error: Ray cluster not available at $RAY_ADDRESS"
        echo "Please start Ray cluster first:"
        echo "  ray start --head --node-ip-address \$MASTER_ADDR --num-gpus 8"
        exit 1
    }
}

# 函数：启动Orchestrator
launch_orchestrator() {
    echo "Launching Orchestrator..."
    
    ray job submit \
        --address="$RAY_ADDRESS" \
        --job-id="plan_act_orchestrator" \
        --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"$PYTHONPATH_EXTRA\"}}" \
        -- python3 examples/plan_act/orchestrator.py \
            --shared-namespace "$SHARED_NAMESPACE" \
            --coordination-mode "async" \
            --max-rollouts $NUM_ROLLOUTS \
            --plan-timeout $PLAN_TIMEOUT \
            --feedback-timeout $FEEDBACK_TIMEOUT
    
    echo "Orchestrator submitted (Job ID: plan_act_orchestrator)"
}

# 函数：启动Planner
launch_planner() {
    echo "Launching Planner Job..."
    
    ray job submit \
        --address="$RAY_ADDRESS" \
        --job-id="plan_act_planner" \
        --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"$PYTHONPATH_EXTRA\"}}" \
        -- python3 train.py \
            --job-role "planner" \
            --cross-job-namespace "$SHARED_NAMESPACE" \
            --rollout-function-path "examples.plan_act.planner:plan_generate_rollout" \
            --rollout-batch-size $ROLLOUT_BATCH_SIZE \
            --plan-batch-size $PLAN_BATCH_SIZE \
            --plan-timeout $PLAN_TIMEOUT \
            --debug-train-only \
            --hf-checkpoint "$PLANNER_MODEL" \
            --data-path "$TRAIN_DATA" \
            --rollout-global-dataset \
            --allow-random-init \
            --num-layers 2 \
            --hidden-size 128 \
            --num-attention-heads 8 \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 1 \
            --micro-batch-size 1 \
            --global-batch-size $ROLLOUT_BATCH_SIZE \
            --lr 1e-5 \
            --train-iters 1000 \
            --lr-decay-iters 1000 \
            --lr-warmup-iters 100 \
            --weight-decay 0.1 \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --init-method-std 0.006 \
            --clip-grad 1.0 \
            --bf16 \
            --log-interval 1 \
            --save-interval 100 \
            --eval-interval 100
    
    echo "Planner Job submitted (Job ID: plan_act_planner)"
}

# 函数：启动Actor
launch_actor() {
    echo "Launching Actor Job..."
    
    ray job submit \
        --address="$RAY_ADDRESS" \
        --job-id="plan_act_actor" \
        --runtime-env-json="{\"env_vars\": {\"PYTHONPATH\": \"$PYTHONPATH_EXTRA\"}}" \
        -- python3 train.py \
            --job-role "actor" \
            --cross-job-namespace "$SHARED_NAMESPACE" \
            --rollout-function-path "examples.plan_act.actor:actor_generate_rollout" \
            --custom-generate-function-path "examples.plan_act.actor:custom_actor_generate" \
            --actor-plan-timeout 60 \
            --rollout-batch-size $ROLLOUT_BATCH_SIZE \
            --rollout-num-gpus $ACTOR_NUM_GPUS \
            --rollout-num-gpus-per-engine 1 \
            --rollout-num-gpus-per-node 8 \
            --sglang-mem-fraction-static $SGLANG_MEM_FRACTION \
            --hf-checkpoint "$ACTOR_MODEL" \
            --ref-load "$REF_MODEL" \
            --data-path "$TRAIN_DATA" \
            --rollout-global-dataset \
            --num-layers 32 \
            --hidden-size 4096 \
            --num-attention-heads 32 \
            --tensor-model-parallel-size 4 \
            --pipeline-model-parallel-size 2 \
            --micro-batch-size 1 \
            --global-batch-size $ROLLOUT_BATCH_SIZE \
            --lr 1e-6 \
            --train-iters 1000 \
            --lr-decay-iters 1000 \
            --lr-warmup-iters 100 \
            --weight-decay 0.1 \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --init-method-std 0.006 \
            --clip-grad 1.0 \
            --bf16 \
            --log-interval 1 \
            --save-interval 100 \
            --eval-interval 100 \
            --kl-coef 0.1 \
            --rollout-temperature 0.7 \
            --rollout-top-p 0.9 \
            --rollout-max-response-len 512 \
            --n-samples-per-prompt 4
    
    echo "Actor Job submitted (Job ID: plan_act_actor)"
}

# 函数：监控Job状态
monitor_jobs() {
    echo ""
    echo "=== Monitoring Job Status ==="
    
    while true; do
        echo "$(date): Checking job status..."
        
        # 检查所有jobs状态
        echo "Ray Jobs Status:"
        ray job status --address="$RAY_ADDRESS" plan_act_orchestrator 2>/dev/null || echo "  Orchestrator: Not Found"
        ray job status --address="$RAY_ADDRESS" plan_act_planner 2>/dev/null || echo "  Planner: Not Found"
        ray job status --address="$RAY_ADDRESS" plan_act_actor 2>/dev/null || echo "  Actor: Not Found"
        
        echo "---"
        sleep 30
    done
}

# 函数：清理Jobs
cleanup_jobs() {
    echo ""
    echo "=== Cleaning up Jobs ==="
    
    echo "Stopping jobs..."
    ray job stop --address="$RAY_ADDRESS" plan_act_orchestrator 2>/dev/null || true
    ray job stop --address="$RAY_ADDRESS" plan_act_planner 2>/dev/null || true
    ray job stop --address="$RAY_ADDRESS" plan_act_actor 2>/dev/null || true
    
    sleep 5
    
    echo "Deleting jobs..."
    ray job delete --address="$RAY_ADDRESS" plan_act_orchestrator 2>/dev/null || true
    ray job delete --address="$RAY_ADDRESS" plan_act_planner 2>/dev/null || true
    ray job delete --address="$RAY_ADDRESS" plan_act_actor 2>/dev/null || true
    
    echo "Cleanup completed"
}

# 函数：显示日志
show_logs() {
    local job_id="$1"
    echo "=== Logs for $job_id ==="
    ray job logs --address="$RAY_ADDRESS" "$job_id" 2>/dev/null || echo "No logs available for $job_id"
    echo ""
}

# 函数：显示使用帮助
show_help() {
    cat << EOF
Plan/Act Architecture Launch Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start           Start all components (default)
    stop            Stop all components
    restart         Restart all components
    status          Show job status
    logs            Show logs for all jobs
    monitor         Monitor job status continuously
    cleanup         Clean up all jobs
    help            Show this help message

Environment Variables:
    SHARED_NAMESPACE        Shared namespace for communication (default: plan_act_shared)
    RAY_ADDRESS            Ray cluster address (default: http://127.0.0.1:8265)
    PLANNER_MODEL          Path to planner model
    ACTOR_MODEL            Path to actor model
    REF_MODEL              Path to reference model
    TRAIN_DATA             Path to training data
    ROLLOUT_BATCH_SIZE     Rollout batch size (default: 32)
    PLAN_BATCH_SIZE        Plan batch size (default: 64)
    ACTOR_NUM_GPUS         Number of GPUs for actor (default: 8)
    SGLANG_MEM_FRACTION    SGLang memory fraction (default: 0.7)

Examples:
    # Start with custom configuration
    ROLLOUT_BATCH_SIZE=64 ACTOR_NUM_GPUS=16 $0 start
    
    # Monitor running jobs
    $0 monitor
    
    # Check logs
    $0 logs
    
    # Clean restart
    $0 restart
EOF
}

# 信号处理
trap cleanup_jobs EXIT INT TERM

# 主逻辑
case "${1:-start}" in
    "start")
        check_ray_cluster
        echo "Starting Plan/Act architecture components..."
        launch_orchestrator
        sleep 10
        launch_planner
        sleep 10
        launch_actor
        echo ""
        echo "All components launched successfully!"
        echo "Use '$0 monitor' to monitor job status"
        echo "Use '$0 logs' to view logs"
        ;;
        
    "stop")
        cleanup_jobs
        ;;
        
    "restart")
        cleanup_jobs
        sleep 10
        check_ray_cluster
        launch_orchestrator
        sleep 10
        launch_planner
        sleep 10
        launch_actor
        echo "All components restarted successfully!"
        ;;
        
    "status")
        echo "=== Job Status ==="
        ray job status --address="$RAY_ADDRESS" plan_act_orchestrator
        ray job status --address="$RAY_ADDRESS" plan_act_planner
        ray job status --address="$RAY_ADDRESS" plan_act_actor
        ;;
        
    "logs")
        show_logs "plan_act_orchestrator"
        show_logs "plan_act_planner"
        show_logs "plan_act_actor"
        ;;
        
    "monitor")
        monitor_jobs
        ;;
        
    "cleanup")
        cleanup_jobs
        ;;
        
    "help"|"-h"|"--help")
        show_help
        ;;
        
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac