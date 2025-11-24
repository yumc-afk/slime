#!/bin/bash

# Plan/Act模式 - Planner启动脚本
# 负责启动大模型Planner job (70B配置)

set -e

# ===== 配置参数 =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Ray和环境配置
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-"/root/Megatron-LM/"}
SHARED_NAMESPACE=${SHARED_NAMESPACE:-"plan_act_shared"}

# 模型和数据路径
PLANNER_MODEL=${PLANNER_MODEL:-"/path/to/planner/70B/model"}
REF_MODEL=${REF_MODEL:-"/path/to/reference/model"}
TRAIN_DATA=${TRAIN_DATA:-"/path/to/train/data"}

# 70B模型配置参数
PLANNER_NUM_LAYERS=${PLANNER_NUM_LAYERS:-80}
PLANNER_HIDDEN_SIZE=${PLANNER_HIDDEN_SIZE:-8192}
PLANNER_FFN_HIDDEN_SIZE=${PLANNER_FFN_HIDDEN_SIZE:-28672}
PLANNER_NUM_ATTENTION_HEADS=${PLANNER_NUM_ATTENTION_HEADS:-64}
PLANNER_NUM_QUERY_GROUPS=${PLANNER_NUM_QUERY_GROUPS:-8}

# 并行策略配置 (70B模型典型配置)
PLANNER_TP=${PLANNER_TP:-8}  # Tensor Parallel
PLANNER_PP=${PLANNER_PP:-4}  # Pipeline Parallel
PLANNER_CP=${PLANNER_CP:-1}  # Context Parallel
PLANNER_EP=${PLANNER_EP:-1}  # Expert Parallel (非MoE模型)

# GPU和节点配置
PLANNER_NUM_NODES=${PLANNER_NUM_NODES:-4}
PLANNER_NUM_GPUS_PER_NODE=${PLANNER_NUM_GPUS_PER_NODE:-8}

# 训练参数
PLAN_BATCH_SIZE=${PLAN_BATCH_SIZE:-64}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
PLAN_TIMEOUT=${PLAN_TIMEOUT:-300}

# 学习率和优化器 (大模型通常使用较小学习率)
PLANNER_LR=${PLANNER_LR:-5e-7}
PLANNER_WEIGHT_DECAY=${PLANNER_WEIGHT_DECAY:-0.1}

# 内存配置
PLANNER_RECOMPUTE_LAYERS=${PLANNER_RECOMPUTE_LAYERS:-20}
PLANNER_MAX_TOKENS_PER_GPU=${PLANNER_MAX_TOKENS_PER_GPU:-2048}

# ===== 辅助函数 =====
show_help() {
    cat << EOF
Planner启动脚本 - Plan/Act架构的计划生成组件

Usage: $0 [OPTIONS]

关键环境变量:
    PLANNER_MODEL           Planner模型路径 (70B模型)
    REF_MODEL              参考模型路径
    TRAIN_DATA             训练数据路径
    SHARED_NAMESPACE       跨job通信命名空间 (default: plan_act_shared)

大模型配置:
    PLANNER_TP             Tensor并行度 (default: 8)
    PLANNER_PP             Pipeline并行度 (default: 4)  
    PLANNER_NUM_NODES      节点数 (default: 4)
    PLANNER_LR             学习率 (default: 5e-7)

性能调优:
    PLAN_BATCH_SIZE        计划批次大小 (default: 64)
    PLAN_TIMEOUT           计划超时时间 (default: 300s)
    PLANNER_RECOMPUTE_LAYERS  重计算层数 (default: 20)

示例:
    # 使用默认70B配置启动Planner
    PLANNER_MODEL=/data/Qwen2.5-70B-Instruct $0
    
    # 自定义并行配置
    PLANNER_TP=16 PLANNER_PP=2 $0
    
    # 多节点配置
    PLANNER_NUM_NODES=8 PLANNER_NUM_GPUS_PER_NODE=8 $0

注意事项:
    1. 需要先启动Ray集群
    2. 确保所有节点都能访问模型路径
    3. 70B模型需要大量GPU内存，建议使用H100/A100
    4. 与Actor job共享SHARED_NAMESPACE以实现通信
EOF
}

check_ray_cluster() {
    echo "检查Ray集群状态..."
    ray status --address="$RAY_ADDRESS" || {
        echo "错误: Ray集群未就绪 ($RAY_ADDRESS)"
        echo "请先启动Ray集群:"
        echo "  ray start --head --node-ip-address \$MASTER_ADDR --num-gpus 8"
        exit 1
    }
}

check_model_paths() {
    if [[ "$PLANNER_MODEL" == "/path/to/planner/70B/model" ]]; then
        echo "警告: 使用默认模型路径，请设置PLANNER_MODEL环境变量"
        echo "示例: PLANNER_MODEL=/data/Qwen2.5-70B-Instruct $0"
        exit 1
    fi
}

calculate_total_gpus() {
    local total_gpus=$((PLANNER_TP * PLANNER_CP * PLANNER_PP))
    local required_gpus=$((PLANNER_NUM_NODES * PLANNER_NUM_GPUS_PER_NODE))
    
    echo "GPU配置验证:"
    echo "  计算需求: TP($PLANNER_TP) × CP($PLANNER_CP) × PP($PLANNER_PP) = $total_gpus GPUs"
    echo "  可用资源: $PLANNER_NUM_NODES nodes × $PLANNER_NUM_GPUS_PER_NODE GPUs = $required_gpus GPUs"
    
    if [[ $total_gpus -gt $required_gpus ]]; then
        echo "错误: GPU需求($total_gpus)超过可用资源($required_gpus)"
        exit 1
    fi
}

# ===== 主启动逻辑 =====
launch_planner() {
    echo "=== 启动Planner (70B配置) ==="
    echo "模型路径: $PLANNER_MODEL"
    echo "共享命名空间: $SHARED_NAMESPACE"
    echo "并行配置: TP=$PLANNER_TP, PP=$PLANNER_PP, CP=$PLANNER_CP"
    echo "节点配置: $PLANNER_NUM_NODES nodes × $PLANNER_NUM_GPUS_PER_NODE GPUs"
    
    # 构建70B模型参数
    MODEL_ARGS=(
        --swiglu
        --num-layers $PLANNER_NUM_LAYERS
        --hidden-size $PLANNER_HIDDEN_SIZE
        --ffn-hidden-size $PLANNER_FFN_HIDDEN_SIZE
        --num-attention-heads $PLANNER_NUM_ATTENTION_HEADS
        --group-query-attention
        --num-query-groups $PLANNER_NUM_QUERY_GROUPS
        --use-rotary-position-embeddings
        --disable-bias-linear
        --add-qkv-bias
        --normalization "RMSNorm"
        --norm-epsilon 1e-06
        --rotary-base 1000000
        --vocab-size 152064
        --untie-embeddings-and-output-weights
    )
    
    # 并行配置
    PERF_ARGS=(
        --tensor-model-parallel-size $PLANNER_TP
        --pipeline-model-parallel-size $PLANNER_PP
        --context-parallel-size $PLANNER_CP
        --expert-model-parallel-size $PLANNER_EP
        --sequence-parallel
        
        # 大模型内存优化
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers $PLANNER_RECOMPUTE_LAYERS
        
        # 动态批处理
        --use-dynamic-batch-size
        --max-tokens-per-gpu $PLANNER_MAX_TOKENS_PER_GPU
    )
    
    # 优化器配置
    OPTIMIZER_ARGS=(
        --optimizer adam
        --lr $PLANNER_LR
        --lr-decay-style constant
        --weight-decay $PLANNER_WEIGHT_DECAY
        --adam-beta1 0.9
        --adam-beta2 0.95
        --clip-grad 1.0
        
        # 大模型优化器优化
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
        --use-precision-aware-optimizer
    )
    
    # Planner特定参数
    PLANNER_ARGS=(
        --job-role "planner"
        --cross-job-namespace "$SHARED_NAMESPACE"
        --rollout-function-path "examples.plan_act.planner:plan_generate_rollout"
        
        # 计划生成参数
        --plan-batch-size $PLAN_BATCH_SIZE
        --rollout-batch-size $ROLLOUT_BATCH_SIZE
        --plan-timeout $PLAN_TIMEOUT
        --plan-base-temperature 0.7
        --plan-base-top-p 0.9
        --plan-feedback-window 10
    )
    
    # 其他配置
    MISC_ARGS=(
        --bf16
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        
        # 训练控制
        --train-iters 10000
        --lr-decay-iters 10000
        --lr-warmup-iters 500
        --log-interval 10
        --save-interval 1000
        --eval-interval 500
    )
    
    # 检测NVLink
    NVLINK_COUNT=$(nvidia-smi 2>/dev/null | grep -o "NVLink" | wc -l || echo "0")
    HAS_NVLINK=$([[ $NVLINK_COUNT -gt 0 ]] && echo "1" || echo "0")
    echo "NVLink检测: $HAS_NVLINK (发现 $NVLINK_COUNT 个NVLink引用)"
    
    # 构建运行时环境
    RUNTIME_ENV_JSON="{
        \"env_vars\": {
            \"PYTHONPATH\": \"$PYTHONPATH_EXTRA\",
            \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
            \"NCCL_NVLS_ENABLE\": \"$HAS_NVLINK\",
            \"PYTHONBUFFERED\": \"16\"
        }
    }"
    
    # 提交Ray job
    ray job submit \
        --address="$RAY_ADDRESS" \
        --job-id="plan_act_planner" \
        --runtime-env-json="$RUNTIME_ENV_JSON" \
        -- python3 train.py \
        --actor-num-nodes $PLANNER_NUM_NODES \
        --actor-num-gpus-per-node $PLANNER_NUM_GPUS_PER_NODE \
        --hf-checkpoint "$PLANNER_MODEL" \
        --ref-load "$REF_MODEL" \
        --data-path "$TRAIN_DATA" \
        --rollout-global-dataset \
        --micro-batch-size 1 \
        --global-batch-size $PLAN_BATCH_SIZE \
        "${MODEL_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${PLANNER_ARGS[@]}" \
        "${MISC_ARGS[@]}"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Planner job已成功提交 (Job ID: plan_act_planner)"
        echo ""
        echo "监控命令:"
        echo "  ray job status --address=\"$RAY_ADDRESS\" plan_act_planner"
        echo "  ray job logs --address=\"$RAY_ADDRESS\" plan_act_planner"
        echo ""
        echo "下一步: 启动Actor job"
        echo "  ./launch_actor.sh"
    else
        echo "❌ Planner job提交失败"
        exit 1
    fi
}

# ===== 命令行处理 =====
case "${1:-start}" in
    "start")
        check_model_paths
        check_ray_cluster
        calculate_total_gpus
        launch_planner
        ;;
    "stop")
        echo "停止Planner job..."
        ray job stop --address="$RAY_ADDRESS" plan_act_planner 2>/dev/null || true
        ray job delete --address="$RAY_ADDRESS" plan_act_planner 2>/dev/null || true
        echo "Planner已停止"
        ;;
    "status")
        ray job status --address="$RAY_ADDRESS" plan_act_planner
        ;;
    "logs")
        ray job logs --address="$RAY_ADDRESS" plan_act_planner
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "未知命令: $1"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac