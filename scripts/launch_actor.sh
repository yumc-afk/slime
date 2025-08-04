#!/bin/bash

# Plan/Act模式 - Actor启动脚本
# 负责启动小模型Actor job (7B配置)

set -e

# ===== 配置参数 =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Ray和环境配置
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-"/root/Megatron-LM/"}
SHARED_NAMESPACE=${SHARED_NAMESPACE:-"plan_act_shared"}

# 模型和数据路径
ACTOR_MODEL=${ACTOR_MODEL:-"/path/to/actor/7B/model"}
REF_MODEL=${REF_MODEL:-"/path/to/reference/model"}
TRAIN_DATA=${TRAIN_DATA:-"/path/to/train/data"}

# 7B模型配置参数
ACTOR_NUM_LAYERS=${ACTOR_NUM_LAYERS:-28}
ACTOR_HIDDEN_SIZE=${ACTOR_HIDDEN_SIZE:-3584}
ACTOR_FFN_HIDDEN_SIZE=${ACTOR_FFN_HIDDEN_SIZE:-18944}
ACTOR_NUM_ATTENTION_HEADS=${ACTOR_NUM_ATTENTION_HEADS:-28}
ACTOR_NUM_QUERY_GROUPS=${ACTOR_NUM_QUERY_GROUPS:-4}

# 并行策略配置 (7B模型典型配置)
ACTOR_TP=${ACTOR_TP:-2}  # Tensor Parallel
ACTOR_PP=${ACTOR_PP:-1}  # Pipeline Parallel  
ACTOR_CP=${ACTOR_CP:-1}  # Context Parallel
ACTOR_EP=${ACTOR_EP:-1}  # Expert Parallel (非MoE模型)

# GPU和节点配置
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}

# SGLang rollout配置
ACTOR_ROLLOUT_NUM_GPUS=${ACTOR_ROLLOUT_NUM_GPUS:-8}
ACTOR_ROLLOUT_GPUS_PER_ENGINE=${ACTOR_ROLLOUT_GPUS_PER_ENGINE:-2}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.7}

# 训练参数
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
ACTOR_PLAN_TIMEOUT=${ACTOR_PLAN_TIMEOUT:-60}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}

# 学习率和优化器 (小模型可以使用相对较大的学习率)
ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-0.1}

# 内存配置
ACTOR_RECOMPUTE_LAYERS=${ACTOR_RECOMPUTE_LAYERS:-4}
ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-8192}

# 生成参数
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.9}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}

# ===== 辅助函数 =====
show_help() {
    cat << EOF
Actor启动脚本 - Plan/Act架构的执行组件

Usage: $0 [OPTIONS]

关键环境变量:
    ACTOR_MODEL            Actor模型路径 (7B模型)
    REF_MODEL              参考模型路径
    TRAIN_DATA             训练数据路径
    SHARED_NAMESPACE       跨job通信命名空间 (default: plan_act_shared)

小模型配置:
    ACTOR_TP               Tensor并行度 (default: 2)
    ACTOR_PP               Pipeline并行度 (default: 1)
    ACTOR_NUM_NODES        节点数 (default: 1)
    ACTOR_LR               学习率 (default: 1e-6)

Rollout配置:
    ACTOR_ROLLOUT_NUM_GPUS        Rollout使用的GPU数 (default: 8)
    ACTOR_ROLLOUT_GPUS_PER_ENGINE 每个SGLang引擎的GPU数 (default: 2)
    SGLANG_MEM_FRACTION           SGLang内存分配比例 (default: 0.7)

性能调优:
    ROLLOUT_BATCH_SIZE     Rollout批次大小 (default: 32)
    ACTOR_PLAN_TIMEOUT     等待计划超时时间 (default: 60s)
    N_SAMPLES_PER_PROMPT   每个prompt的样本数 (default: 4)

示例:
    # 使用默认7B配置启动Actor
    ACTOR_MODEL=/data/Qwen2.5-7B-Instruct $0
    
    # 自定义rollout配置
    ACTOR_ROLLOUT_NUM_GPUS=16 ACTOR_ROLLOUT_GPUS_PER_ENGINE=4 $0
    
    # 调整生成参数
    ROLLOUT_TEMPERATURE=0.7 ROLLOUT_MAX_RESPONSE_LEN=2048 $0

注意事项:
    1. 需要先启动Ray集群和Planner job
    2. 确保与Planner使用相同的SHARED_NAMESPACE
    3. SGLang引擎会自动处理推理加速
    4. 支持降级到标准rollout（无计划时）
EOF
}

check_ray_cluster() {
    echo "检查Ray集群状态..."
    ray status --address="$RAY_ADDRESS" || {
        echo "错误: Ray集群未就绪 ($RAY_ADDRESS)"
        echo "请先启动Ray集群"
        exit 1
    }
}

check_planner_job() {
    echo "检查Planner job状态..."
    local planner_status=$(ray job status --address="$RAY_ADDRESS" plan_act_planner 2>/dev/null | grep -o "RUNNING\|SUCCEEDED\|PENDING" || echo "NOT_FOUND")
    
    if [[ "$planner_status" == "NOT_FOUND" ]]; then
        echo "警告: 未找到Planner job，Actor将以降级模式运行"
        echo "建议先启动Planner: ./launch_planner.sh"
    else
        echo "Planner job状态: $planner_status"
    fi
}

check_model_paths() {
    if [[ "$ACTOR_MODEL" == "/path/to/actor/7B/model" ]]; then
        echo "警告: 使用默认模型路径，请设置ACTOR_MODEL环境变量"
        echo "示例: ACTOR_MODEL=/data/Qwen2.5-7B-Instruct $0"
        exit 1
    fi
}

calculate_total_gpus() {
    local training_gpus=$((ACTOR_TP * ACTOR_CP * ACTOR_PP))
    local rollout_gpus=$ACTOR_ROLLOUT_NUM_GPUS
    local total_required=$((training_gpus > rollout_gpus ? training_gpus : rollout_gpus))
    local available_gpus=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
    
    echo "GPU配置验证:"
    echo "  训练需求: TP($ACTOR_TP) × CP($ACTOR_CP) × PP($ACTOR_PP) = $training_gpus GPUs"
    echo "  Rollout需求: $rollout_gpus GPUs"
    echo "  总需求: $total_required GPUs"
    echo "  可用资源: $ACTOR_NUM_NODES nodes × $ACTOR_NUM_GPUS_PER_NODE GPUs = $available_gpus GPUs"
    
    if [[ $total_required -gt $available_gpus ]]; then
        echo "错误: GPU需求($total_required)超过可用资源($available_gpus)"
        exit 1
    fi
}

# ===== 主启动逻辑 =====
launch_actor() {
    echo "=== 启动Actor (7B配置) ==="
    echo "模型路径: $ACTOR_MODEL"
    echo "共享命名空间: $SHARED_NAMESPACE"
    echo "并行配置: TP=$ACTOR_TP, PP=$ACTOR_PP, CP=$ACTOR_CP"
    echo "Rollout配置: $ACTOR_ROLLOUT_NUM_GPUS GPUs, $ACTOR_ROLLOUT_GPUS_PER_ENGINE GPUs/engine"
    
    # 构建7B模型参数
    MODEL_ARGS=(
        --swiglu
        --num-layers $ACTOR_NUM_LAYERS
        --hidden-size $ACTOR_HIDDEN_SIZE
        --ffn-hidden-size $ACTOR_FFN_HIDDEN_SIZE
        --num-attention-heads $ACTOR_NUM_ATTENTION_HEADS
        --group-query-attention
        --num-query-groups $ACTOR_NUM_QUERY_GROUPS
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
        --tensor-model-parallel-size $ACTOR_TP
        --pipeline-model-parallel-size $ACTOR_PP
        --context-parallel-size $ACTOR_CP
        --expert-model-parallel-size $ACTOR_EP
        --sequence-parallel
        
        # 小模型内存优化（相对轻量）
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers $ACTOR_RECOMPUTE_LAYERS
        
        # 动态批处理
        --use-dynamic-batch-size
        --max-tokens-per-gpu $ACTOR_MAX_TOKENS_PER_GPU
    )
    
    # 优化器配置
    OPTIMIZER_ARGS=(
        --optimizer adam
        --lr $ACTOR_LR
        --lr-decay-style constant
        --weight-decay $ACTOR_WEIGHT_DECAY
        --adam-beta1 0.9
        --adam-beta2 0.95
        --clip-grad 1.0
    )
    
    # Actor特定参数
    ACTOR_ARGS=(
        --job-role "actor"
        --cross-job-namespace "$SHARED_NAMESPACE"
        --rollout-function-path "examples.plan_act.actor:actor_generate_rollout"
        --custom-generate-function-path "examples.plan_act.actor:custom_actor_generate"
        
        # Plan/Act协调参数
        --actor-plan-timeout $ACTOR_PLAN_TIMEOUT
        --rollout-batch-size $ROLLOUT_BATCH_SIZE
        --n-samples-per-prompt $N_SAMPLES_PER_PROMPT
        
        # 执行历史管理
        --actor-max-history 100
    )
    
    # Rollout配置
    ROLLOUT_ARGS=(
        --rollout-num-gpus $ACTOR_ROLLOUT_NUM_GPUS
        --rollout-num-gpus-per-engine $ACTOR_ROLLOUT_GPUS_PER_ENGINE
        --rollout-num-gpus-per-node $ACTOR_NUM_GPUS_PER_NODE
        
        # 生成参数
        --rollout-temperature $ROLLOUT_TEMPERATURE
        --rollout-top-p $ROLLOUT_TOP_P
        --rollout-max-response-len $ROLLOUT_MAX_RESPONSE_LEN
        
        # 数据和奖励
        --rollout-shuffle
        --rm-type deepscaler
        --balance-data
    )
    
    # SGLang配置
    SGLANG_ARGS=(
        --sglang-mem-fraction-static $SGLANG_MEM_FRACTION
    )
    
    # GRPO算法配置
    GRPO_ARGS=(
        --advantage-estimator grpo
        --use-kl-loss
        --kl-coef 0.1
        --kl-loss-type low_var_kl
        --entropy-coef 0.0
        --eps-clip 0.2
        --eps-clip-high 0.28
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
        --lr-warmup-iters 200
        --log-interval 5
        --save-interval 500
        --eval-interval 100
        
        # Colocate模式 (训练和推理在同一GPU)
        --colocate
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
        --job-id="plan_act_actor" \
        --runtime-env-json="$RUNTIME_ENV_JSON" \
        -- python3 train.py \
        --actor-num-nodes $ACTOR_NUM_NODES \
        --actor-num-gpus-per-node $ACTOR_NUM_GPUS_PER_NODE \
        --hf-checkpoint "$ACTOR_MODEL" \
        --ref-load "$REF_MODEL" \
        --data-path "$TRAIN_DATA" \
        --rollout-global-dataset \
        --micro-batch-size 1 \
        --global-batch-size $ROLLOUT_BATCH_SIZE \
        "${MODEL_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${ACTOR_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${MISC_ARGS[@]}"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Actor job已成功提交 (Job ID: plan_act_actor)"
        echo ""
        echo "监控命令:"
        echo "  ray job status --address=\"$RAY_ADDRESS\" plan_act_actor"
        echo "  ray job logs --address=\"$RAY_ADDRESS\" plan_act_actor"
        echo ""
        echo "Plan/Act架构已完全启动!"
        echo "监控整体状态:"
        echo "  ./monitor_plan_act.sh"
    else
        echo "❌ Actor job提交失败"
        exit 1
    fi
}

# ===== 命令行处理 =====
case "${1:-start}" in
    "start")
        check_model_paths
        check_ray_cluster
        check_planner_job
        calculate_total_gpus
        launch_actor
        ;;
    "stop")
        echo "停止Actor job..."
        ray job stop --address="$RAY_ADDRESS" plan_act_actor 2>/dev/null || true
        ray job delete --address="$RAY_ADDRESS" plan_act_actor 2>/dev/null || true
        echo "Actor已停止"
        ;;
    "status")
        ray job status --address="$RAY_ADDRESS" plan_act_actor
        ;;
    "logs")
        ray job logs --address="$RAY_ADDRESS" plan_act_actor
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