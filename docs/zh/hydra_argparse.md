# Hydra 与 argparse 的整合

[English](../en/hydra_argparse.md)

本文总结了在 slime 项目中关于参数解析的几次讨论，主要内容包括：

- 如何在使用 Hydra 的同时复用现有依赖 `argparse` 的代码；
- 迁移到 Hydra/Hybra 配置方式的大致实施路径；
- 参数在 Megatron 与 sglang 之间的分发机制；
- DeepSeek‑v3 RL 实验的启动流程以及参数如何传递给训练脚本。

## 1. 从 Hydra 配置生成 argparse 对象

训练脚本 `train.py` 当前依赖 `slime.utils.arguments.parse_args()` 解析命令行参数，内部调用 Megatron 的解析器。因此若在外部使用 Hydra，只需将 `DictConfig` 转换成 `argparse.Namespace` 即可：

```python
from hydra import main
from omegaconf import OmegaConf
import argparse

def hydra_to_namespace(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return argparse.Namespace(**cfg_dict)

@main(config_path="conf", config_name="config")
def hydra_entry(cfg):
    args = hydra_to_namespace(cfg)
    train(args)
```

这样便能在保持现有接口的同时，通过 YAML 配置和命令行覆盖获得更灵活的管理方式。

## 2. 迁移到 Hydra 的实施建议

若希望整体改用 "Hybra" 的写法，可按以下步骤进行：

1. **引入依赖**：在 `requirements.txt` 中添加 `hydra-core`。
2. **准备默认配置**：在 `conf/` 目录编写 `config.yaml`，将现有参数按 Megatron、SGLang、slime 自身等分类整理。
3. **改写入口脚本**：在 `train.py` 使用 `@hydra.main` 读取配置，再转换成 `argparse.Namespace`，保持与 Megatron 的接口兼容。
4. **替换 `slime.utils.arguments`**：重写该模块，使其从 Hydra 配置构造 `Namespace` 后继续调用 Megatron 的初始化逻辑。
5. **调整启动脚本与文档**：原来的 `bash` 启动脚本需要改为接受 Hydra 的 `--config-name` 或 `+key=value` 覆盖方式，文档也需相应更新。

## 3. 参数在 Megatron 与 sglang 之间的分发

`train.py` 在解析参数时同时注册了 Megatron 与 sglang 的选项。sglang 的选项统一带有 `--sglang-` 前缀，仅在推理相关逻辑中使用，而其他普通参数则传递给 Megatron。具体流程如下：

1. `scripts/run-deepseek-r1.sh` 等启动脚本将各类参数分成若干数组，最后统一传入 `python3 train.py`。
2. `slime.utils.arguments.parse_args()` 调用 Megatron 的 `parse_args`，并通过 `add_sglang_arguments` 加入带前缀的选项。
3. 在 `RolloutRayActor` 中，创建 `SglangEngine` 时只取 `args.sglang_*` 相关字段，而训练部分的 `MegatronTrainRayActor` 仅依赖普通的 Megatron 参数。

因此，两套系统共享同一个 `args` 对象，但通过前缀区分各自的配置，互不冲突。

## 4. DeepSeek‑v3 RL 实验的启动与参数传递

脚本 `scripts/run-deepseek-r1.sh` 演示了完整的启动流程。其关键片段如下：

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/DeepSeek-R1/
   --ref-load $BASE_DIR/DeepSeek-R1_torch_dist/
   --load $BASE_DIR/DeepSeek-R1_slime/
   --save $BASE_DIR/DeepSeek-R1_slime/
   --save-interval 20
)
# 省略若干数组定义
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{ ... }' \
   -- python3 train.py \
   --actor-num-nodes 16 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
```

脚本首先从 `scripts/models/deepseek-v3.sh` 读取模型配置，再按类别组装参数数组，最终通过 Ray 启动 `train.py`。`train.py` 解析这些参数后，将普通参数传给 Megatron 训练逻辑，将带有 `--sglang-` 的参数用于构建推理引擎。由此实现单脚本同时启动训练与推理。

---

以上即为本次讨论的要点汇总，供团队成员参考与实施。
