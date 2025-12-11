# Integrating Hydra with argparse

[中文](../zh/hydra_argparse.md)

This note summarizes several discussions around parameter parsing in the slime project:

- How to keep using existing `argparse` based code while adopting Hydra.
- An outline for migrating to a Hydra/Hybra configuration style.
- How parameters are split between Megatron and sglang.
- How the DeepSeek‑v3 RL experiment is launched and how options reach Megatron.

## 1. Converting Hydra config to an argparse object

`train.py` currently relies on `slime.utils.arguments.parse_args()` which wraps Megatron's parser. If you run the program under Hydra, convert the `DictConfig` into an `argparse.Namespace` first:

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

This preserves Megatron's expected interface while letting you manage settings via YAML and command‑line overrides.

## 2. Suggested steps for a full Hydra migration

1. **Add the dependency**: include `hydra-core` in `requirements.txt`.
2. **Prepare default configs**: create `conf/config.yaml` and categorize existing arguments (Megatron, SGLang, slime, etc.).
3. **Rewrite the entry script**: decorate `train.py` with `@hydra.main`, then convert the config into an `argparse.Namespace` for downstream code.
4. **Replace `slime.utils.arguments`**: rebuild this module so it constructs the namespace from Hydra configs before invoking Megatron's initialization routines.
5. **Adjust launch scripts and docs**: update bash scripts to pass `--config-name` or `+key=value` overrides, and document the new workflow.

## 3. Passing parameters to Megatron and sglang

Both Megatron and sglang options are registered in the same parser. sglang flags use the `--sglang-` prefix and are only consumed by inference components:

1. Launch scripts such as `scripts/run-deepseek-r1.sh` build arrays of arguments and pass them all to `python3 train.py`.
2. `slime.utils.arguments.parse_args()` calls Megatron's parser and extends it via `add_sglang_arguments`.
3. The `RolloutRayActor` constructs a `SglangEngine` using only the `args.sglang_*` fields, while `MegatronTrainRayActor` reads the standard Megatron arguments.

Thus the same `args` object is shared but each subsystem picks out its own values by prefix.

## 4. Launching the DeepSeek‑v3 RL experiment

`scripts/run-deepseek-r1.sh` demonstrates the full launch procedure. Key lines are:

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/DeepSeek-R1/
   --ref-load $BASE_DIR/DeepSeek-R1_torch_dist/
   --load $BASE_DIR/DeepSeek-R1_slime/
   --save $BASE_DIR/DeepSeek-R1_slime/
   --save-interval 20
)
# many arrays omitted
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

The script sources `scripts/models/deepseek-v3.sh` to load model settings, assembles argument arrays, and finally invokes `train.py` via a Ray job. `train.py` parses all options, forwarding generic ones to Megatron and the `--sglang-*` subset to the inference engine. This allows training and inference to launch together from one script.
## 5. Driving bash scripts with Hydra

Sometimes we prefer not to change the existing shell framework at all. You can keep `scripts/run-qwen3-235B-A22B.sh` untouched and use Hydra only to build the argument arrays it expects. A sample config lives in `conf/run_qwen3_235B_A22B.yaml`. Generate the arrays with:

```bash
python3 tools/generate_args.py --config-name run_qwen3_235B_A22B > args.sh
source args.sh
```

The helper prints variables like `CKPT_ARGS=( ... )`. Once sourced, the original script can be executed normally and receives the same arguments as before.


---

This document captures the main points from our discussion for future reference.
