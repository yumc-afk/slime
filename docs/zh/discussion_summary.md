# Qwen3-235B 训练脚本与 checkpoint 转换说明

本文档整理自团队讨论，主要涵盖以下两个主题：

1. 项目是否提供 Qwen3‑235B 的训练脚本；
2. slime 是否可以直接通过 mbridge 加载 Hugging Face 格式的 checkpoint。

## 1. Qwen3‑235B 训练脚本

仓库在 `scripts` 目录下提供了 Qwen3‑235B 的训练脚本。示例脚本名称如下：

```bash
$ ls scripts | grep qwen3-235
run-qwen3-235B-A22B-sft.sh
run-qwen3-235B-A22B.sh
```

在 `run-qwen3-235B-A22B.sh` 中，可看到加载和保存 235B 模型的参数设置（节选）：

```bash
source "${SCRIPT_DIR}/models/qwen3-235B-A22B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3-235B-A22B
   --ref-load ${BASE_FOLDER}/Qwen3-235B-A22B_torch_dist
   --load ${BASE_FOLDER}/Qwen3-235B-A22B_slime/
   --save ${BASE_FOLDER}/Qwen3-235B-A22B_slime/
   --save-interval 20
)
```

这些脚本展示了如何加载 HF 模型、Megatron 的 `torch_dist` 格式以及 slime 自身的保存路径，便于在此基础上自定义训练流程。

## 2. checkpoint 转换与 mbridge

slime 底层依赖 Megatron，Megatron 无法直接加载 Hugging Face checkpoint。项目文档在 [README](../../README_zh.md) 中说明，需要先将 HF checkpoint 转换为 `torch_dist` 格式：

```bash
cd slime/
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

目前 slime 不支持直接通过 mbridge 加载 HF checkpoint，因此仍需先进行格式转换后再训练。文档中也提醒，使用 mbridge 转换得到的 `torch_dist` checkpoint 由于不保存 `args`，无法直接再转回 HF 格式。

