# slime 讨论总结

本文档整理了之前对 slime 框架的相关讨论，供团队成员学习与实现参考。

## 1. 官方镜像与训练数据

官方 Docker 镜像只预装了 SGLang 与 Megatron 等依赖，并未包含模型权重或训练数据，需要用户自行下载。文档中的示例下载命令如下：

```bash
# hf checkpoint
huggingface-cli download THUDM/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

# train data
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# eval data
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

以上内容摘自 `docs/zh/models/glm4-9B.md`【F:docs/zh/models/glm4-9B.md†L20-L28】。

## 2. 训练中进行评测

`train.py` 在训练循环中根据 `--eval-interval` 触发评测，会调用 `actor_model.async_eval`：

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    if args.eval_interval is not None and rollout_id == 0:
        eval_rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id, evaluation=True))
        ray.get(actor_model.async_eval(rollout_id, eval_rollout_data_ref))

    ...

    if args.eval_interval is not None and (
        (rollout_id + 1) % args.eval_interval == 0
        or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
    ):
        eval_rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id, evaluation=True))
        ray.get(actor_model.async_eval(rollout_id, eval_rollout_data_ref))
```

代码位于【F:train.py†L45-L79】。

相应的评测参数示例可在文档中找到，例如：

```bash
EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)
```

摘自 `docs/zh/models/glm4-9B.md`【F:docs/zh/models/glm4-9B.md†L122-L133】。

## 3. 监控指标

W&B 初始化时定义了各类指标名称：

```python
def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
```

见【F:slime/utils/wandb_utils.py†L52-L62】。

在 `log_rollout_data`、`log_eval_data` 等函数中会具体记录日志。例如：

```python
# 部分 log_rollout_data 逻辑
print(f"rollout {rollout_id}: {reduced_log_dict}")
if args.use_wandb:
    reduced_log_dict["rollout/step"] = (
        rollout_id if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    wandb.log(reduced_log_dict)
```

节选自【F:slime/backends/megatron_utils/data.py†L171-L219】。

`log_eval_data` 会计算评测平均 reward 及截断率并输出到 W&B：

```python
log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
if "truncated" in data[key]:
    log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
print(f"eval {rollout_id}: {log_dict}")
if args.use_wandb:
    log_dict["eval/step"] = (
        rollout_id if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    wandb.log(log_dict)
```

节选自【F:slime/backends/megatron_utils/data.py†L354-L378】。

此外还有 `log_passrate` 等函数记录 pass@k 等指标。

综上，slime 通过 W&B 记录以下主要指标：

- `train/*`：训练损失、学习率等训练相关指标；
- `rollout/*`：rollout 数据的平均 reward 等；
- `multi_turn/*`：多轮对话统计信息；
- `passrate/*`：评测数据集的 pass@k；
- `eval/*`：评测集平均 reward 及截断率；
- `perf/*`：训练耗时、TFLOPS 等性能数据。

这些指标可用于在训练过程中持续监控模型效果与性能。
