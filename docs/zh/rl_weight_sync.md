# slime RL 训练与权重同步解析

本文整理了内部关于 slime 的一些讨论，归纳了训练与推理的协同方式、权重量化与同步的顺序，以及在大规模 EP 推理时的通信拓扑，供团队学习。

## 训推一体与分离的代码复用

在训练端始终使用同一个 `MegatronTrainRayActor`。初始化时会根据是否传入 `--colocate` 创建不同的 `weight_updator`：

```python
update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
self.weight_updator = update_weight_cls(...)
```

`UpdateWeightFromTensor`（训推一体）和 `UpdateWeightFromDistributed`（训推分离）均位于 `update_weight_utils.py`，并实现同样的接口：

```python
class UpdateWeightFromTensor:
    def connect_rollout_engines(...):
        ...
    def update_weights(...):
        ...

class UpdateWeightFromDistributed:
    def connect_rollout_engines(...):
        ...
    def update_weights(...):
        ...
```

两者共享 `all_gather_param`、`named_parameters` 等辅助函数，因此除通信方式（同卡 IPC vs NCCL 广播）外，逻辑完全复用。
训练端只与 `weight_updator` 交互（`connect_rollout_engines`、`update_weights`），其余流程保持一致。

## 异步训练的 Rollout 与 Train 错开

在启用训推分离且使用 `train_async.py` 时，slime 会在训练当前 rollout 的同时生成下一个 rollout，二者相差一个步骤：

```python
rollout_data_next_future = rollout_manager.async_generate(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.async_generate(rollout_id + 1)
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

这样可以并行利用训练 GPU 和推理 GPU，提高整体效率。

## 权重聚合与量化顺序

在 `update_weight_utils.py` 中，从 Megatron 到 HuggingFace 的权重同步流程如下：

1. 使用 `all_gather` 将 TP/EP 分片的参数在训练端聚合；
2. 调用 `convert_to_hf` 将张量转换为 HF 格式，并根据 `quantization_config` 进行量化；
3. 将量化后的张量通过 IPC 或 NCCL 传给推理端。

因此量化发生在 `all_gather` 之后、传输之前。

在 FP8 rollout 场景下，权重会在训练端先完成量化，再传输给推理端加载。

## 大规模 EP 下的通信拓扑

- **训推一体 (`--colocate`)**：`UpdateWeightFromTensor` 直接在 GPU 内收集参数并通过 IPC 句柄传给 SGLang；
- **训推分离**：`UpdateWeightFromDistributed` 会在训练 rank0 与各 SGLang 服务器之间建立额外的 NCCL 进程组，先 `all_gather` 所有分片，再 `broadcast` 给所有服务器；
- **推理端配置**：可通过 `--sglang-enable-ep-moe`、`--sglang-enable-dp-attention`、`--sglang-dp-size` 等参数在推理服务器上支持如 EP64、DP8 等大规模 MoE。`--rollout-num-gpus-per-engine` 决定每个 rollout engine 的 GPU 数，对应 SGLang 的 `tp_size`。

在这种拓扑下，训练 rank0 收集并量化所有并行分片的权重，通过相应方式同步给各个 SGLang 服务器，后者再加载到模型中。

