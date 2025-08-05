# Rank-to-Rank MoE 权重同步

本文档介绍在启用 `--enable-moe-rank2rank-sync` 后，slime 所创建的额外 NCCL 进程组。

## 背景

默认情况下，slime 会先将所有专家的参数聚合到 Pipeline Parallel (PP) 的源 rank，然后再统一广播给 SGLang 引擎。如果训练端与推理端的专家并行（EP）划分完全一致，可以直接将每个专家的权重发送到对应的 SGLang rank，避免这一额外的聚合步骤。

## 组拓扑

共有两类通信组：

1. **模型更新组** —— `slime-pp_{PP}`。组大小为 `rollout_num_gpus + 1`，包含 PP 源 rank 与所有 rollout 引擎 rank，用于广播非专家权重以及未启用 rank-to-rank 时的专家权重。
2. **Rank-to-Rank 专家组** —— `slime-pp_{PP}-r2r_{EP}`。每个专家 rank 都会建立一个同样大小的组，训练端仅 PP 源 rank 加入。更新权重时，源 rank 直接向对应的 rollout rank 广播自己持有的专家权重。

下图展示了在 EP=TP=4、仅一个 PP stage 时的关系：

```
         +---------+                       +------------+
         | PP src  | -- slime-pp_0 -->>    |  engines   |
         +---------+                       +------------+
              |                                  |
              +-- slime-pp_0-r2r_0 -------------->|
              +-- slime-pp_0-r2r_1 -------------->|
              +-- slime-pp_0-r2r_2 -------------->|
              +-- slime-pp_0-r2r_3 -------------->|
```

每个专家组均包含一个训练 rank 与所有 rollout rank，通信方向从训练端广播至各 rollout rank。

## 与 Ray 的集成

`UpdateWeightFromDistributed.connect_rollout_engines` 在训练进程与各 Ray rollout 引擎中分别创建这些 NCCL 进程组。Rollout actor 新增 `update_weights_from_distributed_rank2rank` 接口用于接收专家权重，当开启该选项时训练端会调用此接口。
