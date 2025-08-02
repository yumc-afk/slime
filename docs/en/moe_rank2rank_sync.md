# Rank-to-Rank MoE Weight Synchronization

This document describes the additional NCCL process groups created when
`--enable-moe-rank2rank-sync` is enabled for Mixture-of-Experts (MoE)
models.

## Background

slime typically aggregates the parameters of all expert ranks to the
pipeline-parallel (PP) source rank before broadcasting them to SGLang
engines. When the training and inference sides use the same expert
parallelism (EP) layout, we can avoid this extra gather by sending each
expert's weights directly to the matching SGLang rank.

## Group Topology

Two kinds of communication groups are involved:

1. **Model update group** – `slime-pp_{PP}`. This group has a world
   size of `rollout_num_gpus + 1` and contains the PP source rank and
   all rollout engine ranks. It is used to broadcast non‑expert weights
   and expert weights when rank‑to‑rank sync is disabled.
2. **Rank‑to‑rank expert group** – `slime-pp_{PP}-r2r_{EP}`. For each
   expert rank, a dedicated group of the same world size is created. Only
   the PP source rank participates on the training side. During weight
   updates, the source rank broadcasts its local experts directly to the
   corresponding rollout ranks.

The following diagram illustrates the relation when there are four
rollout GPUs (EP=TP=4) and a single PP stage:

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

Each expert group contains one training rank and all rollout ranks. The
broadcast direction is from the training rank to every rollout rank.

## Ray Integration

`UpdateWeightFromDistributed.connect_rollout_engines` builds these NCCL
process groups on both the training process and each Ray rollout engine.
The rollout actor exposes `update_weights_from_distributed_rank2rank`
for receiving expert weights, while the training side calls this method
when the new option is enabled.
