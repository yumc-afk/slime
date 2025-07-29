# slime Router 与异步训练概览

本文档整理了近期讨论中涉及的几个关键话题，供团队成员学习与实现时参考。

## Router 与 Prefix Cache

- slime 启动训练时会使用 [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) 管理推理进程；若未显式指定 `--sglang-router-ip` 与 `--sglang-router-port`，框架会自动在集群中启动一个 Router。
- 每个 SGLang server 启动后通过 `/add_worker` 接口向 Router 注册；推理请求统一发送至 Router 的 `/generate` 接口，由 Router 负责负载均衡。
- `rollout.py` 在启动 Router 时将 `balance_abs_threshold` 设为 `0`，使 Router 始终处于 "cache-aware" 模式。这意味着 Router 会维护各 worker 的前缀树，根据请求前缀匹配度选择最合适的 worker，从而尽可能命中 prefix cache。
- 只有当负载差异超过 `balance_abs_threshold` 与 `balance_rel_threshold` 条件时，Router 才会切换到基于队列长度的负载均衡策略。

## 多个 SGLang 服务器的协作

- slime 在内部或外部启动 Router 后，会将所有 SGLang server 注册到该 Router 上。
- 训练和生成阶段均通过 Router 转发请求，实现多实例的负载均衡与缓存复用。
- 若通过命令行指定外部 Router 地址，slime 不再启动内部 Router，而是直接向外部 Router 注册所有 server。

## 多轮对话中的 Prefix Cache 使用

- 在带有环境交互的任务（如 `Search-R1 lite`）中，每轮对话都会将当前完整的历史前缀连同最新回复一并发送给 `/generate` 接口。
- SGLang 会自动对相同前缀进行缓存，无需显式维护会话 ID；只要在下一轮请求中带上之前的文本，即可复用此前的计算结果。
- 在训练过程中更新模型权重前，会调用 `reset_prefix_cache`（最终触发 `llm.flush_cache()`）清空旧的前缀缓存，确保新权重生效。

## 与环境交互的训练流程

- slime 通过自定义 `generate` 与 `reward` 函数将环境逻辑嵌入训练流程，`examples/search-r1/generate_with_search.py` 提供了参考实现。
- 该函数内部使用 `asyncio` 与信号量控制并发，既向 SGLang 服务器请求生成，也向外部环境（如 Google Search API）发送异步请求，形成多轮对话直至得到 `<answer>`。
- 在 `train_async.py` 中，slime 使用 Ray 管理分布式训练，并通过异步调用 (`async_generate`、`async_train`) 将数据生成与模型训练错开执行，实现训练与推理的并行。

## Agent 训练支持

- 框架并未封装统一的 Agent Loop 接口，而是通过插件形式扩展，如 `slime_plugins/rollout_buffer`。
- 该插件提供基于 FastAPI 的 HTTP 服务，可将 slime 启动的 LLM Server 与外部 Agent 框架连接，异步生成智能体轨迹。
- 不同任务可继承 `base_generator.py` 实现自定义的 `Generator`，Rollout Buffer 会动态加载并执行对应逻辑。

