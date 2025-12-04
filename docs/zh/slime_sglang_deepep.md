# SGLang 与 deepep 简介

本文档总结了 slime 在使用 SGLang 和 deepep 时需要了解的要点，方便团队成员参考。

## deepep 原理与 NVSHMEM 调用

在 SGLang 的补丁中，`DeepEPBuffer` 模块会创建多条 NVSHMEM 通信队列（`num_qps_per_rank=20`），以在多卡环境下共享专家权重或 KV 数据。训练脚本会设置 `LD_LIBRARY_PATH`，并将 NVSHMEM 的库路径加入其中：

```
"LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/sgl-workspace/nvshmem/install/lib/"
```

这样 deepep 就能依赖 NVSHMEM 提供的高速共享内存接口，在多 GPU 间高效同步和调度专家。

## 环境配置

官方提供了包含所有依赖的 Docker 镜像。如果需要自行搭建环境，需在安装 slime 后，对 SGLang 代码应用仓库自带的 `sglang.patch`。启用 deepep 时常见的参数如下：

```
# enable deepep for sglang
--sglang-enable-deepep-moe
--sglang-deepep-mode auto

# use deepep for megatron
--moe-enable-deepep
--moe-token-dispatcher-type flex
```

训练脚本中的 `LD_LIBRARY_PATH` 设置已包含 NVSHMEM 的安装目录，因此无需手动安装额外库。

## slime 调用 SGLang 的方式

slime 通过 `HttpServerEngineAdapter` 封装了基于 SGLang 的服务，并使用 `sglang-router` 进行负载均衡。所有 SGLang server 启动后会向 router 注册，路由器提供 OpenAI-compatible API：

- server 启动后通过 `/add_worker` 向 router 报到；
- 生成数据时，只需向 router 发送 HTTP 请求（如 `/generate`），router 会将请求转发给相应的 server；
- 通过 `--sglang-router-ip` 与 `--sglang-router-port` 可以让 slime 注册到外部 router。

因此，slime 可以像调用普通 Chat Completion 服务一样，通过 HTTP 接口访问 SGLang，模型加载与并行等参数都以 `--sglang-` 前缀传入。
