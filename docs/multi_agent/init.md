Plan/Act 双代理训练架构设计报告
背景与目标
在大语言模型的强化学习后训练（RLHF）过程中，某些任务需要规划与执行分离：一个代理负责分析任务和制定步骤（Planner），另一个代理根据 Planner 制定的指令逐步完成任务（Actor）。这种 Plan/Act 架构可以在复杂任务（例如文档检索、程序设计或工具使用）中提升效率。我们希望在 slime 框架上实现这样一个双代理训练流程，其中：
- Planner 作为主代理：负责全局决策，包括多轮规划和调整流程。Planner 的输出类似于单一 Actor 的决策，但内容是下一步要执行的指令或子任务。
- Actor 作为执行者：根据 Planner 的指令与环境交互或调用工具完成子任务，并将执行结果反馈给 Planner。
- 奖励共享：根据整个任务最终效果计算一个标量奖励，分配给 Planner 与 Actor 的所有交互轨迹，用于强化训练。
为了快速实现，我们重点关注可以复用 slime 架构和 SGLang 运行时的方式，使得实现复杂度最小化。
架构概述
slime 基本组成
slime 是一套面向 LLM 的 RLHF 训练框架，它将推理引擎与训练器解耦，核心流程如下：
1. train.py 通过 Ray 创建 actor group 和 rollout manager，并在训练循环中先调用 rollout_manager.async_generate 生成训练样本，再调用 actor_model.async_train 更新模型参数[1]。
2. 推理端 RolloutRayActor 调用 SglangEngine 通过 HTTP 与 SGLang 服务器交互，实现模型推理和 KV 缓存管理[2]。
3. 数据缓冲区 Buffer 负责加载数据集和保存生成的轨迹，并通过 rollout_function_path 动态加载自定义的 generate_rollout 函数[3]。
4. Sample 数据结构包含 prompt、response、reward 和 metadata 字段[4]，metadata 可存储自定义信息（如 agent_id、子任务编号）。
Plan/Act 多轮交互模式
在传统 slime 流程中，每个生成样本都是一个独立的单轮对话。为实现 Plan/Act 模式，需要支持多轮交互和动态调整：
1. 多轮规划：Planner 根据任务提示生成计划（例如列出要查询的关键词或分解的步骤），发送给 Actor 执行；根据 Actor 的反馈再迭代调整计划，直至任务完成或达到最大轮次。
2. 全局控制：Planner 负责判断何时结束任务以及如何调整步骤，因此其地位类似单一 Actor，只不过输出内容是指令而非最终答案。这也意味着Planner与环境交互次数较多，需要动态利用历史对话上下文。
3. 共享奖励：在整个多轮流程结束后，根据任务完成质量计算统一奖励，写入 Planner 和 Actor 的所有样本中。
典型使用场景
Plan/Act 模式特别适合需要“拆解－执行”的任务，例如：
- 文档检索与问答：用户提问“比较三篇论文”，Planner 先确定检索范围和阅读顺序，让 Actor 去查找并返回摘要；Planner 根据汇总信息撰写最终答案。
- 复杂编程任务：Planner 分解需求并指示 Actor 实现模块，Actor 写出代码或测试用例；Planner 检查输出并反馈修改意见。
- 工具调度：Planner 判断需要调用哪些外部工具或 API，以及调用顺序，Actor 执行具体调用并返回结果；Planner 综合结果决定下一步。
- 文字游戏或逻辑谜题：Planner 制定策略或猜测下一步行动，Actor 与环境交互并返回状态，Planner 据此更新策略。
这些场景具有共同特点：Planner 掌控全局、制定多轮计划；Actor 只是执行者；最终奖励根据整体表现来评估。
实现路径 1：单实例自定义 generate_rollout（推荐）
slime 支持通过 --rollout-function-path 动态加载自定义的轨迹生成函数[3]。因此，可以在单个 slime 训练进程中实现 Plan/Act 双代理逻辑，具体步骤如下：
1. 编写 plan_act_rollout.py：实现如下签名的函数：
from slime.utils.types import Sample

def generate_rollout(args, rollout_id, data_buffer, evaluation=False) -> list[list[Sample]]:
    """
    在单个 slime 实例中实现多轮 Planner/Actor 交互。
    返回样本列表，每个样本包含 Planner 或 Actor 的输出。
    """
1. 函数内部逻辑：
2. 调用 data_buffer.get_samples(...) 获取用户的任务或 prompt；
3. 创建 Planner 的 Sample，调用 SGLang（通过 slime/backends/sglang_utils/sglang_engine.py 中的接口）生成 <think>... 区块中的计划，并将 agent_id='planner'、plan_steps 等记录在 metadata；
4. 解析计划为多个子任务，将每个子任务作为新的 prompt 创建 Actor 的 Sample，设置 agent_id='actor'；调用 SGLang 生成具体执行结果并写入 response；
5. 如果需要多轮交互，根据 Actor 的反馈更新 Planner 的上下文，再次调用 Planner 生成新的计划，并继续上述流程，直至任务完成或达到 args.max_turns；
6. 计算整体奖励（可调用 reward model 或自定义规则），将奖励值写入 Planner 和 Actor 所有样本的 reward 字段；
7. 返回 [[planner_sample] + actor_samples]（或多轮样本组合）。
8. 训练流程：在启动 slime 训练时指定 --rollout-function-path=/path/to/plan_act_rollout.py，slime 会使用我们的 Plan/Act 逻辑生成样本并训练单个模型。由于只有一个模型，这里 Planner 和 Actor 共享参数，但通过 metadata['agent_id'] 区分角色，可以在训练策略上适当调整（例如为 Planner 加大奖励权重）。
9. 多轮支持与 KV 缓存复用：在生成过程中，应保证每轮调用都携带相同的系统提示和历史上下文，使得 SGLang 的 RadixAttention 可以自动复用 KV 缓存[5]。使用 <think> 标记或 reasoning parser 能让模型输出清晰的计划，便于解析。
优点
- 无需改动 slime 内核，实现简洁；
- Planner 与 Actor 的逻辑完全在 generate_rollout 中控制，易于调试；
- 利用 SGLang KV 缓存自动复用前缀，加速多轮生成。
限制
- 只有一套模型参数，无法真正做到“两个独立模型”；
- 如果需求中 Planner 和 Actor 需要不同的策略网络或超参，则无法满足。
实现路径 2：双实例 + 调度器
若必须让 Planner 与 Actor 拥有独立的 slime 实例和独立参数，可以通过外部调度器协调两个训练进程：
1. 分别启动两个 slime 服务：
2. Planner slime 使用普通 generate_rollout，其任务是生成计划；
3. Actor slime 负责根据任务 prompt 执行操作。
4. 编写调度器（Ray Client 或 HTTP Service）：每轮任务调用顺序如下：
5. 调度器调用 Planner slime 的 rollout_manager.async_generate 生成计划；
6. 解析计划为子任务，将子任务送入 Actor slime 的 rollout_manager.async_generate 生成执行结果；
7. 将执行结果拼接到 Planner 的上下文中，再调用 Planner slime 更新计划或生成最终答案；
8. 根据整个流程计算奖励，然后通过 actor_model.async_train 接口分别向两个 slime 实例发送奖励和样本。
9. 确保 KV 缓存共享：两个 slime 服务需使用同一个 SGLang 路由器地址，调度器在交替调用两者时传入相同的 session_id（或采用相同系统 prompt）以复用 RadixAttention 缓存。否则缓存无法共享，推理效率下降。
优点
- Planner 与 Actor 完全独立，可针对不同角色采用不同模型或优化器；
- 调度器可以根据需要调整执行顺序、轮次数、奖励分配策略。
挑战
- 需要新增接口让 slime 暴露单步生成和单步训练的方法（当前 train.py 封装了完整循环），调度器必须自行管理训练节奏；
- 数据与模型更新需要在外部同步，复杂度较高；
- 调度器需要解析 Planner 输出和 Actor 输入，并在多轮对话中维护会话状态。
代码实现建议
1. 自定义 generate_rollout 模板：可参考 slime/rollout/sglang_rollout.py 的结构，实现异步调用 SGLang 及奖励模型的逻辑。注意在返回 Sample 时正确填充 tokens、response_length 和 metadata[6]。
2. 使用 metadata 区分角色：为 Planner 和 Actor 的 Sample 添加 metadata['agent_id'] = 'planner' 或 'actor'，后续可据此在训练器中实现不同的奖励缩放或损失权重。
3. 奖励函数实现：可以使用现有的 reward model 调用（如 async_rm、batched_async_rm），也可以自定义奖励规则并将值赋给 Sample.reward。Buffer 会自动将 reward 转为训练数据[7]。
4. 多轮迭代：在生成过程中维护一个对话状态列表，每次调用 Planner 或 Actor 时将历史消息拼接到 prompt 中；SGLang 会自动复用 KV 缓存，提高效率。
总结
本报告提出了在 slime 框架中实现 Plan/Act 双代理训练的两种方案：
- 方案一：单实例自定义 generate_rollout。这是最简单、最快实现的方式，通过在自定义生成函数中实现 Planner/Actor 的多轮逻辑、奖励共享即可满足需求，且无需修改 slime 内核。
- 方案二：双实例 + 外部调度器。该方案适合需要完全独立的模型或更灵活的调度逻辑的场景，但需要额外开发调度服务并扩展 slime 接口，实施成本较高。
对于大多数任务，推荐从单实例方案入手验证方法有效性：Planner 作为主进程掌控全局，通过多轮计划和反馈驱动 Actor 执行，最终根据整体表现给予统一奖励。若后续需要扩展到异构模型或更复杂的协作方式，再考虑引入外部调度器。
[1] raw.githubusercontent.com
https://raw.githubusercontent.com/THUDM/slime/main/train.py
[2] raw.githubusercontent.com
https://raw.githubusercontent.com/THUDM/slime/main/slime/ray/rollout.py
[3] [7] raw.githubusercontent.com
https://raw.githubusercontent.com/THUDM/slime/main/slime/ray/buffer.py
[4] raw.githubusercontent.com
https://raw.githubusercontent.com/THUDM/slime/main/slime/utils/types.py
[5] [6] raw.githubusercontent.com
https://raw.githubusercontent.com/THUDM/slime/main/slime/rollout/sglang_rollout.py
