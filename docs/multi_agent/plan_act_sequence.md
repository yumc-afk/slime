# Plan/Act 执行时序详解

## 1. 启动阶段时序

```mermaid
sequenceDiagram
    participant U as User
    participant R as Ray Cluster
    participant P as Planner Job
    participant O as Orchestrator
    participant A as Actor Job
    
    U->>R: ray job submit (planner)
    R->>P: 启动train.py --agent-role planner
    P->>P: 解析参数，检测enable_plan_act
    P->>O: 创建PlanActOrchestrator(detached)
    O->>O: 初始化session管理器
    P->>P: 继续正常训练初始化
    
    U->>R: ray job submit (actor)  
    R->>A: 启动train.py --agent-role actor
    A->>A: 解析参数，检测enable_plan_act
    A->>O: ray.get_actor("plan_act_orchestrator")
    Note over A: 如果找不到会重试60次
    A->>A: 继续正常训练初始化
```

## 2. Rollout生成阶段时序

```mermaid
sequenceDiagram
    participant PT as Planner train.py
    participant PB as Planner Buffer
    participant PR as plan_act_rollout()
    participant O as Orchestrator
    participant AR as plan_act_rollout()
    participant AB as Actor Buffer
    participant AT as Actor train.py
    
    Note over PT,AT: 训练循环中的rollout_id=N
    
    PT->>PB: rollout_manager.async_generate(N)
    PB->>PR: generate(N, data_buffer)
    PR->>O: execute_rollout(N, "planner", buffer_ref)
    
    AT->>AB: rollout_manager.async_generate(N)
    AB->>AR: generate(N, data_buffer)
    AR->>O: execute_rollout(N, "actor", buffer_ref)
    
    Note over O: 同步栅栏：等待两个agent都到达
    
    O->>O: session.planner_data ✓
    O->>O: session.actor_data ✓
    O->>O: if both ready and not executed
    O->>O: 开始_orchestrate_plan_act_loop()
```

## 3. Plan/Act协调执行时序

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant PB as Planner Buffer
    participant SG1 as SGLang (Planner)
    participant AB as Actor Buffer  
    participant SG2 as SGLang (Actor)
    
    loop 每个turn (最多max_turns)
        O->>PB: buffer.get_samples(batch_size)
        PB-->>O: 返回任务样本
        
        Note over O: Plan阶段
        O->>SG1: 生成计划(通过Buffer的机制)
        SG1-->>O: 返回计划样本
        
        O->>O: _parse_concurrent_acts(plan)
        Note over O: 解析出可并发的任务
        
        Note over O: Act阶段（并发）
        par 并发执行多个Act
            O->>SG2: execute_act_1
            O->>SG2: execute_act_2
            O->>SG2: execute_act_3
        end
        SG2-->>O: 返回所有执行结果
        
        O->>O: _update_context(results)
        O->>O: _is_task_complete()?
    end
    
    O->>O: _prepare_results_for_agents()
```

## 4. 结果返回阶段时序

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant PR as Planner rollout
    participant PB as Planner Buffer
    participant PT as Planner train.py
    participant AR as Actor rollout
    participant AB as Actor Buffer
    participant AT as Actor train.py
    
    Note over O: 执行完成，准备返回结果
    
    O->>O: session.result_future.set_result(samples)
    
    par 两个agent同时获取结果
        O-->>PR: await result_future
        PR-->>PB: 返回samples
        PB-->>PT: Box(ray.put(data))
        
        O-->>AR: await result_future  
        AR-->>AB: 返回samples
        AB-->>AT: Box(ray.put(data))
    end
    
    Note over PT,AT: 继续正常的训练流程
    PT->>PT: actor_model.async_train(N, data)
    AT->>AT: actor_model.async_train(N, data)
```

## 5. 异常处理时序

### 5.1 超时处理

```mermaid
sequenceDiagram
    participant R as plan_act_rollout
    participant O as Orchestrator
    participant F as Fallback
    
    R->>O: execute_rollout(timeout=1800)
    
    alt 正常完成
        O-->>R: 返回结果
    else 超时
        O--xR: GetTimeoutError
        R->>R: 检查fallback_on_timeout
        alt 启用降级
            R->>F: 调用标准generate_rollout
            F-->>R: 返回标准结果
        else 不降级
            R->>R: 抛出超时异常
        end
    end
```

### 5.2 Agent崩溃处理

```mermaid
sequenceDiagram
    participant P as Planner
    participant O as Orchestrator
    participant A as Actor
    
    P->>O: execute_rollout("planner")
    Note over O: Planner已注册
    
    A->>O: execute_rollout("actor")
    
    alt Actor正常
        O->>O: 两个都ready，执行
    else Actor崩溃
        Note over O: 只有Planner注册
        O->>O: 等待...
        Note over P: 最终超时
        P--xO: 超时错误
    end
```

## 6. 关键数据流

### 6.1 Sample数据流转

```
用户任务(Buffer) 
    ↓
Planner生成计划(Sample格式)
    ↓
Orchestrator解析和分发
    ↓
Actor执行任务(Sample格式)
    ↓
Orchestrator整合所有Sample
    ↓
返回给两个Job用于训练
```

### 6.2 Session状态管理

```python
RolloutSession {
    rollout_id: int
    planner_data: {arrived_at, buffer_ref}
    actor_data: {arrived_at, buffer_ref}
    result_future: asyncio.Future
    executed: bool
    created_at: float
}
```

## 7. 性能特征

### 7.1 延迟分析

- **启动延迟**：Actor等待Orchestrator创建，最多5分钟
- **同步延迟**：取决于两个job的执行速度差异
- **通信延迟**：Ray actor调用，通常<10ms
- **总体开销**：相比独立训练，增加5-10%的时间

### 7.2 并发特性

- **Rollout级并发**：不同rollout_id可以并发处理
- **Turn内并发**：多个Act可以并发执行
- **Job间独立**：两个job的其他操作（如梯度计算）完全独立

## 8. 调试要点

### 8.1 日志位置

```bash
# Orchestrator日志
ray logs actor --name plan_act_orchestrator

# Job日志
ray job logs <job_id>

# Ray系统日志
/tmp/ray/session_latest/logs/
```

### 8.2 关键检查点

1. **Orchestrator创建**：查看Planner日志确认创建成功
2. **Agent注册**：Orchestrator日志显示两个agent到达
3. **同步等待**：检查是否有agent未到达导致阻塞
4. **结果返回**：确认samples格式正确