"""
Ray跨Job通信实现示例

这些示例展示了在Ray环境中实现跨job通信的几种方案，
特别针对slime这样的分布式训练框架的需求。
"""

import ray
import time
import json
import pickle
import torch
import redis
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

# =============================================================================
# 方案A: Object Store共享机制
# =============================================================================

@ray.remote
class ObjectStoreDataBroker:
    """
    基于Ray Object Store的跨job数据代理
    
    核心思想：利用Ray object store在同一集群所有job间共享的特性，
    通过detached named actor来管理ObjectRef的生命周期。
    """
    
    def __init__(self):
        self.data_refs: Dict[str, ray.ObjectRef] = {}
        self.metadata: Dict[str, Dict] = {}
        print(f"ObjectStoreDataBroker initialized at {time.time()}")
    
    def store_data(self, key: str, data: Any, metadata: Dict = None) -> ray.ObjectRef:
        """存储数据到object store并记录引用"""
        # 将数据存储到object store
        data_ref = ray.put(data)
        self.data_refs[key] = data_ref
        
        # 存储元数据
        self.metadata[key] = {
            'timestamp': time.time(),
            'data_size': len(pickle.dumps(data)) if hasattr(data, '__len__') else 0,
            'metadata': metadata or {}
        }
        
        print(f"Stored data with key: {key}")
        return data_ref
    
    def get_data_ref(self, key: str) -> Optional[ray.ObjectRef]:
        """获取数据引用"""
        return self.data_refs.get(key)
    
    def get_data(self, key: str) -> Any:
        """直接获取数据"""
        data_ref = self.data_refs.get(key)
        if data_ref is None:
            return None
        return ray.get(data_ref)
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """列出所有可用的key"""
        if prefix:
            return [k for k in self.data_refs.keys() if k.startswith(prefix)]
        return list(self.data_refs.keys())
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """获取数据元数据"""
        return self.metadata.get(key)
    
    def cleanup_expired(self, max_age_seconds: int = 3600):
        """清理过期数据"""
        current_time = time.time()
        expired_keys = []
        
        for key, meta in self.metadata.items():
            if current_time - meta['timestamp'] > max_age_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.data_refs[key]
            del self.metadata[key]
            print(f"Cleaned up expired data: {key}")

def setup_object_store_broker(namespace: str = "default") -> ray.actor.ActorHandle:
    """设置Object Store数据代理"""
    # 确保在正确的namespace中
    if not ray.is_initialized():
        ray.init(address="auto", namespace=namespace)
    
    try:
        # 尝试获取已存在的代理
        broker = ray.get_actor("data_broker")
        print("Found existing data broker")
    except ValueError:
        # 创建新的detached代理
        broker = ObjectStoreDataBroker.options(
            name="data_broker",
            lifetime="detached",  # 确保跨job存活
            max_restarts=-1      # 无限重启
        ).remote()
        print("Created new data broker")
    
    return broker

# 使用示例：Job A (数据生产者)
def example_producer_job():
    """数据生产者job示例"""
    broker = setup_object_store_broker("slime_shared")
    
    # 模拟rollout数据
    for rollout_id in range(5):
        rollout_data = {
            'tokens': torch.randint(0, 1000, (32, 512)),
            'rewards': torch.randn(32),
            'response_lengths': torch.randint(10, 100, (32,))
        }
        
        key = f"rollout_{rollout_id}"
        data_ref = ray.get(broker.store_data.remote(
            key, 
            rollout_data,
            metadata={'job_type': 'training', 'model': 'qwen3-4B'}
        ))
        
        print(f"Produced rollout {rollout_id}")
        time.sleep(1)

# 使用示例：Job B (数据消费者) 
def example_consumer_job():
    """数据消费者job示例"""
    broker = setup_object_store_broker("slime_shared")
    
    # 等待数据可用
    while True:
        available_keys = ray.get(broker.list_keys.remote("rollout_"))
        if available_keys:
            break
        print("Waiting for data...")
        time.sleep(2)
    
    # 消费数据
    for key in available_keys:
        data = ray.get(broker.get_data.remote(key))
        metadata = ray.get(broker.get_metadata.remote(key))
        
        print(f"Consumed {key}: {metadata}")
        # 处理数据...

# =============================================================================
# 方案B: 文件系统共享机制  
# =============================================================================

class FileSystemCommunicator:
    """
    基于共享文件系统的跨job通信
    
    适用于大数据集或需要持久化的场景
    """
    
    def __init__(self, base_path: str = "/tmp/ray_shared_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_path / "metadata.json"
        
        # 初始化元数据
        if not self.metadata_path.exists():
            self._save_metadata({})
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict):
        """保存元数据"""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def write_data(self, key: str, data: Any, metadata: Dict = None):
        """写入数据到文件系统"""
        file_path = self.base_path / f"{key}.pt"
        
        # 保存数据
        torch.save(data, file_path)
        
        # 更新元数据
        all_metadata = self._load_metadata()
        all_metadata[key] = {
            'file_path': str(file_path),
            'timestamp': time.time(),
            'size_bytes': file_path.stat().st_size,
            'metadata': metadata or {}
        }
        self._save_metadata(all_metadata)
        
        print(f"Wrote data to {file_path}")
    
    def read_data(self, key: str) -> Optional[Any]:
        """从文件系统读取数据"""
        file_path = self.base_path / f"{key}.pt"
        if file_path.exists():
            return torch.load(file_path)
        return None
    
    def list_available_keys(self) -> List[str]:
        """列出所有可用的key"""
        metadata = self._load_metadata()
        return list(metadata.keys())
    
    def wait_for_key(self, key: str, timeout: int = 60) -> bool:
        """等待特定key的数据可用"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if key in self.list_available_keys():
                return True
            time.sleep(1)
        return False

# 使用示例
def example_filesystem_producer():
    """文件系统生产者示例"""
    comm = FileSystemCommunicator("/shared/slime_data")
    
    for i in range(3):
        data = {
            'model_state': torch.randn(1000, 1000),  # 大型模型状态
            'training_metrics': {'loss': 0.5 - i * 0.1, 'accuracy': 0.8 + i * 0.05}
        }
        
        comm.write_data(
            f"checkpoint_{i}",
            data,
            metadata={'epoch': i, 'model': 'llama-7b'}
        )
        time.sleep(5)

def example_filesystem_consumer():
    """文件系统消费者示例"""
    comm = FileSystemCommunicator("/shared/slime_data")
    
    # 等待数据
    for i in range(3):
        key = f"checkpoint_{i}"
        if comm.wait_for_key(key):
            data = comm.read_data(key)
            print(f"Loaded checkpoint {i}: {data['training_metrics']}")

# =============================================================================
# 方案C: Redis外部存储机制
# =============================================================================

class RedisCommunicator:
    """
    基于Redis的跨job通信机制
    
    提供pub/sub、数据存储、生命周期管理等高级功能
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.metadata_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        
        # 测试连接
        try:
            self.redis_client.ping()
            print("Connected to Redis successfully")
        except redis.ConnectionError:
            raise ConnectionError("Failed to connect to Redis")
    
    def publish_data(self, channel: str, data: Any, ttl: int = 3600) -> str:
        """发布数据到指定频道"""
        message_id = f"{channel}_{int(time.time() * 1000)}"
        
        # 序列化并存储数据
        serialized_data = pickle.dumps(data)
        self.redis_client.setex(f"data:{message_id}", ttl, serialized_data)
        
        # 发布元数据通知
        metadata = {
            "message_id": message_id,
            "data_size": len(serialized_data),
            "timestamp": time.time(),
            "ttl": ttl
        }
        
        self.metadata_client.publish(channel, json.dumps(metadata))
        print(f"Published data to channel {channel}: {message_id}")
        return message_id
    
    def subscribe_channel(self, channel: str, callback: Callable[[Any, Dict], None]):
        """订阅频道并处理消息"""
        pubsub = self.metadata_client.pubsub()
        pubsub.subscribe(channel)
        
        print(f"Subscribed to channel: {channel}")
        
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        metadata = json.loads(message['data'])
                        message_id = metadata['message_id']
                        
                        # 获取实际数据
                        data = self.get_data(message_id)
                        if data is not None:
                            callback(data, metadata)
                    except Exception as e:
                        print(f"Error processing message: {e}")
        finally:
            pubsub.close()
    
    def get_data(self, message_id: str) -> Optional[Any]:
        """根据消息ID获取数据"""
        serialized_data = self.redis_client.get(f"data:{message_id}")
        if serialized_data:
            return pickle.loads(serialized_data)
        return None
    
    def store_persistent_data(self, key: str, data: Any, metadata: Dict = None):
        """存储持久化数据"""
        # 存储数据
        serialized_data = pickle.dumps(data)
        self.redis_client.set(f"persistent:{key}", serialized_data)
        
        # 存储元数据
        if metadata:
            self.metadata_client.hset(f"meta:{key}", mapping={
                'timestamp': time.time(),
                'size': len(serialized_data),
                **metadata
            })
    
    def get_persistent_data(self, key: str) -> Optional[Any]:
        """获取持久化数据"""
        serialized_data = self.redis_client.get(f"persistent:{key}")
        if serialized_data:
            return pickle.loads(serialized_data)
        return None
    
    def list_persistent_keys(self, pattern: str = "*") -> List[str]:
        """列出持久化数据的key"""
        keys = self.redis_client.keys(f"persistent:{pattern}")
        return [key.decode().replace("persistent:", "") for key in keys]

# 使用示例
def example_redis_producer():
    """Redis生产者示例"""
    comm = RedisCommunicator()
    
    # 发布实时数据
    for i in range(5):
        rollout_data = {
            'rollout_id': i,
            'rewards': [0.1 * j for j in range(10)],
            'policy_loss': 0.05 - i * 0.01
        }
        
        comm.publish_data("training_rollouts", rollout_data)
        time.sleep(2)
    
    # 存储持久化checkpoint
    checkpoint = {
        'model_weights': torch.randn(100, 100),
        'optimizer_state': {'lr': 0.001, 'momentum': 0.9}
    }
    
    comm.store_persistent_data(
        "final_checkpoint",
        checkpoint,
        metadata={'epoch': 100, 'loss': 0.01}
    )

def example_redis_consumer():
    """Redis消费者示例"""
    comm = RedisCommunicator()
    
    def handle_rollout_data(data, metadata):
        print(f"Received rollout data: {data['rollout_id']}")
        print(f"Metadata: {metadata}")
        # 处理rollout数据...
    
    # 订阅实时数据
    comm.subscribe_channel("training_rollouts", handle_rollout_data)

# =============================================================================
# slime特定的集成示例
# =============================================================================

@dataclass
class SlimeRolloutData:
    """slime rollout数据结构"""
    rollout_id: int
    tokens: torch.Tensor
    rewards: torch.Tensor
    response_lengths: torch.Tensor
    metadata: Dict

class SlimeCrossJobManager:
    """
    slime特定的跨job通信管理器
    
    结合slime现有的Box机制，提供无缝的跨job数据共享
    """
    
    def __init__(self, communication_method: str = "object_store", **kwargs):
        self.method = communication_method
        
        if communication_method == "object_store":
            self.broker = setup_object_store_broker(kwargs.get("namespace", "slime"))
        elif communication_method == "redis":
            self.comm = RedisCommunicator(**kwargs)
        elif communication_method == "filesystem":
            self.comm = FileSystemCommunicator(kwargs.get("base_path", "/tmp/slime_shared"))
        else:
            raise ValueError(f"Unknown communication method: {communication_method}")
    
    def share_rollout_data(self, rollout_id: int, data: SlimeRolloutData, job_id: str = None):
        """共享rollout数据到其他job"""
        key = f"{job_id or 'default'}_rollout_{rollout_id}"
        
        if self.method == "object_store":
            ray.get(self.broker.store_data.remote(key, data))
        elif self.method == "redis":
            self.comm.publish_data("slime_rollouts", {
                'key': key,
                'data': data
            })
        elif self.method == "filesystem":
            self.comm.write_data(key, data)
    
    def get_rollout_data(self, rollout_id: int, job_id: str = None) -> Optional[SlimeRolloutData]:
        """从其他job获取rollout数据"""
        key = f"{job_id or 'default'}_rollout_{rollout_id}"
        
        if self.method == "object_store":
            return ray.get(self.broker.get_data.remote(key))
        elif self.method == "redis":
            return self.comm.get_persistent_data(key)
        elif self.method == "filesystem":
            return self.comm.read_data(key)
    
    def list_available_rollouts(self, job_id: str = None) -> List[str]:
        """列出可用的rollout数据"""
        prefix = f"{job_id or 'default'}_rollout_" if job_id else "rollout_"
        
        if self.method == "object_store":
            return ray.get(self.broker.list_keys.remote(prefix))
        elif self.method == "redis":
            return self.comm.list_persistent_keys(f"*{prefix}*")
        elif self.method == "filesystem":
            keys = self.comm.list_available_keys()
            return [k for k in keys if prefix in k]

# 集成到slime Buffer的示例修改
class EnhancedSlimeBuffer:
    """
    增强的slime Buffer，支持跨job通信
    
    这是对现有slime/ray/buffer.py的概念性扩展
    """
    
    def __init__(self, args, wandb_run_id, enable_cross_job=False):
        # ... 现有slime Buffer初始化代码 ...
        
        if enable_cross_job:
            self.cross_job_manager = SlimeCrossJobManager(
                communication_method=getattr(args, 'cross_job_method', 'object_store'),
                namespace=getattr(args, 'cross_job_namespace', 'slime_shared')
            )
            self.job_id = getattr(args, 'job_id', 'default')
    
    def generate_with_sharing(self, rollout_id, evaluation=False):
        """生成数据并支持跨job共享"""
        # 现有的generate逻辑...
        # data = self.generate_rollout(...)
        
        # 如果启用跨job共享且不是evaluation
        if hasattr(self, 'cross_job_manager') and not evaluation:
            rollout_data = SlimeRolloutData(
                rollout_id=rollout_id,
                tokens=torch.tensor([]),  # 实际数据
                rewards=torch.tensor([]),
                response_lengths=torch.tensor([]),
                metadata={'job_id': self.job_id, 'timestamp': time.time()}
            )
            
            self.cross_job_manager.share_rollout_data(rollout_id, rollout_data, self.job_id)
        
        # 返回Box包装的数据引用
        # return Box(ray.put(data))

if __name__ == "__main__":
    # 运行示例
    print("Ray Cross-Job Communication Examples")
    print("Choose an example to run:")
    print("1. Object Store Producer")
    print("2. Object Store Consumer") 
    print("3. Filesystem Example")
    print("4. Redis Example")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        example_producer_job()
    elif choice == "2":
        example_consumer_job()
    elif choice == "3":
        example_filesystem_producer()
    elif choice == "4":
        example_redis_producer()
    else:
        print("Invalid choice")