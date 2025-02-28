import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
import yaml
import os
from dataclasses import dataclass, asdict, fields, field

torch.set_printoptions(threshold=np.inf)


class DatasetType(Enum):	
    NORMAL = "normal"
    SPACE_PREFIX = "space_prefix"
    REVERSE = "reverse"
    

@dataclass
class Config:
    # 词表设置
    pad_token: str = "@"
    eos_token: str = "#"
    sos_token: str = "$"
    digits: str = "0123456789"
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    operators: List[str] = field(default_factory=lambda: ["+", "*"])
    relation_operators: List[str] = field(default_factory=lambda: ["=", "!=", ">", "<", ">=", "<="])
    logical_operators: List[str] = field(default_factory=lambda: ["&", "|", "~"])
    saparator: List[str] = field(default_factory=lambda: ["(", ")", "[", "]", "{", "}", ",", ":", ".", ";", " "])
    
    # 两个操作数的位数范围
    min_digits: int = 1
    max_digits: int = 3  # 扩展位数范围
    
    # 模型规模设置
    max_seq_len: int = 25
    d_model: int = 1024  
    nhead: int = 16
    num_layers: int = 24  
    
    # 训练超参
    batch_size: int = 768
    lr: float = 1e-4
    epochs: int = 1000
    train_size: int = 50000
    valid_size: int = 500
    test_size: int = 10000
    log_interval: int = 20  
    early_stop_patience: int = 5  
    grad_clip: float = 1.0  
    
    dataset_type: str = "normal"
    
    top_k: int = 5
    top_p: float = 0.8
    
    # 嵌入连续性损失相关参数
    continuity_weight: float = 0.01  
    continuity_type: str = "l2"  
    normalize_embeddings: bool = False  

    # 熵惩罚相关参数
    entropy_weight: float = 0.5 
    entropy_temperature: float = 0.8
    
    # 设备配置
    device_type: str = "cuda"
    gpu_id: str = "0,1,2"

    def __post_init__(self):
        # 将字符串类型的dataset_type转换为枚举类型
        if isinstance(self.dataset_type, str):
            self.dataset_type = DatasetType(self.dataset_type)
        
        # 设置设备
        if self.device_type == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # 设置GPU ID环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.eos_token, self.sos_token]

    @property
    def vocab(self) -> List[str]:
        return self.special_tokens + list(
            self.digits + self.alphabet
        ) + self.logical_operators + self.operators + self.relation_operators + self.saparator

    @property
    def digit_token_ids(self) -> List[int]:
        """获取数字token的ID列表"""
        tokenizer_vocab = self.vocab
        return [i for i, token in enumerate(tokenizer_vocab) if token in self.digits]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            print(f"配置文件 {yaml_path} 不存在，使用默认配置")
            return cls()
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                
            # 过滤掉yaml中不在Config类定义中的字段
            valid_fields = {f.name for f in fields(cls)}
            filtered_config = {k: v for k, v in yaml_config.items() if k in valid_fields}
                
            return cls(**filtered_config)
        except Exception as e:
            print(f"加载配置文件出错: {e}")
            print("使用默认配置")
            return cls()
    
    def to_yaml(self, yaml_path: str) -> None:
        """将当前配置保存为YAML文件"""
        # 转换为字典，排除不可序列化的属性
        config_dict = {}
        for key, value in asdict(self).items():
            # 跳过device属性，因为torch.device不能直接序列化
            if key == 'device':
                continue
            # 处理枚举类型
            if isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
                
        # 保存到yaml文件
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
    def save_default_yaml(yaml_path: str) -> None:
        """保存默认配置到YAML文件"""
        default_config = Config()
        default_config.to_yaml(yaml_path)
