import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List, Dict, Optional
import random
import numpy as np
from tqdm.auto import tqdm
import argparse
import os

torch.set_printoptions(threshold=np.inf)


# 配置类
class Config:
    
    # 词表设置
    pad_token: str = "@"
    eos_token: str = "#"
    sos_token: str = "$"
    digits: str = "0123456789"
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    operators: list[str] = ["+", "*"]
    relation_operators: list[str] = ["=", "!=", ">", "<", ">=", "<="]
    logical_operators: list[str] = ["&", "|", "~"]
    saparator: list[str] = ["(", ")", "[", "]", "{", "}", ",", ":", ".", ";", " "]
    
    # 两个操作数的位数范围
    min_digits: int = 1
    max_digits: int = 2  # 扩展位数范围
    
    # 模型规模设置
    max_seq_len: int = 35
    d_model: int = 1024  # 增大模型维度
    nhead: int = 8
    num_layers: int = 8  # 增加层数
    
    # 训练超参
    batch_size: int = 256
    lr: float = 1e-4
    epochs: int = 1000
    train_size: int = 50000
    valid_size: int = 500
    test_size: int = 10000
    log_interval: int = 20  # 日志间隔
    early_stop_patience: int = 5  # 早停耐心值
    grad_clip: float = 1.0  # 梯度裁剪
    
    top_k: int = 5
    top_p: float = 0.8
    
    # device
    device: torch.device = torch.device("cuda")
    # 请记得更改id，如果不一致的话
    gpu_id: str = "0,1,2"

    # 嵌入连续性损失相关参数
    continuity_weight: float = 0.01  # 连续性损失的权重 0.01 ~ 0.03 are recommended
    continuity_type: str = "l2"  # 距离类型：'l1', 'l2', 或 'cosine'
    normalize_embeddings: bool = False  # 是否在计算连续性前归一化嵌入

    # 熵惩罚相关参数
    entropy_weight: float = 0.5  # 熵惩罚的权重系数
    entropy_temperature: float = 0.8  # 熵计算的温度系数

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.eos_token, self.sos_token]

    @property
    def vocab(self) -> List[str]:
        return self.special_tokens + list(
            self.digits
            + self.alphabet
        ) + \
        self.logical_operators + self.operators + self.relation_operators + self.saparator

    @property
    def digit_token_ids(self) -> List[int]:
        """获取数字token的ID列表"""
        tokenizer_vocab = self.vocab
        return [i for i, token in enumerate(tokenizer_vocab) if token in self.digits]
