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
    pad_token: str = " "
    eoa_token: str = ">"
    digits: str = "0123456789"
    operators: str = "+*"
    logic_operators: str = "&|!"
    saparator: str = ",;."
    min_digits: int = 2
    max_digits: int = 2  # 扩展位数范围
    max_seq_len: int = 35
    d_model: int = 1024  # 增大模型维度
    nhead: int = 8
    num_layers: int = 8  # 增加层数
    batch_size: int = 256
    lr: float = 1e-4
    epochs: int = 1000
    train_size: int = 500000
    valid_size: int = 5000
    test_size: int = 10000
    log_interval: int = 20  # 日志间隔
    early_stop_patience: int = 5  # 早停耐心值
    grad_clip: float = 1.0  # 梯度裁剪
    device: torch.device = torch.device("cuda")
    gpu_id: str = "0,1,2"
    

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.eoa_token]

    @property
    def vocab(self) -> List[str]:
        return self.special_tokens + list(self.digits + self.operators + "=")
