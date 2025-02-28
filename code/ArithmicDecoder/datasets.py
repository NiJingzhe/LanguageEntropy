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
from tokenizer import EnhancedTokenizer
from config import Config


class EnhancedMathDataset(Dataset):
    def __init__(
        self,
        tokenizer: EnhancedTokenizer,
        config: Config,
        dataset_size: int,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.size = dataset_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        operator = self.rng.choice(list(self.config.operators))
        a = self._generate_number()
        b = self._generate_number()
        question = f"{a}{operator}{b}="
        answer = str(self._calculate(a, b, operator))
        full_seq = f"{self.config.sos_token}{question}{answer}{self.config.eos_token}"
        answer_start = len(question)
        return full_seq, answer_start

    def _generate_number(self) -> int:
        digits = self.rng.randint(self.config.min_digits, self.config.max_digits)
        return self.rng.randint(10 ** (digits - 1), 10**digits - 1)

    def _calculate(self, a: int, b: int, op: str) -> int:
        if op == "+":
            return a + b
        if op == "*":
            return a * b
        raise ValueError(f"Unsupported operator: {op}")


class AnswerWithPrefixPadMathDataset(Dataset):

    def __init__(
        self,
        tokenizer: EnhancedTokenizer,
        config: Config,
        dataset_size: int,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.size = dataset_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[str, int]:

        operator = self.rng.choice(list(self.config.operators))
        a = self._generate_number()
        b = self._generate_number()
        a_str = str(a)
        b_str = str(b)
        answer_len = 0

        if operator == "+":
            answer_len = max(len(a_str), len(b_str)) + 1

        if operator == "*":
            answer_len = len(a_str) + len(b_str)

        question = f"{a}{operator}{b}="
        answer = str(self._calculate(a, b, operator))
        full_seq = (
            f"{self.config.sos_token}{question}"
            + f"{self.config.pad_token}" * (answer_len - len(answer))
            + f"{self.config.eos_token}"
        )
        answer_start = len(question)
        return full_seq, answer_start
    def _generate_number(self) -> int:
        digits = self.rng.randint(self.config.min_digits, self.config.max_digits)
        return self.rng.randint(10 ** (digits - 1), 10**digits - 1)

    def _calculate(self, a: int, b: int, op: str) -> int:
        if op == "+":
            return a + b
        if op == "*":
            return a * b
        raise ValueError(f"Unsupported operator: {op}")



class ReverseAnswerMathDataset(Dataset):
    """
    将Answer倒过来写的数据集
    """
    def __init__(
        self,
        tokenizer: EnhancedTokenizer,
        config: Config,
        dataset_size: int,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.size = dataset_size
        self.rng = random.Random(seed)


    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[str, int]:

        operator = self.rng.choice(list(self.config.operators))
        a = self._generate_number()
        b = self._generate_number()
        question = f"{a}{operator}{b}="
        answer = str(self._calculate(a, b, operator))
        answer = answer[::-1]
        full_seq = f"{self.config.sos_token}{question}{answer}{self.config.eos_token}"
        answer_start = len(question)
        return full_seq, answer_start

    def _calculate(self, a: int, b: int, op: str) -> int:
        if op == "+":
            return a + b
        if op == "*":
            return a * b
        raise ValueError(f"Unsupported operator: {op}")

    def _generate_number(self) -> int:
        digits = self.rng.randint(self.config.min_digits, self.config.max_digits)
        return self.rng.randint(10 ** (digits - 1), 10**digits - 1)


# def enhanced_collate_fn(
#     batch: List[Tuple[str, int]], tokenizer: EnhancedTokenizer, config: Config
# ):
#     sequences, answer_starts = zip(*batch)

#     encoded = [tokenizer.encode(seq) for seq in sequences]
#     max_len = config.max_seq_len

#     all_inputs, all_targets, all_masks = [], [], []

#     for seq, start_idx in zip(encoded, answer_starts):
#         for pos in range(start_idx, len(seq)):
#             # 生成每个时间步的输入/目标
#             inputs = seq[:pos] + [tokenizer.pad_id] * (max_len - pos)
#             targets = seq[: pos + 1] + [tokenizer.pad_id] * (max_len - (pos + 1))
#             # 修改mask，只关注答案部分的loss
#             mask = (
#                 [0.0] * start_idx
#                 + [1.0] * (pos - start_idx + 1)
#                 + [0.0] * (max_len - (pos + 1))
#             )

#             all_inputs.append(inputs)
#             all_targets.append(targets)
#             all_masks.append(mask)

#     return (
#         torch.tensor(all_inputs, dtype=torch.long),
#         torch.tensor(all_targets, dtype=torch.long),
#         torch.tensor(all_masks, dtype=torch.float),
#     )


def improved_collate_fn(batch: List[Tuple[str, int]], tokenizer: EnhancedTokenizer, config: Config):
    """
    优化版collate函数，每个序列只创建一个样本，避免数据爆炸
    
    参数:
        batch: 包含(序列, 答案起始位置)元组的列表
        tokenizer: 分词器实例
        config: 配置实例
        
    返回:
        inputs: 形状为[batch_size, max_seq_len]的编码输入
        targets: 形状为[batch_size, max_seq_len]的目标输出（右移一位）
        masks: 形状为[batch_size, max_seq_len]的损失掩码
    """
    sequences, answer_starts = zip(*batch)
    encoded = [tokenizer.encode(seq) for seq in sequences]
    max_len = config.max_seq_len
    
    # 对序列进行padding
    padded_seqs = [seq + [tokenizer.pad_id] * (max_len - len(seq)) for seq in encoded]
    
    # 创建输入和目标（右移一位）
    inputs = [seq[:-1] + [tokenizer.pad_id] for seq in padded_seqs]
    targets = [seq[1:] + [tokenizer.pad_id] for seq in padded_seqs]
    
    # 创建损失mask：只考虑答案部分和有效序列
    masks = []
    for seq, start_idx in zip(padded_seqs, answer_starts):
        # 计算序列有效长度
        valid_len = len([t for t in seq if t != tokenizer.pad_id])
        # 对问题部分为0，答案部分为1，padding部分为0
        # 由于right shift，mask也要相应调整
        mask = [0.0] * start_idx + [1.0] * (valid_len - start_idx) + [0.0] * (max_len - valid_len)
        # 由于right shift，mask也要前移一位
        mask = mask[1:] + [0.0]
        masks.append(mask)
    
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(masks, dtype=torch.float),
    )
