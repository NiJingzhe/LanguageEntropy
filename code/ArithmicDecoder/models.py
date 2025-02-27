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


# Transformer模型
class EnhancedTransformer(nn.Module):
    def __init__(self, config: Config, tokenizer: EnhancedTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.token_embed = nn.Embedding(tokenizer.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=4 * config.d_model,  # 增加FFN维度
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers
        )
        self.fc = nn.Linear(config.d_model, tokenizer.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in [self.token_embed, self.pos_embed, self.fc]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        seq_len = x.size(1)

        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x_emb = self.token_embed(x) + self.pos_embed(pos)

        mask = self._generate_causal_mask(seq_len, device)
        out = self.decoder(x_emb, x_emb, tgt_mask=mask, memory_mask=mask)
        return self.fc(out)

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)


# 序列生成器
class SequenceGenerator:
    def __init__(
        self, model: EnhancedTransformer, tokenizer: EnhancedTokenizer, config: Config
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()

    def generate(
        self, prompt: str, temperature: float = 0.8, verbose: bool = False
    ) -> str:
        with torch.no_grad():
            # 编码输入序列
            input_seq = self.tokenizer.encode(prompt)
            answer_start_pos = len(input_seq)
            
            if verbose:
                print("\n生成过程:")
                print(f"初始提示: {prompt}")
                print(f"编码后的提示: {input_seq}")
                print(f"答案开始位置: {answer_start_pos}")
            
            # 确保不超过最大长度
            max_length = min(35, self.config.max_seq_len)
            
            # 初始化输入张量，使用pad token填充到最大长度
            padded_input = input_seq.copy()  # 复制输入序列
            current_pos = answer_start_pos
            
            # 生成答案
            while current_pos < max_length - 1:  # 保留一个位置给结束符
                # 准备当前输入
                current_input = padded_input + [self.tokenizer.pad_id] * (max_length - len(padded_input))
                input_tensor = torch.tensor([current_input], dtype=torch.long, device=self.config.device)
                
                if verbose:
                    print("\n" + "-" * 50)
                    print(f"当前位置: {current_pos}")
                    print(f"当前序列: {self.tokenizer.decode(padded_input)}")
                
                # 获取模型输出
                logits = self.model(input_tensor)
                next_token_logits = logits[0, current_pos]  # 使用current_pos而不是-1
                
                # 应用temperature并计算概率
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                
                if verbose:
                    # 获取top-5概率和对应的token
                    top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
                    print("\nTop-5候选词:")
                    for prob, idx in zip(top_probs, top_indices):
                        token = self.tokenizer.itos[idx.item()]
                        print(f"Token: {token:<4} (id: {idx.item():<2}) 概率: {prob.item():.4f}")
                
                # 采样下一个token
                next_token = torch.multinomial(probs, 1).item()
                if verbose:
                    chosen_prob = probs[next_token].item()
                    chosen_token = self.tokenizer.itos[next_token]
                    print(f"\n选择的token: {chosen_token} (id: {next_token}) 概率: {chosen_prob:.4f}")
                
                # 更新序列
                padded_input.append(next_token)
                current_pos += 1
                
                # 如果生成了结束符，停止生成
                if next_token == self.tokenizer.eos_id:
                    if verbose:
                        print("\n检测到结束符，停止生成")
                    break
            
            # 解码最终序列
            final_output = self.tokenizer.decode(padded_input)
            if verbose:
                print("\n" + "=" * 50)
                print(f"最终生成结果: {final_output}")
                print("=" * 50)
            return final_output


