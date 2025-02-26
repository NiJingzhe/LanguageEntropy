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
from config import Config


class EnhancedTokenizer:
    def __init__(self, config: Config):
        self.config = config
        self.stoi: Dict[str, int] = {token: i for i, token in enumerate(config.vocab)}
        self.itos: Dict[int, str] = {i: token for i, token in enumerate(config.vocab)}
        self.pad_id = self.stoi[config.pad_token]
        self.eoa_id = self.stoi[config.eoa_token]
        self.vocab_size = len(config.vocab)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.itos[i] for i in ids])
