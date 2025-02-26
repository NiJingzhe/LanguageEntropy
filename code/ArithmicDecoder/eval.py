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
from .models import EnhancedTransformer, SequenceGenerator
from .config import Config
from .tokenizer import EnhancedTokenizer
from .datasets import EnhancedMathDataset


def evaluate_model(model: EnhancedTransformer, config: Config, num_samples: int = 100, verbose: bool = False):
    """评估模型性能"""
    tokenizer = EnhancedTokenizer(config)
    test_set = EnhancedMathDataset(tokenizer, config, num_samples, seed=44)
    plus_correct = 0
    multiply_correct = 0
    plus_total = 0
    multiply_total = 0
    correct = 0
    total = 0

    generator = SequenceGenerator(model, tokenizer, config)
    results = []

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        question, _ = test_set[idx]
        # 分离问题和答案
        q_part = question.split("=")[0] + "="
        true_answer = question[len(q_part) :].rstrip(">")

        if q_part.find("+") != -1:
            plus_total += 1
        else:
            multiply_total += 1
        
        # 生成答案
        generated = generator.generate(q_part, verbose=verbose)
        gen_answer = generated.rstrip(">")

        # 记录结果
        is_correct = true_answer == gen_answer.split('=')[-1]
        if is_correct:
            if q_part.find('+') != -1:
                plus_correct += 1
            else:
                multiply_correct += 1
                
            correct += 1
        total += 1

        results.append(
            {
                "question": q_part,
                "true_answer": true_answer,
                "generated": gen_answer,
                "correct": is_correct,
            }
        )

        # 打印每10个样本的结果
        if (idx + 1) % 10 == 0 or verbose:
            print(f"\nBatch {(idx + 1) // 10} Results:")
            for r in results[-10:]:
                status = "✓" if r["correct"] else "✗"
                print(
                    f"{status} {r['question']:<10} True: {r['true_answer']:<8} Pred: {r['generated']:<8}"
                )

    accuracy = correct / total
    plus_acc = plus_correct / plus_total
    mult_acc = multiply_correct / multiply_total
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})\nPlus Accuracy: {plus_acc:.2%} ({plus_correct}/{plus_total})\nMultiply Accuracy: {mult_acc:.2%} ({multiply_correct}/{multiply_total})\n")
    return accuracy, results
