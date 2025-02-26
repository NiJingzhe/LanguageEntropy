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
from models import EnhancedTransformer, SequenceGenerator
from config import Config
from tokenizer import EnhancedTokenizer
from datasets import EnhancedMathDataset

def evaluate_model(model: EnhancedTransformer, config: Config, num_samples: int = 100, verbose: bool = False):
    """评估模型性能，并按操作类型和答案长度统计逐位正确率"""
    tokenizer = EnhancedTokenizer(config)
    test_set = EnhancedMathDataset(tokenizer, config, num_samples, seed=44)
    correct = 0
    total = 0

    generator = SequenceGenerator(model, tokenizer, config)
    results = []
    
    # 为加法和乘法创建单独的结果字典
    addition_results = {"3": [], "4": []}
    multiplication_results = {"3": [], "4": []}

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        question, _ = test_set[idx]
        # 分离问题和答案
        q_part = question.split("=")[0] + "="
        true_answer = question[len(q_part):].rstrip(">")

        # 确定运算类型（加法或乘法）
        operation = "+" if "+" in q_part else "*"
        
        # 生成答案
        generated = generator.generate(q_part, verbose=verbose)
        gen_answer = generated[len(q_part):].split(">")[0]  # 去掉可能的结束符

        # 记录结果
        is_correct = true_answer == gen_answer
        if is_correct:
            correct += 1
        total += 1

        # 收集逐位比较的数据
        true_len = len(true_answer)
        
        # 只考虑3位和4位的答案
        if str(true_len) in ["3", "4"]:
            result_dict = addition_results if operation == "+" else multiplication_results
            result_entry = {
                "question": q_part,
                "true_answer": true_answer,
                "gen_answer": gen_answer,
                "correct": is_correct
            }
            result_dict[str(true_len)].append(result_entry)

        results.append({
            "question": q_part,
            "true_answer": true_answer,
            "generated": generated,
            "correct": is_correct,
        })
        
        # 打印每10个样本的结果
        if (idx + 1) % 10 == 0 or verbose:
            print(f"\nBatch {(idx + 1) // 10} Results:")
            for r in results[-10:]:
                status = "✓" if r["correct"] else "✗"
                print(f"{status} {r['question']:<10} True: {r['true_answer']:<8} Pred: {r['generated']:<8}")

    # 计算总体准确率
    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # 计算和打印逐位正确率
    print("\n===== 逐位正确率统计 =====")
    
    # 处理加法结果
    print("\n--- 加法结果 ---")
    for length in ["3", "4"]:
        samples = addition_results[length]
        if not samples:
            print(f"长度为 {length} 的加法样本数量为0，无法计算正确率")
            continue
        
        digit_correct = [0] * int(length)
        total_samples = len(samples)
        
        for sample in samples:
            true_ans = sample["true_answer"]
            gen_ans = sample["gen_answer"]
            
            # 确保生成答案长度足够，如果不够用"_"填充
            gen_ans = gen_ans.ljust(len(true_ans), "_")
            
            for i in range(len(true_ans)):
                if i < len(gen_ans) and true_ans[i] == gen_ans[i]:
                    digit_correct[i] += 1
        
        print(f"\n长度为 {length} 的加法结果 (样本数: {total_samples}):")
        for i in range(len(digit_correct)):
            pos_accuracy = digit_correct[i] / total_samples if total_samples > 0 else 0
            print(f"  位置 {i+1}: {pos_accuracy:.2%}")
            
    # 处理乘法结果
    print("\n--- 乘法结果 ---")
    for length in ["3", "4"]:
        samples = multiplication_results[length]
        if not samples:
            print(f"长度为 {length} 的乘法样本数量为0，无法计算正确率")
            continue
        
        digit_correct = [0] * int(length)
        total_samples = len(samples)
        
        for sample in samples:
            true_ans = sample["true_answer"]
            gen_ans = sample["gen_answer"]
            
            # 确保生成答案长度足够，如果不够用"_"填充
            gen_ans = gen_ans.ljust(len(true_ans), "_")
            
            for i in range(len(true_ans)):
                if i < len(gen_ans) and true_ans[i] == gen_ans[i]:
                    digit_correct[i] += 1
        
        print(f"\n长度为 {length} 的乘法结果 (样本数: {total_samples}):")
        for i in range(len(digit_correct)):
            pos_accuracy = digit_correct[i] / total_samples if total_samples > 0 else 0
            print(f"  位置 {i+1}: {pos_accuracy:.2%}")
    
    return accuracy, results
