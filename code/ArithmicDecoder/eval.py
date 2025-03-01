import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from models import EnhancedTransformer, SequenceGenerator
from config import Config, DatasetType
from tokenizer import EnhancedTokenizer
from datasets import EnhancedMathDataset, AnswerWithPrefixSpaceMathDataset, ReverseAnswerWithSpacePadMathDataset

def evaluate_model(model: EnhancedTransformer, config: Config, num_samples: int = 100, verbose: bool = False):
    """评估模型性能，并按操作类型和答案长度统计逐位正确率、熵值和能量"""
    tokenizer = EnhancedTokenizer(config)
    
    if config.dataset_type == DatasetType.NORMAL:
        test_set = EnhancedMathDataset(tokenizer, config, num_samples, seed=44)
    elif config.dataset_type == DatasetType.SPACE_PREFIX:
        test_set = AnswerWithPrefixSpaceMathDataset(tokenizer, config, num_samples, seed=44)
    elif config.dataset_type == DatasetType.REVERSE:
        test_set = ReverseAnswerWithSpacePadMathDataset(tokenizer, config, num_samples, seed=44)
    
    correct = 0
    total = 0

    generator = SequenceGenerator(model, tokenizer, config)
    results = []
    
    # 用于收集所有样本的熵和能量
    all_entropy_values = []
    all_energy_values = []
    position_entropies = {}  # 按位置索引存储熵值
    position_energies = {}   # 按位置索引存储能量值
    
    for idx in tqdm(range(num_samples), desc="Evaluating"):
        question, _ = test_set[idx]
        # 分离问题和答案
        q_part = question.split("=")[0] + "="
        true_answer = question[len(q_part):].rstrip(">")

        # 生成答案
        generated, stats = generator.generate(q_part, verbose=verbose)
        gen_answer = generated[len(q_part):].split(">")[0]  # 去掉可能的结束符

        # 收集统计数据
        entropy_values = stats["entropy_values"]
        energy_values = stats["energy_values"]
        all_entropy_values.extend(entropy_values)
        all_energy_values.extend(energy_values)
        
        # 按位置收集统计数据
        for pos, (entropy, energy) in enumerate(zip(entropy_values, energy_values)):
            if pos not in position_entropies:
                position_entropies[pos] = []
                position_energies[pos] = []
            position_entropies[pos].append(entropy)
            position_energies[pos].append(energy)

        # 记录结果
        is_correct = true_answer == gen_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": q_part,
            "true_answer": true_answer,
            "generated": gen_answer, 
            "correct": is_correct,
            "entropy_values": entropy_values,
            "energy_values": energy_values,
            "avg_entropy": sum(entropy_values) / len(entropy_values) if entropy_values else 0,
            "avg_energy": sum(energy_values) / len(energy_values) if energy_values else 0
        })
        
        # 打印每10个样本的结果
        if (idx + 1) % 10 == 0 or verbose:
            print(f"\nBatch {(idx + 1) // 10} Results:")
            for r in results[-10:]:
                status = "✓" if r["correct"] else "✗"
                print(f"{status} {r['question']:<10} True: {r['true_answer']:<8} Pred: {r['generated']:<8}")
                print(f"  Avg Entropy: {r['avg_entropy']:.4f}, Avg Energy: {r['avg_energy']:.4f}")

    # 计算总体准确率
    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # 计算所有token的平均熵值和能量
    avg_entropy = sum(all_entropy_values) / len(all_entropy_values) if all_entropy_values else 0
    avg_energy = sum(all_energy_values) / len(all_energy_values) if all_energy_values else 0
    print(f"\n总体统计:")
    print(f"平均熵值: {avg_entropy:.4f}")
    print(f"平均能量: {avg_energy:.4f}")
    
    # 计算并打印每个位置的平均熵值和能量
    print("\n按位置统计:")
    position_stats = []
    for pos in sorted(position_entropies.keys()):
        pos_entropy = sum(position_entropies[pos]) / len(position_entropies[pos])
        pos_energy = sum(position_energies[pos]) / len(position_energies[pos])
        position_stats.append((pos, pos_entropy, pos_energy))
        print(f"位置 {pos}: 平均熵值 = {pos_entropy:.4f}, 平均能量 = {pos_energy:.4f}")
    
    # 可视化位置统计
    try:
        _plot_position_stats(position_stats)
    except Exception as e:
        print(f"绘图失败: {e}")
    
    # 计算和打印逐位正确率
    return accuracy, results

def _plot_position_stats(position_stats):
    """绘制位置统计图表"""
    positions, entropies, energies = zip(*position_stats)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制熵值曲线
    plt.subplot(1, 2, 1)
    plt.plot(positions, entropies, 'b-o', label='熵值')
    plt.title('每个位置的平均熵值')
    plt.xlabel('位置索引')
    plt.ylabel('熵值')
    plt.grid(True)
    
    # 绘制能量曲线
    plt.subplot(1, 2, 2)
    plt.plot(positions, energies, 'r-o', label='能量')
    plt.title('每个位置的平均能量')
    plt.xlabel('位置索引')
    plt.ylabel('能量')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('position_stats.png')
    print("统计图表已保存为 'position_stats.png'")
