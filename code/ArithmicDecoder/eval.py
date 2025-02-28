from tqdm.auto import tqdm
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

        results.append({
            "question": q_part,
            "true_answer": true_answer,
            "generated": gen_answer, 
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
    return accuracy, results
