import numpy as np
from scipy.stats import entropy
from tqdm.contrib.concurrent import thread_map
import csv
import os
import argparse


def generate_number(num_digits, rng):
    """生成指定位数的数字，各数位独立采样"""
    digits = rng.integers(0, 10, size=num_digits)
    # 确保首位不是0
    digits[0] = rng.integers(1, 10)
    return sum(d * (10 ** (num_digits - 1 - i)) for i, d in enumerate(digits))


def worker(args):
    """多线程工作单元"""
    chunk_size, seed, num_digits_a, num_digits_b, operation, result_length = args
    rng = np.random.default_rng(seed)
    local_counts = np.zeros((result_length, 10), dtype=np.int64)

    for _ in range(chunk_size):
        a = generate_number(num_digits_a, rng)
        b = generate_number(num_digits_b, rng)

        # 执行指定运算
        if operation == "multiply":
            result = a * b
        elif operation == "add":
            result = a + b
        else:
            raise ValueError("Invalid operation")

        # 格式化结果为固定长度
        result_str = f"{result:0{result_length}d}"[:result_length]

        # 统计各位置数字
        for pos in range(result_length):
            digit = int(result_str[pos]) if pos < len(result_str) else 0
            local_counts[pos][digit] += 1

    return local_counts


def calculate_entropy(config):
    """主计算流程"""
    # 解析配置参数
    num_digits_a = config["num_digits_a"]
    num_digits_b = config["num_digits_b"]
    operation = config["operation"]
    num_samples = config["num_samples"]

    # 确定结果位数
    result_length = {
        "multiply": num_digits_a + num_digits_b,
        "add": max(num_digits_a, num_digits_b) + 1,
    }[operation]

    # 准备并行任务
    chunk_size = 1000  # 每个任务处理量
    num_chunks = num_samples // chunk_size
    remainder = num_samples % chunk_size
    seeds = np.random.SeedSequence().spawn(num_chunks + (1 if remainder else 0))

    # 构建任务列表
    tasks = []
    for i in range(num_chunks):
        tasks.append(
            (
                chunk_size,
                seeds[i].generate_state(1)[0],
                num_digits_a,
                num_digits_b,
                operation,
                result_length,
            )
        )
    if remainder:
        tasks.append(
            (
                remainder,
                seeds[-1].generate_state(1)[0],
                num_digits_a,
                num_digits_b,
                operation,
                result_length,
            )
        )

    # 并行执行任务
    results = thread_map(
        worker, tasks, max_workers=os.cpu_count(), desc="Processing", unit="chunk"
    )

    # 合并结果
    position_counts = np.sum(results, axis=0)

    # 计算熵值
    entropies = []
    for pos in range(result_length):
        probabilities = position_counts[pos] / num_samples
        entropies.append(entropy(probabilities, base=2))

    return entropies, result_length


def save_to_csv(entropies, config, result_length):
    """保存结果到CSV文件"""
    filename = config["output_file"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Position",
                "PlaceValue",
                "Entropy(bits)",
                "Operation",
                "NumDigitsA",
                "NumDigitsB",
            ]
        )
        for pos, e in enumerate(entropies):
            place_value = 10 ** (result_length - 1 - pos)
            writer.writerow(
                [
                    pos,
                    place_value,
                    round(e, 4),
                    config["operation"],
                    config["num_digits_a"],
                    config["num_digits_b"],
                ]
            )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Calculate entropy of digit positions in arithmetic operations."
    )
    parser.add_argument(
        "--num_digits_a",
        type=int,
        default=3,
        help="Number of digits in the first operand",
    )
    parser.add_argument(
        "--num_digits_b",
        type=int,
        default=3,
        help="Number of digits in the second operand",
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=["multiply", "add"],
        default="multiply",
        help="Operation to perform: multiply or add",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="entropy_results.csv",
        help="Output CSV file name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 配置参数
    config = {
        "num_digits_a": args.num_digits_a,
        "num_digits_b": args.num_digits_b,
        "operation": args.operation,
        "num_samples": args.num_samples,
        "output_file": args.output_file,
    }

    # 执行计算
    entropies, result_length = calculate_entropy(config)

    # 保存结果
    save_to_csv(entropies, config, result_length)
    print(f"\nResults saved to {config['output_file']}")

    # 打印摘要
    print("\nEntropy Summary:")
    print(
        f"Operation: {config['operation'].upper()} ({config['num_digits_a']}-digit and {config['num_digits_b']}-digit numbers)"
    )
    print(f"{'Position':<8} {'Place Value':<12} Entropy (bits)")
    for pos, e in enumerate(entropies):
        place_value = 10 ** (result_length - 1 - pos)
        print(f"{pos:<8} {place_value:<12} {e:.4f}")
