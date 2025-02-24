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
    min_digits: int = 1
    max_digits: int = 3  # 扩展位数范围
    max_seq_len: int = 35
    d_model: int = 256  # 增大模型维度
    nhead: int = 4
    num_layers: int = 4  # 增加层数
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 100
    train_size: int = 50000
    valid_size: int = 5000
    test_size: int = 1000
    log_interval: int = 30  # 日志间隔
    early_stop_patience: int = 5  # 早停耐心值
    grad_clip: float = 1.0  # 梯度裁剪
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.eoa_token]

    @property
    def vocab(self) -> List[str]:
        return self.special_tokens + list(self.digits + self.operators + "=")


# 分词器
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


# 数据集
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
        full_seq = f"{question}{answer}{self.config.eoa_token}"
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


# 数据整理函数
def enhanced_collate_fn(
    batch: List[Tuple[str, int]], tokenizer: EnhancedTokenizer, config: Config
):
    sequences, answer_starts = zip(*batch)

    encoded = [tokenizer.encode(seq) for seq in sequences]
    max_len = config.max_seq_len

    all_inputs, all_targets, all_masks = [], [], []

    for seq, start_idx in zip(encoded, answer_starts):
        for pos in range(start_idx, len(seq)):
            # 生成每个时间步的输入/目标
            inputs = seq[:pos] + [tokenizer.pad_id] * (max_len - pos)
            targets = seq[: pos + 1] + [tokenizer.pad_id] * (max_len - (pos + 1))
            mask = (
                [0.0] * (start_idx)
                + [1.0] * (pos - start_idx + 1)
                + [0.0] * (max_len - (pos + 1))
            )

            all_inputs.append(inputs)
            all_targets.append(targets)
            all_masks.append(mask)

    return (
        torch.tensor(all_inputs, dtype=torch.long),
        torch.tensor(all_targets, dtype=torch.long),
        torch.tensor(all_masks, dtype=torch.float),
    )


# 训练函数
def train_enhanced_model(
    config: Config, base_model_path: Optional[str] = None
) -> EnhancedTransformer:
    tokenizer = EnhancedTokenizer(config)
    writer = SummaryWriter()

    # 数据集
    train_set = EnhancedMathDataset(tokenizer, config, config.train_size, seed=42)
    valid_set = EnhancedMathDataset(tokenizer, config, config.valid_size, seed=43)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        collate_fn=lambda b: enhanced_collate_fn(b, tokenizer, config),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        collate_fn=lambda b: enhanced_collate_fn(b, tokenizer, config),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 初始化模型并移动到指定设备
    model = EnhancedTransformer(config, tokenizer).to(config.device)

    # 如果指定了基础模型，加载它
    if base_model_path and os.path.exists(base_model_path):
        print(f"Loading base model from {base_model_path}")
        model.load_state_dict(torch.load(base_model_path, map_location=config.device))

    print(f"Training on device: {config.device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=2
    )
    best_val_loss = float("inf")
    patience_counter = 0

    # 训练循环
    global_step = 0  # 添加全局步数计数器
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = []
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]", leave=False
        )

        for batch_idx, (inputs, targets, masks) in enumerate(progress_bar):
            # 将数据移动到指定设备
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            masks = masks.to(config.device)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
            )
            masked_loss = (loss * masks.view(-1)).sum() / masks.sum()

            masked_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # 记录每个step的loss
            writer.add_scalar("Train/step_loss", masked_loss.item(), global_step)
            writer.add_scalar(
                "Train/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )

            train_loss.append(masked_loss.item())
            if batch_idx % config.log_interval == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{np.mean(train_loss[-config.log_interval:]):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

            global_step += 1

        # 验证阶段
        model.eval()
        val_loss = []
        display_samples = 5  # 展示的样本数量
        all_samples = []  # 存储所有样本

        with torch.no_grad():
            val_progress = tqdm(
                valid_loader,
                desc=f"Epoch {epoch+1}/{config.epochs} [Valid]",
                leave=False,
            )
            for inputs, targets, masks in val_progress:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                masks = masks.to(config.device)

                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
                )
                masked_loss = (loss * masks.view(-1)).sum() / masks.sum()
                val_loss.append(masked_loss.item())
                val_progress.set_postfix({"val_loss": f"{np.mean(val_loss):.4f}"})

                # 收集所有样本
                predictions = torch.argmax(logits, dim=-1)
                for i in range(inputs.size(0)):
                    input_seq = tokenizer.decode(
                        [t.item() for t in inputs[i] if t.item() != tokenizer.pad_id]
                    )
                    target_seq = tokenizer.decode(
                        [t.item() for t in targets[i] if t.item() != tokenizer.pad_id]
                    )
                    pred_seq = tokenizer.decode(
                        [
                            t.item()
                            for t in predictions[i]
                            if t.item() != tokenizer.pad_id
                        ]
                    )
                    all_samples.append((input_seq, target_seq, pred_seq))

            # 随机选择样本进行展示
            if all_samples:
                print("\nRandom Validation Samples:")
                selected_samples = random.sample(
                    all_samples, min(display_samples, len(all_samples))
                )
                for idx, (input_seq, target_seq, pred_seq) in enumerate(
                    selected_samples, 1
                ):
                    print(f"\nSample {idx}:")
                    print(f"Input:      {input_seq}")
                    print(f"Target:     {target_seq}")
                    print(f"Prediction: {pred_seq}")
                    print("-" * 50)

        # 统计指标
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        writer.add_scalars(
            "Loss", {"train": avg_train_loss, "valid": avg_val_loss}, epoch
        )

        # 学习率调度和早停
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        # 打印日志
        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Valid Loss: {avg_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Patience: {patience_counter}/{config.early_stop_patience}"
        )

        if patience_counter >= config.early_stop_patience:
            print("Early stopping triggered!")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pth"))
    writer.close()
    return model


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
        self, prompt: str, max_length: int = 20, temperature: float = 0.8
    ) -> str:
        with torch.no_grad():
            input_seq = self.tokenizer.encode(prompt)
            prompt_length = len(input_seq)  # 记录问题的长度
            input_tensor = torch.tensor(
                [input_seq], dtype=torch.long, device=self.config.device
            )

            # 开始从提示的末尾生成答案
            for _ in range(max_length):
                logits = self.model(input_tensor)

                # 获取 index 为 prompt length位置的logits
                next_position_logits = logits[0, prompt_length]

                # 使用temperature进行概率调整
                probs = F.softmax(next_position_logits / temperature, dim=-1)

                # 采样下一个token
                next_token = torch.multinomial(probs, 1)

                # 将新token添加到序列中
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

                # 如果生成了结束符，停止生成
                if next_token.item() == self.tokenizer.eoa_id:
                    break

            # 解码完整序列
            generated_sequence = self.tokenizer.decode(input_tensor[0].tolist())
            return generated_sequence


def evaluate_model(model: EnhancedTransformer, config: Config, num_samples: int = 100):
    """评估模型性能"""
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
        true_answer = question[len(q_part) :].rstrip(">")

        # 生成答案
        generated = generator.generate(q_part)
        gen_answer = generated[len(q_part) :].rstrip(">")

        # 记录结果
        is_correct = true_answer == gen_answer
        if is_correct:
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
        if (idx + 1) % 10 == 0:
            print(f"\nBatch {(idx + 1) // 10} Results:")
            for r in results[-10:]:
                status = "✓" if r["correct"] else "✗"
                print(
                    f"{status} {r['question']:<10} True: {r['true_answer']:<8} Pred: {r['generated']:<8}"
                )

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Arithmetic Expression Generator")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        required=True,
        help="Operation mode: train or eval",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Path to base model checkpoint (optional)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples for evaluation (default: 100)",
    )

    args = parser.parse_args()
    config = Config()
    print(f"Using device: {config.device}")

    if args.mode == "train":
        model = train_enhanced_model(config, args.base)
        print("Training completed. Model saved as 'best_model.pth'")

    elif args.mode == "eval":
        # 加载模型
        model_path = args.base if args.base else "best_model.pth"
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            return

        model = EnhancedTransformer(config, EnhancedTokenizer(config)).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"Loaded model from {model_path}")

        # 评估模型
        accuracy, _ = evaluate_model(model, config, args.samples)


if __name__ == "__main__":
    main()
