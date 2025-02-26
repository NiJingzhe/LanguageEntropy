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
from models import EnhancedTransformer
from datasets import EnhancedMathDataset, enhanced_collate_fn
from losses import CombinedLoss

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
    model = EnhancedTransformer(config, tokenizer)
    model = nn.DataParallel(model)
    model = model.to(config.device)
    # 如果指定了基础模型，加载它
    if base_model_path and os.path.exists(base_model_path):
        print(f"Loading base model from {base_model_path}")
        model.load_state_dict(torch.load(base_model_path, map_location=config.device))

    print(f"Training on device: {config.device}")

    # 初始化组合损失函数
    combined_loss = CombinedLoss(
        continuity_weight=config.continuity_weight,
        continuity_type=config.continuity_type,
        normalize_embeddings=config.normalize_embeddings,
        apply_to_digits_only=config.apply_to_digits_only,
        digit_tokens=config.digit_token_ids,
        entropy_weight=config.entropy_weight,
        entropy_temperature=config.entropy_temperature,
        apply_entropy_to_digits_only=config.apply_entropy_to_digits_only
    )
    
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
        train_ce_loss = []
        train_cont_loss = []
        train_entropy_loss = []  # 新增熵损失记录
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
            
            # 使用更新后的组合损失函数，现在返回4个值
            total_loss, ce_loss, cont_loss, entropy_loss = combined_loss.compute_loss(
                logits, targets, masks, model
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # 记录每个step的loss，增加熵损失的记录
            writer.add_scalar("Train/total_loss", total_loss.item(), global_step)
            writer.add_scalar("Train/ce_loss", ce_loss.item(), global_step)
            writer.add_scalar("Train/continuity_loss", cont_loss.item(), global_step)
            writer.add_scalar("Train/entropy_loss", entropy_loss.item(), global_step)
            writer.add_scalar(
                "Train/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )

            train_loss.append(total_loss.item())
            train_ce_loss.append(ce_loss.item())
            train_cont_loss.append(cont_loss.item())
            train_entropy_loss.append(entropy_loss.item())
            
            if batch_idx % config.log_interval == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{np.mean(train_loss[-config.log_interval:]):.4f}",
                        "ce": f"{np.mean(train_ce_loss[-config.log_interval:]):.4f}",
                        "cont": f"{np.mean(train_cont_loss[-config.log_interval:]):.4f}",
                        "ent": f"{np.mean(train_entropy_loss[-config.log_interval:]):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

            global_step += 1

        # 验证阶段
        model.eval()
        val_loss = []
        val_ce_loss = []
        val_cont_loss = []
        val_entropy_loss = []  # 新增熵损失记录
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
                
                # 使用更新后的组合损失函数
                total_loss, ce_loss, cont_loss, entropy_loss = combined_loss.compute_loss(
                    logits, targets, masks, model
                )
                
                val_loss.append(total_loss.item())
                val_ce_loss.append(ce_loss.item())
                val_cont_loss.append(cont_loss.item())
                val_entropy_loss.append(entropy_loss.item())
                
                val_progress.set_postfix({
                    "val_loss": f"{np.mean(val_loss):.4f}",  # 将 '::' 修改为 ':'
                    "val_ce": f"{np.mean(val_ce_loss):.4f}",
                    "val_cont": f"{np.mean(val_cont_loss)::.4f}",
                    "val_ent": f"{np.mean(val_entropy_loss)::.4f}"
                })

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
        avg_train_ce = np.mean(train_ce_loss)
        avg_val_ce = np.mean(val_ce_loss)
        avg_train_cont = np.mean(train_cont_loss) 
        avg_val_cont = np.mean(val_cont_loss)
        avg_train_ent = np.mean(train_entropy_loss)
        avg_val_ent = np.mean(val_entropy_loss)
        
        writer.add_scalars(
            "Loss/Total", {"train": avg_train_loss, "valid": avg_val_loss}, epoch
        )
        writer.add_scalars(
            "Loss/CrossEntropy", {"train": avg_train_ce, "valid": avg_val_ce}, epoch
        )
        writer.add_scalars(
            "Loss/Continuity", {"train": avg_train_cont, "valid": avg_val_cont}, epoch
        )
        writer.add_scalars(
            "Loss/Entropy", {"train": avg_train_ent, "valid": avg_val_ent}, epoch
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
            f"Train Total: {avg_train_loss:.4f} | "
            f"Valid Total: {avg_val_loss:.4f} | "
            f"Train CE: {avg_train_ce:.4f} | "
            f"Valid CE: {avg_val_ce:.4f} | "
            f"Train Cont: {avg_train_cont:.4f} | "
            f"Valid Cont: {avg_val_cont:.4f} | "
            f"Train Ent: {avg_train_ent:.4f} | "
            f"Valid Ent: {avg_val_ent:.4f} | "
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
