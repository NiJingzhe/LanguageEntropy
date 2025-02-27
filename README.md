# LanguageEntropy: 研究语义熵对语言模型表现的影响

这个项目旨在通过数学运算任务研究语言模型中的语义熵及其对模型表现的影响。我们实现了一个具有增强能力的算术解码器（ArithmicDecoder），并设计了特殊的损失函数来操控模型在语义空间中的行为。

## 项目目的

随着语言模型的不断发展，我们对其内部运作机制的理解仍然有限。本项目通过以下方面推进对语言模型的理解：

1. **研究语义熵的影响**：探索预测分布的不确定性（熵）如何影响模型表现，特别是在需要精确性的数学任务中
2. **嵌入空间连续性**：研究嵌入空间中的连续性如何帮助模型学习数字和运算符之间的关系
3. **可视化决策轨迹**：通过跟踪和可视化模型在自回归生成过程中的嵌入轨迹，深入了解模型的决策过程

我们选择数学运算作为研究任务，因为它具有明确定义的正确答案和结构化规则，这使得分析模型的行为变得更加可控。

## 项目结构

```
/home/lildino/Project/PlayGround/LanguageEntropy/
├── code/
│   ├── ArithmicDecoder/
│   │   ├── __init__.py           # 包导出定义
│   │   ├── config.py             # 配置参数定义
│   │   ├── datasets.py           # 数据集实现，包括各类数学运算数据集
│   │   ├── eval.py               # 模型评估脚本
│   │   ├── losses.py             # 损失函数实现，包括熵惩罚和连续性损失
│   │   ├── models.py             # 模型定义（Enhanced Transformer）
│   │   ├── run.py                # 主运行脚本
│   │   ├── tokenizer.py          # 分词器实现
│   │   └── train.py              # 训练循环实现
│   └── utils/                    # 工具函数和辅助脚本
│       └── entropy_calc.py       # 使用数值方法统计算式中结果每一位的熵
```

## 环境要求

```bash
pip install ./requirements.txt
```

## 如何运行ArithmicDecoder

### 1. 训练模型

```bash
# run at project root path
torchrun ./code/ArithmicDecoder/run.py --mode train 

# If you want to do some continue training
torchrun ./code/ArithmicDecoder/run.py --mode train --base your_model_pth_path
```

参数说明:
- `--mode train`: 指定训练模式
- `--base /path/to/model.pth`: 可选，指定基础模型继续训练

### 2. 评估模型

```bash
torchrun ./code/ArithmicDecoder/run.py --mode eval --base best_model.pth --samples 100
```

参数说明:
- `--mode eval`: 指定评估模式
- `--base best_model.pth`: 要评估的模型路径
- `--samples 100`: 评估样本数量
- `--verbose`: 详细输出生成过程, Default True


## 损失函数设计

我们使用了三种损失函数的组合来优化模型：

### 1. 交叉熵损失

标准的序列生成损失，用于最大化目标token的概率:

```python
ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="none")
masked_ce_loss = (ce_loss * masks.view(-1)).sum() / masks.sum()
```

### 2. 连续性损失 (Continuity Loss)

促使模型学习嵌入空间中的连续性，特别是在数字token之间:

```python
# 计算相邻位置嵌入向量之间的距离
distances = torch.sqrt(((current_embeddings - next_embeddings) ** 2).sum(dim=-1) + 1e-8)
# 只在有效位置计算损失
continuity_loss = (distances * valid_pos_mask).sum() / (valid_pos_mask.sum() + 1e-8)
```

这种损失帮助模型构建更有结构的嵌入空间，其中数字的嵌入表示反映了它们之间的数学关系。

### 3. 熵惩罚损失 (Entropy Penalty)

惩罚高不确定性（高熵）的预测，鼓励模型在预测时更加自信:

```python
# 计算每个位置上的预测熵
probs = F.softmax(scaled_logits, dim=-1)
log_probs = F.log_softmax(scaled_logits, dim=-1)
entropy_per_token = -(probs * log_probs).sum(dim=-1)
# 应用掩码并计算平均损失
entropy_loss = (entropy_per_token * masks).sum() / (masks.sum() + 1e-8)
```

总损失函数为三种损失的加权组合:

```python
total_loss = masked_ce_loss + continuity_weight * continuity_loss + entropy_weight * entropy_loss
```

通过调整权重参数可以控制各损失函数的相对重要性。

## 贡献

欢迎贡献代码、提出问题或建议。请通过GitHub Issues提交问题或通过Pull Request贡献代码。
请务必遵循git flow分支模型，所有特性分支采用`feat/`前缀