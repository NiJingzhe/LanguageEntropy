import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss:
    """
    组合损失函数，包含交叉熵损失、嵌入连续性损失和预测熵惩罚
    """

    def __init__(
        self,
        continuity_weight=0.1,
        continuity_type="l2",
        normalize_embeddings=True,
        digit_tokens=None,
        entropy_weight=0.1,
        entropy_temperature=1.0,
    ):
        """
        初始化组合损失函数

        参数:
        - continuity_weight: 连续性损失的权重系数
        - continuity_type: 距离计算类型，'l1', 'l2', 或 'cosine'
        - normalize_embeddings: 是否在计算连续性时归一化嵌入向量
        - apply_to_digits_only: 是否仅对数字token应用连续性损失
        - digit_tokens: 数字token的ID列表
        - entropy_weight: 熵惩罚的权重系数
        - entropy_temperature: 熵计算的温度系数
        - apply_entropy_to_digits_only: 是否只对数字token应用熵惩罚
        """
        self.continuity_weight = continuity_weight
        self.continuity_type = continuity_type
        self.normalize_embeddings = normalize_embeddings
        self.digit_tokens = set(digit_tokens) if digit_tokens else None

        # 熵惩罚相关参数
        self.entropy_weight = entropy_weight
        self.entropy_temperature = entropy_temperature

    def compute_loss(self, logits, targets, masks, model, answer_masks=None):
        """
        计算组合损失

        参数:
        - logits: 模型输出的logits，形状为[batch_size, seq_len, vocab_size]
        - targets: 目标序列，形状为[batch_size, seq_len]
        - masks: 用于masking的张量，形状为[batch_size, seq_len]
        - model: 模型对象，用于获取token embeddings
        - answer_masks: 标识答案部分的mask，形状为[batch_size, seq_len]，
                       如果为None，则使用masks作为答案mask

        返回:
        - 总损失、交叉熵损失、连续性损失和熵损失
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        )
        masked_ce_loss = (ce_loss * masks.view(-1)).sum() / masks.sum()

        # 初始化连续性损失和熵损失为0
        continuity_loss = torch.tensor(0.0, device=logits.device)
        entropy_loss = torch.tensor(0.0, device=logits.device)

        # 计算连续性损失（如果需要）
        if self.continuity_weight > 0:
            # 使用answer_masks如果提供，否则使用masks
            answer_masks = answer_masks if answer_masks is not None else masks

            # 获取模型的token嵌入层
            token_embedding = (
                model.module.token_embed
                if hasattr(model, "module")
                else model.token_embed
            )

            # 获取目标序列的token的嵌入
            embeddings = token_embedding(targets)  # [batch_size, seq_len, d_model]

            # 计算连续性损失
            continuity_loss = self._compute_continuity_loss(embeddings, answer_masks)

        # 计算熵损失（如果需要）
        if self.entropy_weight > 0:
            entropy_loss = self._compute_entropy_loss(logits, targets, masks)

        # 组合损失
        total_loss = (
            masked_ce_loss
            + self.continuity_weight * continuity_loss
            + self.entropy_weight * entropy_loss
        )

        return total_loss, masked_ce_loss, continuity_loss, entropy_loss

    def _compute_entropy_loss(self, logits, targets, masks):
        """
        计算预测分布的熵，用于惩罚不确定性高的预测

        参数:
        - logits: 模型输出的logits，形状为[batch_size, seq_len, vocab_size]
        - targets: 目标序列，形状为[batch_size, seq_len]
        - masks: 用于masking的张量，形状为[batch_size, seq_len]

        返回:
        - 熵损失
        """
        batch_size, seq_len, vocab_size = logits.size()

        effective_mask = masks

        # 如果没有有效位置，返回0损失
        if effective_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # 应用温度系数
        scaled_logits = logits / self.entropy_temperature

        # 计算每个位置上预测分布的熵
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)  # [batch_size, seq_len]

        # 应用mask并计算平均熵损失
        masked_entropy = entropy_per_token * effective_mask
        entropy_loss = masked_entropy.sum() / (effective_mask.sum() + 1e-8)

        return entropy_loss

    def _compute_continuity_loss(self, embeddings, masks):
        """计算嵌入连续性损失"""
        # 准备用于计算损失的mask
        # 我们只对相邻位置计算连续性损失，最后一个位置没有下一个位置

        # 一个直观的说明:
        # mask = [0, 0, 0, 1, 1, 1, 0, 0, 0]
        # 得到的valid_pos_mask = [0,0,0,1,1,1,0,0] * [0,0,1,1,1,0,0,0] = [0,0,0,1,1,0,0,0]
        # 这样一来，valid pos mask就和需要求解的“间隔”对应了
        valid_pos_mask = masks[:, :-1] * masks[:, 1:]

        # 如果没有有效位置，返回0损失
        if valid_pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # 获取当前位置和下一位置的嵌入
        current_embeddings = embeddings[:, :-1]  # [batch_size, seq_len-1, d_model]
        next_embeddings = embeddings[:, 1:]  # [batch_size, seq_len-1, d_model]

        # 可选的嵌入归一化
        if self.normalize_embeddings:
            current_embeddings = F.normalize(current_embeddings, p=2, dim=-1)
            next_embeddings = F.normalize(next_embeddings, p=2, dim=-1)

        # 计算距离
        if self.continuity_type == "l1":
            distances = torch.abs(current_embeddings - next_embeddings).sum(dim=-1)
        elif self.continuity_type == "l2":
            distances = torch.sqrt(
                ((current_embeddings - next_embeddings) ** 2).sum(dim=-1) + 1e-8
            )
        elif self.continuity_type == "cosine":
            # 余弦相似度转换为距离：1 - cosine_similarity
            cos_sim = (current_embeddings * next_embeddings).sum(dim=-1)
            distances = 1 - cos_sim
        else:
            raise ValueError(f"不支持的连续性类型: {self.continuity_type}")

        # 应用mask并计算平均损失
        masked_distances = distances * valid_pos_mask
        continuity_loss = masked_distances.sum() / (valid_pos_mask.sum() + 1e-8)

        return continuity_loss
