import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss:
    """
    组合损失函数，包含交叉熵损失和嵌入连续性损失
    """

    def __init__(
        self,
        continuity_weight=0.1,
        continuity_type="l2",
        normalize_embeddings=True,
        apply_to_digits_only=True,
        digit_tokens=None,
    ):
        """
        初始化组合损失函数

        参数:
        - continuity_weight: 连续性损失的权重系数
        - continuity_type: 距离计算类型，'l1', 'l2', 或 'cosine'
        - normalize_embeddings: 是否在计算连续性时归一化嵌入向量
        - apply_to_digits_only: 是否仅对数字token应用连续性损失
        - digit_tokens: 数字token的ID列表
        """
        self.continuity_weight = continuity_weight
        self.continuity_type = continuity_type
        self.normalize_embeddings = normalize_embeddings
        self.apply_to_digits_only = apply_to_digits_only
        self.digit_tokens = set(digit_tokens) if digit_tokens else None

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
        - 总损失、交叉熵损失和连续性损失
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        )
        masked_ce_loss = (ce_loss * masks.view(-1)).sum() / masks.sum()

        # 如果连续性权重为0，直接返回交叉熵损失
        if self.continuity_weight == 0:
            return (
                masked_ce_loss,
                masked_ce_loss,
                torch.tensor(0.0, device=logits.device),
            )

        # 使用answer_masks如果提供，否则使用masks
        answer_masks = answer_masks if answer_masks is not None else masks

        # 获取模型的token嵌入层
        token_embedding = (
            model.module.token_embed if hasattr(model, "module") else model.token_embed
        )

        # 获取目标token的嵌入
        # 我们获取实际token的嵌入，而不是预测的概率分布
        batch_size, seq_len = targets.shape
        embeddings = token_embedding(targets)  # [batch_size, seq_len, d_model]

        # 计算连续性损失
        continuity_loss = self._compute_continuity_loss(
            embeddings, targets, answer_masks, batch_size, seq_len
        )

        # 组合损失
        total_loss = masked_ce_loss + self.continuity_weight * continuity_loss

        return total_loss, masked_ce_loss, continuity_loss

    def _compute_continuity_loss(self, embeddings, targets, masks, batch_size, seq_len):
        """计算嵌入连续性损失"""
        # 准备用于计算损失的mask
        # 我们只对相邻位置计算连续性损失，最后一个位置没有后续位置
        valid_pos_mask = masks[:, :-1] * masks[:, 1:]

        # 如果需要只对数字token应用损失
        if self.apply_to_digits_only and self.digit_tokens:
            is_digit_current = torch.zeros_like(targets, dtype=torch.bool)
            is_digit_next = torch.zeros_like(targets, dtype=torch.bool)

            for digit in self.digit_tokens:
                is_digit_current = is_digit_current | (targets == digit)
                is_digit_next = is_digit_next | (targets == digit)

            digit_mask_current = is_digit_current[:, :-1]
            digit_mask_next = is_digit_next[:, 1:]

            # 只有当前位置和下一个位置都是数字时才应用损失
            digit_pair_mask = digit_mask_current & digit_mask_next
            valid_pos_mask = valid_pos_mask & digit_pair_mask

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
