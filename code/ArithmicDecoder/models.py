import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import EnhancedTokenizer
from config import Config


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
        self, prompt: str, temperature: float = 0.8, verbose: bool = False, top_k: int = 0, top_p: float = 0.0
    ) -> str:
        with torch.no_grad():
            # 编码输入序列
            input_seq = self.tokenizer.encode(prompt)
            answer_start_pos = len(input_seq)
            
            if verbose:
                print("\n生成过程:")
                print(f"初始提示: {prompt}")
                print(f"编码后的提示: {input_seq}")
                print(f"答案开始位置: {answer_start_pos}")
            
            # 确保不超过最大长度
            max_length = min(35, self.config.max_seq_len)
            
            # 初始化输入张量，使用pad token填充到最大长度
            padded_input = input_seq.copy()  # 复制输入序列
            current_pos = answer_start_pos - 1  # 修改：从答案开始位置的前一个位置开始预测
            
            # 生成答案
            while current_pos < max_length - 2:  # 修改：保留两个位置，一个给预测token一个给结束符
                # 准备当前输入
                current_input = padded_input + [self.tokenizer.pad_id] * (max_length - len(padded_input))
                input_tensor = torch.tensor([current_input], dtype=torch.long, device=self.config.device)
                
                if verbose:
                    print("\n" + "-" * 50)
                    print(f"当前位置: {current_pos}")
                    print(f"当前序列: {self.tokenizer.decode(padded_input)}")
                
                # 获取模型输出
                logits = self.model(input_tensor)
                # 修改：现在我们使用current_pos预测下一个位置
                next_token_logits = logits[0, current_pos]
                
                # 应用sampling策略
                next_token = self._sample_next_token(
                    next_token_logits, 
                    temperature=temperature, 
                    top_k=top_k,
                    top_p=top_p,
                    verbose=verbose
                )
                
                # 更新序列 - 将预测的token放在当前位置的下一个位置
                if len(padded_input) <= current_pos + 1:
                    padded_input.append(next_token)
                else:
                    # 如果下一个位置已经有token，替换它
                    padded_input[current_pos + 1] = next_token
                
                current_pos += 1  # 向前移动位置
                
                # 如果生成了结束符，停止生成
                if next_token == self.tokenizer.eos_id:
                    if verbose:
                        print("\n检测到结束符，停止生成")
                    break
            
            # 解码最终序列
            final_output = self.tokenizer.decode(padded_input)
            if verbose:
                print("\n" + "=" * 50)
                print(f"最终生成结果: {final_output}")
                print("=" * 50)
            return final_output
            
    def _sample_next_token(self, logits, temperature=0.8, top_k=0, top_p=0.0, verbose=False):
        """
        使用多种采样策略选择下一个token
        
        参数:
            logits: 当前位置的logits
            temperature: softmax的温度系数
            top_k: 如果>0，只从概率最高的k个token中采样
            top_p: 如果>0，使用nucleus sampling (只从累积概率达到p的token子集中采样)
            verbose: 是否打印详细信息
        """
        # 应用temperature
        scaled_logits = logits / temperature
        
        # 应用top_k采样
        if top_k > 0:
            top_k = min(top_k, scaled_logits.size(-1))
            indices_to_remove = torch.topk(scaled_logits, top_k)[0][-1].unsqueeze(-1)
            scaled_logits = torch.where(
                scaled_logits < indices_to_remove,
                torch.ones_like(scaled_logits) * float('-inf'),
                scaled_logits
            )
        
        # 应用top_p (nucleus) 采样
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过阈值的token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过阈值的token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 散置回原始索引顺序
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
        
        # 计算概率分布
        probs = F.softmax(scaled_logits, dim=-1)
        
        if verbose:
            # 获取top-5概率和对应的token
            top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
            print("\nTop-5候选词:")
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.itos[idx.item()]
                print(f"Token: {token:<4} (id: {idx.item():<2}) 概率: {prob.item():.4f}")
        
        # 采样下一个token
        next_token = torch.multinomial(probs, 1).item()
        
        if verbose:
            chosen_prob = probs[next_token].item()
            chosen_token = self.tokenizer.itos[next_token]
            print(f"\n选择的token: {chosen_token} (id: {next_token}) 概率: {chosen_prob:.4f}")
            
        return next_token


