import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from models import EnhancedTransformer
from config import Config
from tokenizer import EnhancedTokenizer
from eval import evaluate_model
from train import train_enhanced_model

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed generation process",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_local.yaml",
        help="Path to config file (default: config_local.yaml)",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save default config template to config_template.yaml",
    )

    parser.add_argument("--local-rank", "--local_rank", type=int)
    
    args = parser.parse_args()
    
    # 生成默认配置模板
    if args.save_config:
        Config.save_default_yaml("config_template.yaml")
        print("默认配置模板已保存到 config_template.yaml")
    
    # 加载配置
    config = Config.from_yaml(args.config)
    
    print(f"使用设备: {config.device}")
    
    if args.mode == "train":
        model = train_enhanced_model(config, args.base)
        print("训练完成。模型已保存为 'best_model.pth'")

    elif args.mode == "eval":
        # 加载模型
        model_path = args.base if args.base else "best_model.pth"
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在!")
            return

        model = EnhancedTransformer(config, EnhancedTokenizer(config))
        model = nn.DataParallel(model)
        model = model.to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"已从 {model_path} 加载模型")

        # 评估模型
        accuracy, _ = evaluate_model(model, config, args.samples, verbose=args.verbose)


if __name__ == "__main__":
    torch.set_printoptions(threshold=np.inf)
    main()
