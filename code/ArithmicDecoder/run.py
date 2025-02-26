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
from models import EnhancedTransformer, SequenceGenerator
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

    parser.add_argument("--local-rank", "--local_rank", type=int)
    
    args = parser.parse_args()
    config = Config()
    print(f"Using device: {config.device}")

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    if args.mode == "train":
        model = train_enhanced_model(config, args.base)
        print("Training completed. Model saved as 'best_model.pth'")

    elif args.mode == "eval":
        # 加载模型
        model_path = args.base if args.base else "best_model.pth"
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            return

        model = EnhancedTransformer(config, EnhancedTokenizer(config))
        model = nn.DataParallel(model)
        model = model.to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"Loaded model from {model_path}")

        # 评估模型
        accuracy, _ = evaluate_model(model, config, args.samples, verbose=args.verbose)


if __name__ == "__main__":
    torch.set_printoptions(threshold=np.inf)
    main()
