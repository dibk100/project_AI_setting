import argparse
from train import *
from eval import *
import yaml
import wandb
from dotenv import load_dotenv
import os
import torch.multiprocessing as mp

def main():
    load_dotenv(verbose=True)               # .env심기, wandb로그인 key
    parser = argparse.ArgumentParser(description="CNN+Transformer multi-head classification")
    parser.add_argument('--mode', choices=['train', 'eval','profile'], required=True)
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        
        # wandb 로그인
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            print("Warning: WANDB_API_KEY not set. wandb login skipped.")
            assert False, "WANDB_API_KEY environment variable is missing."
        
        train_model(config)
        wandb.finish()
    
    elif args.mode == 'profile':
        profile_one_batch(config)
        
    elif args.mode == 'eval':
        evaluate_model(config, split='test')

    else :
        assert False, "다시 코드 실행"
        
if __name__ == "__main__":
    main()