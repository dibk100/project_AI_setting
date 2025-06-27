import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import os
import torch
import random
import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_best_model(model, optimizer, scheduler, save_dir, base_name, epoch, val_loss, score):
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    ckpt_dir = os.path.join(save_dir, base_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    model_filename = f"{timestamp}_epoch{epoch}_valLoss{val_loss:.4f}_macro_f1_{score:.4f}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'macro_f1': score,
    }, ckpt_path)

    print(f"✅ Best model saved: {ckpt_path} (macro_f1: {score:.4f})")
    return 


def get_label_maps_from_config(config):
    return {
        'action': {label: i for i, label in enumerate(config['label_names']['action'])},
        'emotion': {label: i for i, label in enumerate(config['label_names']['emotion'])},
        'situation': {label: i for i, label in enumerate(config['label_names']['situation'])},
    }