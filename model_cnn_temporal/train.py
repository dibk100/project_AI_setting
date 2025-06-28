import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import wandb
import torch.nn as nn
from utils import *
from dataset import *
from model import *
from eval import *

def train_model(config):
    torch.cuda.empty_cache()
    set_seed(config['seed'])
    device = config['device']
    
    # 자동 run_name 생성
    run_name = f"{config['model_name']}_lr{config['learning_rate']}_bs{config['batch_size']}_ep{config['epochs']}"
    wandb.init(project=config['wandb_project'], name=run_name, config=config)

    # 데이터셋 & DataLoader
    train_dataset = get_dataset(config, split='train')
    val_dataset = get_dataset(config, split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,collate_fn=collate_fn,pin_memory=True)      # pin_memory=True :: GPU 사용시 병목 줄임
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn,num_workers=0,pin_memory=True)

    # 모델 초기화
    model_class = MODEL_CLASSES.get(config['model_name'].lower(), MultiLabelVideoTransformerClassifier)
    model_wrapper = model_class(
        num_actions=len(config['label_names']['action']),
        num_emotions=len(config['label_names']['emotion']),
        num_situations=len(config['label_names']['situation']),
        backbone_name=config.get('backbone_name', 'resnet18'),
        pretrained=config.get('pretrained', True),
        debug=True
    )
    model = model_wrapper.get_model()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    num_training_steps = config['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion_action = nn.CrossEntropyLoss()
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_situation = nn.CrossEntropyLoss()

    best_score = 0
    
    # conv1 출력 저장용
    conv1_outputs = []
    def hook_fn(module, input, output):
        conv1_outputs.append(output.detach().cpu())
    # hook 등록
    model.shared_encoder.conv1.register_forward_hook(hook_fn)
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for batch_idx, (frames, y_action, y_emotion, y_situation) in enumerate(train_loader):
            optimizer.zero_grad()
            
            frames = frames.to(device)
            y_action = y_action.to(device)
            y_emotion = y_emotion.to(device)
            y_situation = y_situation.to(device)
            outputs_action, outputs_emotion, outputs_situation = model(frames,epoch=epoch, batch_idx=batch_idx)
            
            loss_a = criterion_action(outputs_action, y_action)
            loss_e = criterion_emotion(outputs_emotion, y_emotion)
            loss_s = criterion_situation(outputs_situation, y_situation)

            loss = loss_a + loss_e + loss_s
            loss.backward()

            if batch_idx%100==0:
                print(f"[conv1 output] shape: {conv1_outputs[0].shape}")
                print(f"[conv1 output] mean: {conv1_outputs[0].mean().item():.4f}, std: {conv1_outputs[0].std().item():.4f}, max: {conv1_outputs[0].max().item():.4f}")
                conv1_outputs.clear()  # 다음 배치 출력을 위해 비워줌
            
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f}")

        print("----logging : 검증 평가 전----")
        # 검증 평가
        val_loss, macro_f1, micro_f1, partial_score, exact_match_acc, label_wise_acc = evaluate_model_val(model, val_loader, config['device'])
        print(
            f"Epoch {epoch} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Micro F1: {micro_f1:.4f} | "
            f"Partial Match Score: {partial_score:.4f} | "
            f"Exact Match Acc: {exact_match_acc:.4f} | "
            f"Label-wise Acc: {label_wise_acc}"
        )
        print("----logging : wandb----")
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'partial_score': partial_score,
            'exact_match_acc': exact_match_acc,
            'label_wise_acc/action': label_wise_acc['action'],
            'label_wise_acc/emotion': label_wise_acc['emotion'],
            'label_wise_acc/situation': label_wise_acc['situation'],
        })

        # ✅ Macro F1 기준으로 모델 저장
        if macro_f1 > best_score:
            best_score = macro_f1
            save_best_model(
                model,
                optimizer,
                lr_scheduler,
                save_dir=config['save_path'],
                base_name=config['model_name'],
                epoch=epoch,
                val_loss=val_loss,
                score=best_score,
            )

import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_one_batch(config):
    set_seed(config['seed'])
    device = config['device']

    # 데이터셋 & DataLoader
    train_dataset = get_dataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # 모델 초기화
    model_class = MODEL_CLASSES.get(config['model_name'].lower(), MultiLabelVideoTransformerClassifier)
    model_wrapper = model_class(
        num_actions=len(config['label_names']['action']),
        num_emotions=len(config['label_names']['emotion']),
        num_situations=len(config['label_names']['situation']),
        backbone_name=config.get('backbone_name', 'resnet18'),
        pretrained=config.get('pretrained', True),
        debug=True
    )
    model = model_wrapper.get_model()
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']))
    criterion_action = nn.CrossEntropyLoss()
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_situation = nn.CrossEntropyLoss()

    # 한 배치만 프로파일링
    for batch_idx, (frames, y_action, y_emotion, y_situation) in enumerate(train_loader):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("batch_train"):
                optimizer.zero_grad()
                frames = frames.to(device)
                y_action = y_action.to(device)
                y_emotion = y_emotion.to(device)
                y_situation = y_situation.to(device)

                outputs_action, outputs_emotion, outputs_situation = model(frames, epoch=0, batch_idx=batch_idx)

                loss_a = criterion_action(outputs_action, y_action)
                loss_e = criterion_emotion(outputs_emotion, y_emotion)
                loss_s = criterion_situation(outputs_situation, y_situation)
                loss = loss_a + loss_e + loss_s

                loss.backward()
                optimizer.step()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        break  # 한 배치만 프로파일링 후 종료
