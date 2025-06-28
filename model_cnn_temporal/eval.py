from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch
from dataset import *
from model import *
import os
from torch.utils.data import DataLoader
from utils import *
from sklearn.metrics import f1_score, accuracy_score

@torch.no_grad()
def evaluate_model_val(model, dataloader, device):
    model.eval()
    criterion_action = nn.CrossEntropyLoss()
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_situation = nn.CrossEntropyLoss()

    total_loss = 0
    all_preds_action, all_labels_action = [], []
    all_preds_emotion, all_labels_emotion = [], []
    all_preds_situation, all_labels_situation = [], []

    for images, y_action, y_emotion, y_situation in dataloader:
        images = images.to(device)
        y_action = y_action.to(device)
        y_emotion = y_emotion.to(device)
        y_situation = y_situation.to(device)

        a_logits, e_logits, s_logits = model(images)

        loss_a = criterion_action(a_logits, y_action)
        loss_e = criterion_emotion(e_logits, y_emotion)
        loss_s = criterion_situation(s_logits, y_situation)
        loss = loss_a + loss_e + loss_s
        total_loss += loss.item()

        all_preds_action.extend(torch.argmax(a_logits, dim=1).cpu().tolist())
        all_preds_emotion.extend(torch.argmax(e_logits, dim=1).cpu().tolist())
        all_preds_situation.extend(torch.argmax(s_logits, dim=1).cpu().tolist())

        all_labels_action.extend(y_action.cpu().tolist())
        all_labels_emotion.extend(y_emotion.cpu().tolist())
        all_labels_situation.extend(y_situation.cpu().tolist())

    assert len(all_preds_action) == len(all_labels_action) == len(all_preds_emotion) == len(all_labels_emotion) == len(all_preds_situation) == len(all_labels_situation)

    macro_f1 = f1_score(
        all_labels_action + all_labels_emotion + all_labels_situation,
        all_preds_action + all_preds_emotion + all_preds_situation,
        average='macro',
        zero_division=0
    )
    micro_f1 = f1_score(
        all_labels_action + all_labels_emotion + all_labels_situation,
        all_preds_action + all_preds_emotion + all_preds_situation,
        average='micro',
        zero_division=0
    )

    partial_correct = sum(
        (p1 == l1 or p2 == l2 or p3 == l3)
        for p1, l1, p2, l2, p3, l3 in zip(
            all_preds_action, all_labels_action,
            all_preds_emotion, all_labels_emotion,
            all_preds_situation, all_labels_situation
        )
    )
    partial_score = partial_correct / len(all_labels_action)

    exact_match_acc = sum(
        (p1 == l1 and p2 == l2 and p3 == l3)
        for p1, l1, p2, l2, p3, l3 in zip(
            all_preds_action, all_labels_action,
            all_preds_emotion, all_labels_emotion,
            all_preds_situation, all_labels_situation
        )
    ) / len(all_labels_action)

    label_wise_acc = {
        'action': accuracy_score(all_labels_action, all_preds_action),
        'emotion': accuracy_score(all_labels_emotion, all_preds_emotion),
        'situation': accuracy_score(all_labels_situation, all_preds_situation),
    }

    avg_loss = total_loss / len(dataloader)

    return avg_loss, macro_f1, micro_f1, partial_score, exact_match_acc, label_wise_acc

@torch.no_grad()
def evaluate_model(config, split='test'):
    device = config['device']
    config['label_maps'] = get_label_maps_from_config(config)
    batch_size = config['batch_size']

    # 데이터셋 로딩
    dataset = get_dataset(config, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 모델 초기화
    model_class = MODEL_CLASSES.get(config['model_name'].lower(), MultiLabelVideoTransformerClassifier)
    model_wrapper = model_class(
        num_actions=len(config['label_maps']['action']),
        num_emotions=len(config['label_maps']['emotion']),
        num_situations=len(config['label_maps']['situation']),
        backbone_name=config.get('backbone_name', 'resnet18'),
        pretrained=False
    )
    model = model_wrapper.get_model()

    # 저장된 best 모델 로드
    model_path = os.path.join(config['save_path'], config['model_name'], config['best_model_path']) 
    assert os.path.exists(model_path), f"❌ 모델 경로 {model_path}가 존재하지 않습니다."

    checkpoint = torch.load(model_path, map_location=device)
    
    # 가중치만 저장된 경우 vs 전체 딕셔너리 저장된 경우 모두 처리
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded full checkpoint: epoch={checkpoint.get('epoch', '?')}, macro_f1={checkpoint.get('macro_f1', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights only")

    model.to(device)

    # 평가
    val_loss, macro_f1, micro_f1, partial_score, exact_match_acc, label_wise = evaluate_model_val(model, loader, device)

    print(f"\n✅🔍 Evaluation ({split.upper()} Set):")
    print(f"Loss: {val_loss:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Partial Match Score: {partial_score:.4f}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
    print(f"Label-wise Accuracy:\n  - Action: {label_wise['action']:.4f}  | Emotion: {label_wise['emotion']:.4f}  | Situation: {label_wise['situation']:.4f}")
