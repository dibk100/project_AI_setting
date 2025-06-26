import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Callable, Dict, Any
import torch
from utils import *
from glob import glob

class CatVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        label2idx_action: Optional[Dict[str, int]] = None,
        label2idx_emotion: Optional[Dict[str, int]] = None,
        label2idx_situation: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        max_frames: int = 150,                                      # 최대 프레임 150 확인함.
    ):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames

        self.label2idx_action = label2idx_action or {label: i for i, label in enumerate(self.df['cat_action'].unique())}
        self.label2idx_emotion = label2idx_emotion or {label: i for i, label in enumerate(self.df['cat_emotion'].unique())}
        self.label2idx_situation = label2idx_situation or {label: i for i, label in enumerate(self.df['owner_situation'].unique())}

        self.idx2label_action = {v: k for k, v in self.label2idx_action.items()}
        self.idx2label_emotion = {v: k for k, v in self.label2idx_emotion.items()}
        self.idx2label_situation = {v: k for k, v in self.label2idx_situation.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        video_folder = os.path.join(self.root_dir, row['file_path'])

        # 프레임 파일 경로 모두 수집 (정렬)
        frame_paths = sorted(glob(os.path.join(video_folder, "*.jpg")))
        frame_count = len(frame_paths)

        # 프레임 수 확인
        if frame_count != row['number of frames']:
            print(f"[Warning] Frame count mismatch for video {row['meta_json']} (CSV: {row['number of frames']}, Actual: {frame_count})")

        # 라벨 인덱스 변환
        label_action = self.label2idx_action[row['cat_action']]
        label_emotion = self.label2idx_emotion[row['cat_emotion']]
        label_situation = self.label2idx_situation[row['owner_situation']]

        frames = []
        for i, frame_path in enumerate(frame_paths[:self.max_frames]):
            if os.path.exists(frame_path):
                image = Image.open(frame_path).convert("RGB")

                # if i == 0 and idx % 500 == 0:
                #     print(f"[Debug] Video: {row['meta_json']}, frame: {os.path.basename(frame_path)}, size: {image.size}, mode: {image.mode}")

                if self.transform:
                    image_tensor = self.transform(image)
                    if i == 0 and idx % 500 == 0:
                        print(f"[Debug] Transformed Tensor - shape: {image_tensor.shape}, "
                            f"min: {image_tensor.min().item():.4f}, max: {image_tensor.max().item():.4f}, "
                            f"mean: {image_tensor.mean().item():.4f}")
                else:
                    image_tensor = transforms.ToTensor()(image)

                frames.append(image_tensor)

        # 프레임이 없는 경우
        if len(frames) == 0:
            print(f"[Warning] No frames loaded for video: {row['meta_json']} at index {idx}")

        # 패딩 처리
        if len(frames) < self.max_frames:
            pad_frame = torch.zeros_like(frames[0]) if frames else torch.zeros(3, 64, 64)
            while len(frames) < self.max_frames:
                frames.append(pad_frame)

        # 첫 프레임 텐서 상태 확인
        # if len(frames) > 0:
        #     print(f"First frame tensor stats - min: {frames[0].min():.4f}, max: {frames[0].max():.4f}, mean: {frames[0].mean():.4f}")

        frames_tensor = torch.stack(frames)  # (T, C, H, W)

        return {
            "meta_json": row['meta_json'],
            "frames": frames_tensor,
            "label_action": label_action,
            "label_emotion": label_emotion,
            "label_situation": label_situation
        }

def get_dataset(config: dict, split: str = 'train') -> Dataset:
    csv_path = config['data'][f'{split}_csv']
    root_dir = config['data'].get('root_dir', './data')
    max_frames = config.get('max_frames', 150)

    label_maps = get_label_maps_from_config(config)

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    dataset = CatVideoDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        label2idx_action=label_maps['action'],
        label2idx_emotion=label_maps['emotion'],
        label2idx_situation=label_maps['situation'],
        transform=transform,
        max_frames=max_frames,
    )
    return dataset

def collate_fn(batch):
    frames = torch.stack([item['frames'] for item in batch])  # (B, T, C, H, W)
    labels_action = torch.tensor([item['label_action'] for item in batch])
    labels_emotion = torch.tensor([item['label_emotion'] for item in batch])
    labels_situation = torch.tensor([item['label_situation'] for item in batch])
    return frames, labels_action, labels_emotion, labels_situation
