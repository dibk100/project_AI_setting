# CNN-Temporal 🐾🐾
CNN-Temporal based multi-task video classification model   
🔄 TASK : 시계열 정보를 처리할 수 있는 모델로 확장
- CNN + LSTM   
- CNN + Transformer

## 🧩 Model Architecture Overview
### 1️⃣ MultiLabelVideoLSTMClassifier
```
Input Video Frames: [B, T, C, H, W]
       │
       ▼
CNN Backbone (e.g., ResNet18)
[각 프레임별 특징 추출 → B, T, D]
       │
       ▼
Step-wise LSTM 입력
- 방식: 프레임별로 CNN 추출 후 순차적으로 LSTM에 입력
- Hidden Dim: H
- Num Layers: L
       │
       ▼
LSTM Output Sequence: [B, T, H]
       │
       ▼
Mean Pooling (Across Time Axis)
       │
       ▼
Dropout (optional)
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼              
 Action Head     Emotion Head   Situation Head  (3-way Classifier)
 (Linear)        (Linear)       (Linear)

```

### 2️⃣ MultiLabelVideoTransformerClassifier -> 수정필요
```
Input Video Frames: [B, T, C, H, W]
       │
       ▼
CNN Backbone (e.g., ResNet18)
[각 프레임별 특징 추출 → B, T, D]
       │
       ▼
Positional Encoding (Temporal Order 반영)
       │
       ▼
Transformer Encoder
- Layers: N
- Heads: H
- Hidden Dim: D
       │
       ▼
Temporal Feature Sequence: [B, T, D]
       │
       ▼
Mean Pooling (Across Time Axis)
       │
       ▼
Dropout & LayerNorm (optional)
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
 Action Head     Emotion Head   Situation Head  (3-way Classifier)
 (MLP)           (MLP)          (MLP)
```

## 📁 Data Structure
```
data_image/
├── 20201028_cat-arch-000156.mp4/
│   ├── f20201028_cat-arch-000156.mp4.json     # 해당 영상 메타데이터 (프레임별 timestamp, keypoints, bbox 등)
│   ├── frame_12_timestamp_800.jpg
│   └── ... (프레임 이미지들)
├── cat-armstretch-080706/    
│   ├── cat-armstretch-080706.json
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   └── ... (프레임 이미지들)
└── ~    
```

```
data_image/
├── ARCH/
│   ├── 20201028_cat-arch-000156.mp4/
│   │   ├── f20201028_cat-arch-000156.mp4.json     # 해당 영상 메타데이터 (프레임별 timestamp, keypoints, bbox 등)
│   │   ├── frame_12_timestamp_800.jpg
│   │   └── ... (프레임 이미지들)
│   ├── cat-armstretch-080706/    
│   │   ├── cat-armstretch-080706.json
│   │   ├── frame_0.jpg
│   │   ├── frame_1.jpg
│   │   └── ... (프레임 이미지들)
└── ~    
```


## 📁 dataset.py
```
CatVideoDataset
 ├─ __getitem__ : (T, C, H, W) 텐서 반환
 ├─ frame 부족 시 padding
 ├─ action/emotion/situation 라벨 인코딩
 └─ PIL → Tensor 변환 transform 적용

get_dataset()
 └─ config로부터 label map, transform 구성 후 dataset 반환

collate_fn()
 └─ batch 단위 텐서 묶기 (frames, 3가지 라벨)
```

