# project_AI_PetBabel 🐾💬
**Subject**: 
- A Multi-task AI Model for Cat Language Translation  
  (Translating Cat Behaviors and Emotions into Natural Language)   

**Purpose**:
- 고양이 동영상에서 행동, 감정, 상황을 자동 인식하고, 이를 바탕으로 고양이 ‘언어’를 자연어로 생성하는 AI 시스템 개발   

**Goal**: 
- 반려동물의 상태를 이해하고 소통할 수 있는 인터페이스 구현

## 📋 To-Do
- [ ] 🔄 Data : 데이터 전처리  
- [ ] 🔄 Task01 : 반려동물 분류 모델(CNN+Temporal) 개발   
- [ ] 🕒 Task02 : 반려동물 언어 번역기(LLM) 개발
<!--
### To-Do

- [ ] 🔄 작업 중 : 품질 예측 성능 평가 코드 개선 중
- [ ] ✅ 완료됨 : 데이터셋 병합 및 전처리 (2025-05-23)
- [ ] 📌🕒 다음 할 일 : inference 모듈 디버깅
-->


## ⚙️ 전체 파이프라인
```
[1] 동영상, 고양이 개체 정보 입력
    ↓
[2] TASK01 : 비전 모델 (감정/행동/상황 인식)
    ↓
[3] 1차 출력(감정/행동/상황 라벨) + 사용자 개체 정보 결합 :: Prompt Engineering
    ↓
[4] TASK02 : LLM 입력 → 자연어 출력
    ex) “저 낯선 사람 뭐야? 기분 나빠.”
```

## 🧪 주요 연구 내용

- **TASK01 : 영상 기반 고양이 상태 인식**  
  CNN + LSTM 기반 멀티태스크 분류기로 고양이 행동, 감정, 상황 라벨 자동 추출  
  (입력: 고양이 동영상 / 출력: 행동·감정·상황 라벨 및 keypoints(추후 작업) 분석)

- **2차 단계: LLM 기반 고양이 언어 생성**  
  TASK01 출력과 고양이 메타정보(품종, 성별, 나이 등)를 활용해  
  자연어 ‘고양이 언어’ 생성 (GPT-4o, Flamingo 등 최신 LLM 실험 예정)

### 📁 Folder Structure
```
project_AI_PetBabel/
│
├── data/                    # 데이터 관련: 원본, 전처리, EDA 등
│   ├── raw/                 # 원본 데이터 (프레임, json 등)
│   ├── processed/           # 전처리된 데이터 (예: resize, crop 등)
│   ├── meta/                # 메타 정보 및 분석 결과
│   └── scripts/             # 데이터 처리용 스크립트
│
├── model_cnn_temporal/     # CNN + LSTM 기반 비디오 시계열 모델
│   ├── single_task/         # 행동 / 감정 / 상황 단일 태스크 실험
│   ├── multi_task/          # 멀티태스크 모델 실험
│   └── scripts/             # 학습/평가 코드
│
├── model_llm_translator/   # LLM 기반 반려동물 언어 번역기 파트
│   ├── prompts/             # 프롬프트 디자인 및 실험
│   ├── models/              # 사용한 LLM 모델과 파인튜닝 코드
│   └── eval/                # 번역기 평가 방식 및 결과
│
├── notebooks/              # ipynb 실험 기록
│
├── app/                    # 실제 구현
│
└── README.md               # 프로젝트 설명

```
