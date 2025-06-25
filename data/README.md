# 📊 Data Overview
> 데이터 분석 및 전처리 작업 공간.

## 📦 Data Description
- [AI-Hub : 반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aihubdata/data/view.do?-currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59).   
- Training/라벨링데이터/CAT :: data_json
- Training/원천데이터/CAT :: data_image
    ```
    aihubshell -mode d -datasetkey 59 -filekey 42287,42288,42289,42290,42202,42203,42204,42205,42206,42207,42208,42209
    ```
- 🔁To-Do : validation 데이터로 TEST셋 구축.


###  📁 data Folder Structure
```
data/     
├── data_fin/                       # 목적에 맞게 전처리한 파일
├── data_image/               
├── data_json/               
├── dataEDA.ipynb                   # 데이터 분석
├── dataProcessing.ipynb            # 데이터 전처리
├── dataStrategy.py                 # 데이터 전처리 - 불균형 전략
└── README.md    
```
   
<details>
<summary>data_image structure(raw dataset)</summary>

### 📁 RAW DATA(data_image) Structure
```
data_image/
├── arch/
│    ├── 20201030_cat-armstretch-000253.mp4/            # (폴더)
│    │    ├── frame_0_timestamp_0.jpg
│    │    └── ~~.jpg
│    ├── cat-armstretch-013346/
│    │    ├── frame_0_timestamp_0.jpg
│    │    └── ~~.jpg
│
├── cat_sleep/
│    ├── ~~~.mp4/            # (폴더)
│    │    ├── frame_0_timestamp_0.jpg
│    │    └── ~~.jpg
│    ├── cat-sleep-013346/
│    │    ├── frame_0_timestamp_0.jpg
│    │    └── ~~.jpg
│
├── jump/
...
```
</details>


###  📊 EDA Summary (`dataEDA.ipynb`)

- **총 JSON 파일 수 (원본 영상 수)**: `21,544`
- **1차 전처리**:  
  - 기준: `프레임 수 80~110` 사이 영상만 유지
  - ✅ **잔존 JSON 파일 수**: `8,976`개

- **2차 전처리 (라벨 정제)**:  
  - **Action**: 12개 클래스  
  - **Emotion**: 6개 → 5개로 통합 (공포 + 불안 / 슬픔 통합)  
  - **Situation**: 17개 → 12개로 축소 (불균형 제거 및 유사 클래스 통합)

### 🧩 Preprocessing Strategy (`dataProcessing.ipynb`, `dataStrategy.py`)

- **목표**: 멀티 태스크 (행동 + 감정 + 상황) 분류를 위한 안정된 학습 데이터 구성

- **ISSUE**:
  - 3개 태스크의 라벨 조합을 기준으로 하면 **클래스 불균형 심각**
  - 단순 무작위 샘플링 시 데이터 손실 또는 과소/과대표집 우려

- **전략 제안**:
  - **언더샘플링** 기반 클래스 균형 조정
  - **YOLO 기반 고양이 검출 비율** 고려한 정교한 샘플링
    - 예: `cat-armstretch-037365`는 90프레임 중 34프레임에 고양이 검출됨 → 고양이 등장 비율 높은 영상과 적은 영상을 적절히 선택
