# project_AI_PetBabel 🐾🐾
- Subject :A Multi-task AI Model for Understanding Cat Behavior and Emotion(고양이 행동과 감정 이해를 위한 다중 태스크 AI 모델 개발 )

## 🔄 To-Do
- Task01 : 반려동물 분류 모델(CNN+Temporal) 개발
- Task02 : 반려동물 언어 번역기(LLM) 개발

## 📦 Data Description
- [AI-Hub : 반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59)

## 📁 Folder Structure
```
project_AI_PetBabel/     
├── DataSetting/       
├── TASK01_Classifier/               
├── TASK02_Translator/               
├── main.py                     # CLI 실행 파일
├── video_util.py               # [1] 입력 처리
├── vision_model.py             # [2] 감정/행동/상황 분류
├── llm_generator.py            # [3] 자연어 생성
└── requirements.txt    
```