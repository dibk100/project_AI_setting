import os
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
import pandas as pd

# ---------------------- 설정 ----------------------
# YOLOv8 모델 로드 (n: 가장 가벼운 모델)
model = YOLO('yolov8n.pt')

# 이미지 데이터 루트 경로
base_path = "./data_image"

# COCO 클래스 중 'cat'의 클래스 ID (16번)
cat_class_id = 15  # zero-based index

# 결과 저장 파일 경로
save_path = "./cat_detection_frame_counts.csv"

# ---------------------- 메인 로직 ----------------------
results_summary = []

# data_image 하위 폴더 순회
for category_dir in os.listdir(base_path):
    category_path = os.path.join(base_path, category_dir)
    if not os.path.isdir(category_path):
        continue

    # 각 영상 단위 폴더 순회
    for video_folder in os.listdir(category_path):
        video_path = os.path.join(category_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        frames = sorted(glob(os.path.join(video_path, "*.jpg")))
        cat_frame_count = 0

        for frame_path in tqdm(frames, desc=f"[{category_dir}] {video_folder}", leave=False):
            try:
                results = model(frame_path, verbose=False)[0]
                if any(det.cls == cat_class_id for det in results.boxes):
                    cat_frame_count += 1
            except Exception as e:
                print(f"⚠️ Error on {frame_path}: {e}")
                continue

        print(f"[✓] {video_folder}: {cat_frame_count} / {len(frames)} 프레임에 고양이 감지됨")

        results_summary.append({
            "json_file": video_folder + ".json",
            "file_path": video_path,
            "number_of_frames": len(frames),
            "real_cat": cat_frame_count,
        })

# ---------------------- 저장 ----------------------
df_results = pd.DataFrame(results_summary)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_results.to_csv(save_path, index=False)
print(f"\n✅ 고양이 프레임 감지 결과 저장 완료: {save_path}")
