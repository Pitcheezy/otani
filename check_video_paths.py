"""영상 경로 확인 스크립트"""
from pathlib import Path
import sys
sys.path.insert(0, '.')
import config.config as cfg
import pandas as pd

# CSV에서 샘플 경로 확인
csv_path = cfg.FINAL_MERGED_CSV
df = pd.read_csv(csv_path)

# 2025-10-28 날짜의 샘플 확인
sample = df[df['game_date'].str.contains('2025-10-28', na=False)].head(2)
print("=== CSV의 샘플 경로 ===")
for idx, row in sample.iterrows():
    print(f"\nRow {idx}:")
    print(f"  output_video_path: {row['output_video_path']}")
    
    # 경로 변환 테스트
    path_str = str(row['output_video_path'])
    if "/" in path_str:
        filename = path_str.split("/")[-1]
    elif "\\" in path_str:
        filename = path_str.split("\\")[-1]
    else:
        filename = path_str
    
    print(f"  추출된 파일명: {filename}")
    
    # 경로 확인
    path1 = cfg.ANALYZED_VIDEOS_DIR / filename
    path2 = cfg.DE_RESULTS_DIR / "2025_data" / "videos" / filename
    
    print(f"  경로1 (analyzed_videos): {path1}")
    print(f"    존재: {path1.exists()}")
    print(f"  경로2 (2025_data/videos): {path2}")
    print(f"    존재: {path2.exists()}")

