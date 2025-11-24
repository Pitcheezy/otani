from pathlib import Path

# 프로젝트 루트 디렉토리 자동 인식
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === 여기부터 추가 ===
# 상위 작업 폴더 (예: "새 폴더 (2)")
WORKSPACE_ROOT = PROJECT_ROOT.parent

# 형제 프로젝트: data_extraction_mlb 루트
DATA_EXTRACTION_ROOT = WORKSPACE_ROOT / "data_extraction_mlb"

# 그 안의 results 디렉토리 및 주요 파일/폴더
DE_RESULTS_DIR = DATA_EXTRACTION_ROOT / "results"
FINAL_MERGED_CSV = DE_RESULTS_DIR / "FINAL_ohtani_data_with_video_analysis.csv"
VIDEO_ANALYSIS_CSV = DE_RESULTS_DIR / "video_analysis_results.csv"
ANALYZED_VIDEOS_DIR = DE_RESULTS_DIR / "analyzed_videos"

# 원본 영상 디렉토리 (data/raw/videos/ohtani_videos)
ORIGINAL_VIDEOS_DIR = DATA_EXTRACTION_ROOT / "data" / "raw" / "videos" / "ohtani_videos"
# === 추가 끝 ===

# 디렉토리 경로 설정
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# 디렉토리 자동 생성 (없으면 생성)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 수집 설정
DATA_COLLECTION = {
    "player_id": 660271,  # 오타니 쇼헤이
    "start_date": "2025-03-01",
    "end_date": "2025-11-01",
}

# 모델 설정
MODEL_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,  # 과적합 방지를 위해 깊이 제한
    "min_samples_split": 5,
    "random_state": 42,
}

RANDOM_SEED = 42