# config/config.py
import os
from pathlib import Path

# 프로젝트 루트 디렉토리 자동 인식
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    "end_date": "2025-11-01"
}

# 모델 설정
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,        # 과적합 방지를 위해 깊이 제한
    'min_samples_split': 5,
    'random_state': 42
}

RANDOM_SEED = 42