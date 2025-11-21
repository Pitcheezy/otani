# main.py
import sys
from pathlib import Path
import pandas as pd
import joblib

# 모듈 경로 설정
sys.path.append(str(Path(__file__).resolve().parent))

import config.config as cfg
# 아래 모듈들은 src 폴더에 파일이 생성되어 있어야 import 가능합니다.
# from src.data.loader import load_raw_data (나중에 구현 필요)
# from src.features.engineering import FeatureEngineer
# from src.models.trainer import PitchRecommendationTrainer

def main():
    print(">>> [1] 프로젝트 환경 설정 확인...")
    print(f"데이터 저장 경로: {cfg.DATA_DIR}")
    print(f"모델 저장 경로: {cfg.MODEL_DIR}")
    
    # 여기에 파이프라인 코드가 들어갈 예정입니다.
    # 1. 데이터 로드
    # 2. 전처리 & 피처 엔지니어링 (Leakage 제거)
    # 3. 모델 학습
    # 4. 모델 저장
    
    print("\\n>>> 환경 설정이 완료되었습니다. 이제 각 모듈을 구현하면 됩니다.")

if __name__ == "__main__":
    main()