# check_env.py
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# config 모듈 인식을 위해 경로 추가
sys.path.append(str(Path(__file__).resolve().parent))

try:
    import config.config as cfg
    print("[Success] Config 모듈 로드 성공")
    print(f"   - 프로젝트 루트: {cfg.PROJECT_ROOT}")
    print(f"   - 데이터 경로: {cfg.RAW_DATA_DIR}")
except ImportError as e:
    print(f"[Error] Config 로드 실패: {e}")

# 라이브러리 버전 체크
import pandas as pd
import pybaseball
print(f"[Check] Pandas version: {pd.__version__}")
print(f"[Check] Pybaseball version: {pybaseball.__version__}")

# 시각화 라이브러리 체크
try:
    sns.set_theme()
    print("[Success] Seaborn/Matplotlib 설정 완료")
except Exception as e:
    print(f"[Error] 시각화 라이브러리 오류: {e}")

print("\n환경 설정 완료")