# run_phase2_2.py
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.data.loader import load_raw_data
from src.features.engineering import FeatureEngineer
import config.config as cfg

def visualize_correlations(X, y):
    """피처 간 상관관계 및 타겟과의 관계 시각화"""
    
    # 시각화를 위해 X와 y를 잠시 합침
    df_vis = X.copy()
    df_vis['target_pitch'] = y
    
    plt.figure(figsize=(12, 10))
    
    # 상관관계 계산
    corr = df_vis.corr()
    
    # 히트맵 그리기
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Feature Correlation Matrix (Check for Leakage)')
    plt.show()

def main():
    print(">>> [Phase 2-2] Feature Engineering & Verification...")
    
    # 1. 데이터 로드
    df_raw = load_raw_data()
    if df_raw is None: return

    # 2. 피처 엔지니어링 수행
    engineer = FeatureEngineer()
    X, y, le = engineer.create_features(df_raw, is_training=True)
    
    print("-" * 40)
    print(f"Input Features (X) Shape: {X.shape}")
    print(f"Target Labels (y) Shape: {y.shape}")
    print("-" * 40)
    
    # 3. 중요: 데이터 누수 체크 (Leakage Check)
    # X 컬럼 중에 금지된 단어들이 포함되어 있는지 확인
    forbidden_words = ['speed', 'spin', 'plate', 'zone', 'event', 'des']
    leakage_cols = [col for col in X.columns if any(word in col.lower() for word in forbidden_words)]
    
    if leakage_cols:
        print(f"[WARNING] Data Leakage Detected! Suspicious columns: {leakage_cols}")
    else:
        print("[SUCCESS] No obvious leakage columns found in input features.")
        print(f"Final Feature List: {list(X.columns)}")

    # 4. 타겟 클래스 확인
    print("-" * 40)
    print("Encoded Classes (Pitch Types):")
    for idx, label in enumerate(le.classes_):
        count = (y == idx).sum()
        print(f"  ID {idx}: {label} (Count: {count})")
    
    # 5. 시각화 실행
    print(">>> [Phase 2-2] Visualizing Correlations...")
    visualize_correlations(X, y)
    
    print(">>> Phase 2-2 Complete.")

if __name__ == "__main__":
    main()