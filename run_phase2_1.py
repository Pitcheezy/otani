# run_phase2_1.py
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# src 모듈 import를 위한 경로 설정
sys.path.append(str(Path(__file__).resolve().parent))

from src.data.loader import load_raw_data
import config.config as cfg

def visualize_data(df):
    """수집된 데이터의 기초 통계를 시각화합니다."""
    
    # 시각화 스타일 설정
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif' # 한글 폰트 깨짐 방지용 (영문 권장)
    
    # 1. 구종별 투구 수 (Bar Chart)
    plt.figure(figsize=(10, 5))
    sns.countplot(x='pitch_type', data=df, order=df['pitch_type'].value_counts().index, palette='viridis')
    plt.title('Distribution of Pitch Types')
    plt.xlabel('Pitch Type')
    plt.ylabel('Count')
    plt.show()
    
    # 2. 구종별 구속 분포 (Box Plot)
    # release_speed가 있는 데이터만 필터링
    if 'release_speed' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='pitch_type', y='release_speed', data=df, palette='coolwarm')
        plt.title('Velocity Distribution by Pitch Type')
        plt.xlabel('Pitch Type')
        plt.ylabel('Velocity (mph)')
        plt.show()
    
    # 3. 스트라이크 존 투구 위치 (Scatter Plot)
    # 포수 시점(plate_x, plate_z)
    if 'plate_x' in df.columns and 'plate_z' in df.columns:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='plate_x', y='plate_z', hue='pitch_type', data=df, alpha=0.6, s=30)
        
        # 스트라이크 존 가이드라인 (대략적)
        plt.plot([-0.85, 0.85], [1.5, 1.5], color='red', linestyle='--') # 하단
        plt.plot([-0.85, 0.85], [3.5, 3.5], color='red', linestyle='--') # 상단
        plt.plot([-0.85, -0.85], [1.5, 3.5], color='red', linestyle='--') # 좌측
        plt.plot([0.85, 0.85], [1.5, 3.5], color='red', linestyle='--') # 우측
        
        plt.title('Pitch Location (Catcher\'s View)')
        plt.xlabel('Horizontal Location (ft)')
        plt.ylabel('Vertical Location (ft)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('equal')
        plt.show()

def main():
    print(">>> [Phase 2-1] Collecting Data...")
    
    # 1. 데이터 로드
    df = load_raw_data(force_update=True) # 최초 실행이므로 강제 업데이트
    
    if df is None:
        print("[ERROR] Data collection failed. Exiting.")
        return

    # 2. 기본 정보 출력
    print("-" * 40)
    print(f"Data Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])} ...")
    print("-" * 40)
    
    # 3. 결측치 확인 (중요한 컬럼만)
    check_cols = ['pitch_type', 'release_speed', 'plate_x', 'plate_z', 'balls', 'strikes']
    print("[INFO] Missing Values Summary:")
    print(df[check_cols].isnull().sum())
    
    # 4. 시각화 실행
    print(">>> [Phase 2-1] Visualizing Data...")
    print("[INFO] Generating plots... (Check popup windows)")
    visualize_data(df)
    
    print(">>> Phase 2-1 Complete.")

if __name__ == "__main__":
    main()