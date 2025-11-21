# run_phase5_rl.py
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.data.loader import load_raw_data
from src.features.engineering import FeatureEngineer
from src.models.trainer import PitchRecommendationTrainer
from src.models.location_trainer import PitchLocationTrainer
import config.config as cfg

def visualize_pitch_location(pitch_name, pred_loc, actual_locs=None):
    """예측된 투구 위치를 스트라이크 존 히트맵으로 시각화"""
    plt.figure(figsize=(6, 6))
    
    # 1. 스트라이크 존 그리기 (포수 시점)
    plt.plot([-0.83, 0.83], [1.5, 1.5], 'k-', lw=2)
    plt.plot([-0.83, 0.83], [3.5, 3.5], 'k-', lw=2)
    plt.plot([-0.83, -0.83], [1.5, 3.5], 'k-', lw=2)
    plt.plot([0.83, 0.83], [1.5, 3.5], 'k-', lw=2)
    
    # 2. 실제 데이터 분포 (KDE Plot) - 해당 구종의 일반적인 분포
    if actual_locs is not None:
        sns.kdeplot(x=actual_locs['plate_x'], y=actual_locs['plate_z'], 
                   fill=True, cmap="Blues", alpha=0.3, thresh=0.1)
    
    # 3. AI 예측 위치 (Point)
    plt.scatter(pred_loc[0], pred_loc[1], color='red', s=200, marker='X', label='AI Target')
    
    plt.xlim(-2.5, 2.5)
    plt.ylim(0, 5)
    plt.title(f"AI Recommended Location: {pitch_name}")
    plt.xlabel("Horizontal (ft)")
    plt.ylabel("Vertical (ft)")
    plt.legend()
    plt.axis('equal')
    plt.show()

def main():
    print(">>> [Phase 5] RL-based Training & Location Visualization...")
    
    # 1. 데이터 로드
    df = load_raw_data()
    if df is None: return
    
    # 2. 피처 엔지니어링 (RL 필터링 적용!)
    engineer = FeatureEngineer()
    # use_rl_filter=True: 잘 던진 공(Reward 상위 50%)만 학습함
    X, y_type, le_type, y_locs = engineer.create_features(df, is_training=True, use_rl_filter=True)
    
    print("-" * 40)
    print(f"[Data] Filtered Training Samples: {len(X)}")
    print("-" * 40)
    
    # 3. 구종 모델 학습 (Type Model)
    type_trainer = PitchRecommendationTrainer()
    type_trainer.train(X, y_type)
    type_trainer.save_model("pitch_model_rl.joblib")
    
    # 4. 위치 모델 학습 (Location Model)
    # 위치 예측을 위해선 '어떤 구종을 던질지(y_type)'도 입력 정보로 중요함
    X_loc = X.copy()
    X_loc['pitch_type_code'] = y_type # 정답 구종을 피처로 추가해서 학습
    
    loc_trainer = PitchLocationTrainer()
    loc_trainer.train(X_loc, y_locs)
    loc_trainer.save_model("location_model.joblib")
    
    # 5. 시연 및 시각화 (Demo)
    print("\n⚾ AI Strategy Demo (2 Strikes, 0 Balls)")
    
    # 가상 상황
    demo_input = X.iloc[0:1].copy() # 데이터 형식 유지를 위해 첫 행 복사
    demo_input['balls'] = 0
    demo_input['strikes'] = 2
    demo_input['on_1b'] = 0
    
    # 5-1. 구종 추천
    pred_prob = type_trainer.model.predict_proba(demo_input)[0]
    best_idx = np.argmax(pred_prob)
    best_pitch = le_type.inverse_transform([best_idx])[0]
    
    print(f"   Recommended Pitch: {best_pitch} ({pred_prob[best_idx]*100:.1f}%)")
    
    # 5-2. 위치 추천
    # 구종 정보를 입력에 추가
    demo_input_loc = demo_input.copy()
    demo_input_loc['pitch_type_code'] = best_idx
    
    pred_loc = loc_trainer.model.predict(demo_input_loc)[0]
    print(f"   Target Location: X={pred_loc[0]:.2f}, Z={pred_loc[1]:.2f}")
    
    # 5-3. 시각화
    # 시각화 배경용: 해당 구종의 실제 투구 분포 필터링
    mask = (y_type == best_idx)
    pitch_actual_locs = y_locs[mask]
    
    visualize_pitch_location(best_pitch, pred_loc, pitch_actual_locs)

if __name__ == "__main__":
    main()