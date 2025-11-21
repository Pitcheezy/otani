# run_phase4.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parent))

from src.data.loader import load_raw_data
from src.features.engineering import FeatureEngineer
from src.recommendation.recommender import PitchRecommender
import config.config as cfg

def calculate_top_k_accuracy(model, X, y, k=3):
    """Top-K 정확도를 계산합니다."""
    # 각 클래스별 확률 예측
    probs = model.predict_proba(X)
    # 확률 상위 k개의 인덱스 추출
    best_n = np.argsort(probs, axis=1)[:, -k:]
    
    # 정답(y)이 상위 k개 안에 있는지 확인
    success = 0
    for i in range(len(y)):
        if y[i] in best_n[i]:
            success += 1
    return success / len(y)

def main():
    print(">>> [Phase 4] Recommendation Engine Test...")

    # 1. 데이터 및 모델 준비
    df_raw = load_raw_data()
    if df_raw is None: return

    engineer = FeatureEngineer()
    X, y, le = engineer.create_features(df_raw, is_training=True)
    
    # 테스트 데이터 분리 (Phase 3와 동일하게)
    # (엄밀한 검증을 위해선 Trainer에서 저장한 test set을 써야 하지만, 약식으로 뒷부분 20% 사용)
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y[split_idx:]

    try:
        recommender = PitchRecommender()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 2. Top-3 정확도 측정 (여기가 중요!)
    print("-" * 40)
    print("Evaluating Performance...")
    top1_acc = accuracy_score(y_test, recommender.model.predict(X_test))
    top3_acc = calculate_top_k_accuracy(recommender.model, X_test, y_test, k=3)
    
    print(f"   Top-1 Accuracy: {top1_acc:.4f} (정확히 맞힘)")
    print(f"   Top-3 Accuracy: {top3_acc:.4f} (3개 안에 정답 있음)")
    print("-" * 40)

    # 3. 실제 추천 시연 (Demo)
    print("Recommendation Demo")
    
    # 상황: 2스트라이크 0볼, 주자 없음, 좌타자 (직전 공은 'No_Pitch'라고 가정)
    demo_situation = {
        'inning': 5,
        'balls': 0,
        'strikes': 2,
        'outs_when_up': 0,
        'score_diff': 0,
        'on_1b': 0, 'on_2b': 0, 'on_3b': 0,
        'is_batter_lefty': 1,
        'pitcher_throws_left': 0,
        'prev_pitch_type_code': 0 # 임의값
    }
    
    print(f"   Situation: 2 Strikes, 0 Balls, Lefty Batter")
    results = recommender.recommend(demo_situation, top_k=3)
    
    print("   AI Suggestion:")
    for res in results:
        # 원래 구종 이름 복원 (le.inverse_transform 사용)
        # 여기서는 편의상 ID와 함께 출력
        try:
            pitch_name = le.inverse_transform([int(res['pitch_type'])])[0]
        except:
            pitch_name = res['pitch_type']
            
        print(f"   [{res['rank']}위] {pitch_name} ({res['probability']*100:.1f}%)")

    print(">>> Phase 4 Complete.")

if __name__ == "__main__":
    main()