# run_phase3.py
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))

from src.data.loader import load_raw_data
from src.features.engineering import FeatureEngineer
from src.models.trainer import PitchRecommendationTrainer
import config.config as cfg

def main():
    print(">>> [Phase 3] Model Training & Evaluation...")

    # 1. 데이터 준비 (Data Preparation)
    df_raw = load_raw_data()
    if df_raw is None: return

    # 피처 엔지니어링 (Leakage-Free)
    engineer = FeatureEngineer()
    X, y, le = engineer.create_features(df_raw, is_training=True)
    
    # [Future Work] 여기에 RE288 관련 피처 추가 로직이 들어갈 수 있음
    # X = add_re288_features(X) 
    
    print(f"[INFO] Features prepared: {X.shape[1]} columns")
    print(f"      - Columns: {list(X.columns)}")

    # 2. 모델 학습 (Training)
    trainer = PitchRecommendationTrainer()
    results = trainer.train(X, y)
    
    # 3. 성능 평가 (Evaluation)
    print("-" * 40)
    print(f"🏆 Model Accuracy: {results['accuracy']:.4f}")
    print("-" * 40)
    print("[Classification Report]")
    print(results['report'])
    
    # 4. 시각화 (Visualization)
    print(">>> [Phase 3] Generating Evaluation Plots...")
    
    # 4-1. 피처 중요도 (어떤 상황이 중요한가?)
    trainer.plot_feature_importance()
    
    # 4-2. 혼동 행렬 (어떤 구종을 헷갈려하는가?)
    # 클래스 이름(구종) 복원
    class_names = le.classes_
    trainer.plot_confusion_matrix(results['y_test'], results['y_pred'], class_names)

    # 5. 모델 저장 (Save)
    # Label Encoder도 나중에 복원을 위해 함께 저장해야 하므로 trainer에 포함시키거나 별도 저장 필요
    # 여기서는 모델 파일만 저장 (실제 서비스 시엔 Label Encoder도 필요함)
    trainer.save_model()
    
    print(">>> Phase 3 Complete. Model is ready for recommendation!")

if __name__ == "__main__":
    main()