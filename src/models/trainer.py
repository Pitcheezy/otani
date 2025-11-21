# src/models/trainer.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

import config.config as cfg

class PitchRecommendationTrainer:
    def __init__(self, model_config=None):
        self.config = model_config if model_config else cfg.MODEL_CONFIG
        
        # Mac M4 자원 활용 정보 출력
        n_cores = os.cpu_count()
        processor = platform.processor()
        print(f"[SYSTEM] Detected Processor: {platform.machine()} ({processor})")
        print(f"[SYSTEM] Available CPU Cores: {n_cores}")
        print(f"[SYSTEM] Using all {n_cores} cores for parallel processing (n_jobs=-1)")

        self.model = RandomForestClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 10),
            min_samples_split=self.config.get('min_samples_split', 5),
            random_state=cfg.RANDOM_SEED,
            # [핵심] M4의 모든 코어를 사용하여 병렬 학습 수행
            n_jobs=-1,
            class_weight='balanced'  # [추가] 소수 구종(커브 등) 학습 가중치 부여
        )
        self.label_encoder = None
        self.feature_names = None

    def train(self, X, y, test_size=0.2):
        """
        데이터를 Train/Test로 나누고 모델을 학습시킵니다.
        """
        self.feature_names = list(X.columns)
        
        # 데이터 분할 (Split Data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=cfg.RANDOM_SEED, stratify=y
        )
        
        print(f"[INFO] Training Set: {X_train.shape}, Test Set: {X_test.shape}")
        print("[INFO] Starting training on M4 Neural Engine/CPU...")
        
        # 학습 (Fit) - 여기서 모든 코어가 100% 가동됩니다.
        self.model.fit(X_train, y_train)
        print("[INFO] Training Complete!")
        
        # 예측 (Predict)
        y_pred = self.model.predict(X_test)
        
        # 평가 (Evaluate)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "report": report,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }

    def plot_feature_importance(self):
        """피처 중요도를 시각화합니다."""
        if not self.model:
            return
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Mac 환경에서 한글 폰트 깨짐 방지 (AppleGothic)
        if platform.system() == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances (Random Forest)")
        sns.barplot(x=importances[indices], y=np.array(self.feature_names)[indices], palette="viridis")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """혼동 행렬을 시각화합니다."""
        # Mac 환경 폰트 설정
        if platform.system() == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'

        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self, filename="pitch_model.joblib"):
        """학습된 모델과 메타데이터를 저장합니다."""
        save_path = cfg.MODEL_DIR / filename
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
        }
        joblib.dump(model_data, save_path)
        print(f"[INFO] Model saved to {save_path}")

        