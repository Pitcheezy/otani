# src/recommendation/recommender.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import config.config as cfg

class PitchRecommender:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = cfg.MODEL_DIR / "pitch_model.joblib"
            
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run Phase 3 first.")
            
        # 저장된 모델과 메타데이터 로드
        loaded_data = joblib.load(model_path)
        self.model = loaded_data['model']
        self.feature_names = loaded_data['feature_names']
        
        # 모델 학습 시 사용된 클래스(구종) 이름
        # (Trainer에서 label encoder를 저장 안 했다면 모델의 classes_ 속성 사용)
        self.classes = self.model.classes_ 

    def recommend(self, input_data, top_k=3):
        """
        상황(input_data)을 받아 Top-K 구종을 추천합니다.
        input_data: dict or pd.DataFrame
        """
        # 1. 입력 데이터 포맷 맞추기
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
            
        # 학습 때 쓴 컬럼 순서대로 정렬 (누락된 건 0으로 채움)
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[self.feature_names]
        
        # 2. 확률 예측 (Predict Proba)
        # 예: [[0.1, 0.05, 0.6, ...]]
        probs = self.model.predict_proba(input_df)[0]
        
        # 3. 순위 매기기
        # 확률 높은 순으로 인덱스 정렬
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            pitch_name = str(idx) # LabelEncoder가 없어서 일단 ID로 출력 (나중에 매핑 필요)
            # 만약 classes_가 문자열이라면 그걸 사용
            if hasattr(self, 'classes'):
                pitch_name = str(self.classes[idx])
                
            recommendations.append({
                "rank": rank,
                "pitch_type": pitch_name,
                "probability": float(probs[idx])
            })
            
        return recommendations