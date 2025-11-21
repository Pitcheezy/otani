# src/models/location_trainer.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import config.config as cfg

class PitchLocationTrainer:
    def __init__(self):
        # X, Z 좌표 두 개를 동시에 예측해야 하므로 MultiOutputRegressor 사용
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=cfg.RANDOM_SEED,
                n_jobs=-1
            )
        )
        
    def train(self, X, y_locations):
        """
        X: 상황 정보 + 추천된 구종 ID (중요)
        y_locations: 실제 좌표 (plate_x, plate_z)
        """
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_locations, test_size=0.2, random_state=cfg.RANDOM_SEED
        )
        
        print("[Location] Training Location Model...")
        self.model.fit(X_train, y_train)
        
        # 평가
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"[Location] Model RMSE: {rmse:.4f} ft")
        
        return self.model

    def save_model(self, filename="location_model.joblib"):
        save_path = cfg.MODEL_DIR / filename
        joblib.dump(self.model, save_path)
        print(f"[INFO] Location Model saved to {save_path}")