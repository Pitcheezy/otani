# src/features/engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

# --- [수정] config 모듈 임포트 추가 ---
# 프로젝트 루트 경로를 sys.path에 추가하여 config를 찾을 수 있게 함
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config.config as cfg
# ------------------------------------

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        
        # [기본 피처] 상황 정보
        self.feature_columns = [
            'inning', 'balls', 'strikes', 'outs_when_up',
            'score_diff', 'on_1b', 'on_2b', 'on_3b',
            'is_batter_lefty', 'pitcher_throws_left',
            'prev_pitch_type_code'
        ]
        self.target_column = 'pitch_type'
        
        # [New] RE288 테이블 로드
        self.re_map = {}
        self._load_re288()

    def _load_re288(self):
        """CSV에서 RE288 테이블을 메모리로 로드"""
        # cfg가 정의되어 있어야 이 부분이 작동합니다.
        re_path = cfg.PROCESSED_DATA_DIR / "re288_table.csv"
        if re_path.exists():
            try:
                df_re = pd.read_csv(re_path)
                # 딕셔너리로 변환하여 검색 속도 향상 {state_key: re_value}
                self.re_map = dict(zip(df_re['state'], df_re['re_value']))
                print(f"[Info] RE288 Table loaded ({len(self.re_map)} states)")
            except Exception as e:
                print(f"[Warning] Failed to load RE288 table: {e}")
        else:
            print("[Warning] RE288 table not found. Using default values.")

    def get_re_value(self, row):
        """행(Row) 데이터에서 State Key를 추출하고 RE 값을 반환"""
        b = int(row.get('balls', 0))
        s = int(row.get('strikes', 0))
        o = int(row.get('outs_when_up', 0))
        r1 = 1 if pd.notnull(row.get('on_1b')) and row.get('on_1b') != 0 else 0
        r2 = 1 if pd.notnull(row.get('on_2b')) and row.get('on_2b') != 0 else 0
        r3 = 1 if pd.notnull(row.get('on_3b')) and row.get('on_3b') != 0 else 0
        
        # RE288 키 생성
        key = f"{b}-{s}-{o}-{r1}-{r2}-{r3}"
        return self.re_map.get(key, 0.5) # 없으면 기본값 0.5

    def preprocess(self, df):
        df = df.copy()
        df = df.dropna(subset=[self.target_column])
        
        fill_zero_cols = ['balls', 'strikes', 'outs_when_up', 'on_3b', 'on_2b', 'on_1b']
        for col in fill_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def calculate_reward(self, row):
        """
        RE288 기반 정교한 보상 계산
        Reward = RE(투구 전) - RE(투구 후) + 이벤트 보너스
        """
        pre_re = self.get_re_value(row)
        
        # 1. MLB 데이터에 이미 계산된 값이 있으면 최우선 사용
        if 'delta_run_exp' in row and pd.notnull(row['delta_run_exp']):
            return -row['delta_run_exp'] # 투수 입장이므로 부호 반대
            
        if 'delta_pitcher_run_exp' in row and pd.notnull(row['delta_pitcher_run_exp']):
             return row['delta_pitcher_run_exp']

        # 2. 없을 경우 휴리스틱 계산
        desc = str(row.get('description', '')).lower()
        event = str(row.get('events', '')).lower()
        
        risk_weight = 1.0 + pre_re
        
        reward = 0.0
        if 'strike' in desc:
            reward += 0.05 * risk_weight
        elif 'ball' in desc:
            reward -= 0.05
        elif 'hit' in desc or 'single' in event or 'double' in event or 'home_run' in event:
            reward -= 1.0 
        elif 'out' in event:
            reward += 0.3 * risk_weight
            
        return reward

    def create_features(self, df, is_training=True, use_rl_filter=False):
        df = self.preprocess(df)
        
        # 시퀀스 정렬
        if 'game_pk' in df.columns and 'at_bat_number' in df.columns and 'pitch_number' in df.columns:
            df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        
        # 기본 파생 변수 생성
        if 'fld_score' in df.columns and 'bat_score' in df.columns:
            df['score_diff'] = df['fld_score'] - df['bat_score']
        else:
            df['score_diff'] = 0
            
        for col in ['on_1b', 'on_2b', 'on_3b']:
            df[col] = df[col].apply(lambda x: 1 if pd.notnull(x) and x != 0 else 0)

        df['is_batter_lefty'] = (df['stand'] == 'L').astype(int) if 'stand' in df.columns else 0
        df['pitcher_throws_left'] = (df['p_throws'] == 'L').astype(int) if 'p_throws' in df.columns else 0

        # 피칭 디자인 (이전 구종)
        df['prev_pitch_type'] = df.groupby(['game_pk', 'at_bat_number'])[self.target_column].shift(1)
        df['prev_pitch_type'] = df['prev_pitch_type'].fillna('No_Pitch')
        
        le_prev = LabelEncoder()
        df['prev_pitch_type_code'] = le_prev.fit_transform(df['prev_pitch_type'].astype(str))
        
        if is_training:
            self.label_encoders['prev_pitch_type'] = le_prev
        
        # --- [RL 핵심] 데이터 필터링 (Behavioral Cloning) ---
        if is_training and use_rl_filter:
            # 1. 보상 계산
            df['reward'] = df.apply(self.calculate_reward, axis=1)
            
            # 2. 좋은 투구만 남기기 (상위 50%)
            threshold = df['reward'].quantile(0.5) 
            df_filtered = df[df['reward'] > threshold].copy()
            
            print(f"[RL] Filtering Data: {len(df)} -> {len(df_filtered)} (Threshold: {threshold:.4f})")
            df = df_filtered

        X = df[self.feature_columns]
        
        # 위치 예측을 위해 좌표 데이터도 반환 (학습 시)
        locations = None
        if is_training and 'plate_x' in df.columns and 'plate_z' in df.columns:
            locations = df[['plate_x', 'plate_z']]

        if is_training:
            y = df[self.target_column]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders['pitch_type'] = le
            
            return X, y_encoded, le, locations
        else:
            return X