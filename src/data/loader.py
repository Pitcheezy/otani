# src/data/loader.py
import sys
from pathlib import Path
import pandas as pd
import pybaseball
from pybaseball import statcast_pitcher

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import config.config as cfg

def load_raw_data(force_update=False):
    """
    오타니 쇼헤이의 투구 데이터를 로드합니다.
    파일이 존재하면 로컬에서 읽고, 없거나 force_update가 True면 새로 다운로드합니다.
    """
    file_path = cfg.RAW_DATA_DIR / "pitch_data.csv"
    
    if file_path.exists() and not force_update:
        print(f"[INFO] Loading existing data from {file_path}")
        df = pd.read_csv(file_path)
        return df
    
    print(f"[INFO] Downloading data for Player ID: {cfg.DATA_COLLECTION['player_id']}")
    print(f"[INFO] Period: {cfg.DATA_COLLECTION['start_date']} ~ {cfg.DATA_COLLECTION['end_date']}")
    
    # pybaseball을 이용해 데이터 수집
    try:
        df = statcast_pitcher(
            start_dt=cfg.DATA_COLLECTION['start_date'], 
            end_dt=cfg.DATA_COLLECTION['end_date'], 
            player_id=cfg.DATA_COLLECTION['player_id']
        )
        
        if df is not None and not df.empty:
            df.to_csv(file_path, index=False)
            print(f"[INFO] Data saved to {file_path}")
            print(f"[INFO] Total rows: {len(df)}")
            return df
        else:
            print("[ERROR] No data found for the given period.")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to download data: {e}")
        return None