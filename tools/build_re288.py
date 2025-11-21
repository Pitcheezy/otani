# tools/build_re288.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pybaseball import statcast
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import config.config as cfg

def fetch_league_data():
    """RE288 계산을 위한 리그 데이터 수집 (약 1개월 치 권장)"""
    print(">>> [Data Mining] Fetching MLB League Data for RE288...")
    # 데이터가 충분해야 정확한 RE가 나옵니다. 2024년 5월 한 달 치 데이터를 예시로 사용합니다.
    # M4 Mac mini라면 금방 처리할 겁니다.
    df = statcast(start_dt='2024-05-01', end_dt='2024-05-31')
    return df

def calculate_re288(df):
    """
    투구별 기대 득점(Run Expectancy) 매트릭스 생성
    State: (Balls, Strikes, Outs, 1B, 2B, 3B) -> 288가지
    """
    print(">>> [Processing] Calculating Run Expectancy Matrix...")
    
    # 1. 필요한 컬럼 선택 및 전처리
    cols = ['game_pk', 'inning', 'inning_topbot', 'batter', 'pitcher', 
            'events', 'description', 'balls', 'strikes', 'outs_when_up', 
            'on_1b', 'on_2b', 'on_3b', 'des', 'home_score', 'away_score']
    
    df = df[cols].copy().sort_values(['game_pk', 'inning', 'inning_topbot', 'outs_when_up', 'batter'])
    
    # 2. 이닝별 잔여 득점(Runs Rest of Inning) 계산
    # 현재 점수 상태 확인
    df['runs_now'] = df['home_score'] + df['away_score']
    
    # 이닝의 마지막(최종) 점수를 각 행에 매핑
    # (같은 게임, 같은 이닝, 같은 공격팀 그룹의 'runs_now' 최대값)
    # 주의: 대량 데이터 처리 시 groupby 속도가 느릴 수 있으므로 최적화 필요
    # 여기서는 직관적인 방법 사용
    
    # 이닝 종료 시점의 점수 구하기
    # 그룹 키: 게임ID + 이닝 + 초/말
    df['half_inning_id'] = df['game_pk'].astype(str) + "_" + df['inning'].astype(str) + "_" + df['inning_topbot']
    
    # 각 하프 이닝의 최종 점수 (Max Runs)
    inning_end_runs = df.groupby('half_inning_id')['runs_now'].transform('max')
    
    # 잔여 득점 = (이닝 최종 점수) - (현재 점수)
    df['runs_roi'] = inning_end_runs - df['runs_now']
    
    # 3. State 정의 (288 States)
    # 주자 유무 (0 or 1)
    df['on_1b'] = df['on_1b'].notna().astype(int)
    df['on_2b'] = df['on_2b'].notna().astype(int)
    df['on_3b'] = df['on_3b'].notna().astype(int)
    
    # 상태 키 생성: "B-S-O-1-2-3"
    # 예: "2-1-0-1-0-1" (2볼 1스트 0아웃, 1,3루)
    df['state_key'] = (
        df['balls'].astype(str) + "-" +
        df['strikes'].astype(str) + "-" +
        df['outs_when_up'].astype(str) + "-" +
        df['on_1b'].astype(str) + "-" +
        df['on_2b'].astype(str) + "-" +
        df['on_3b'].astype(str)
    )
    
    # 4. State별 평균 잔여 득점 계산 (이게 바로 RE 값)
    re_table = df.groupby('state_key')['runs_roi'].mean().reset_index()
    re_table.columns = ['state', 're_value']
    
    # 카운트별 데이터 수(Count)도 확인 (신뢰도 체크용)
    re_counts = df['state_key'].value_counts().reset_index()
    re_counts.columns = ['state', 'sample_size']
    
    final_re = pd.merge(re_table, re_counts, on='state')
    
    return final_re

def main():
    # 1. 데이터 수집
    try:
        df = fetch_league_data()
    except Exception as e:
        print(f"[Error] 데이터 수집 실패: {e}")
        return

    # 2. RE288 계산
    re288_df = calculate_re288(df)
    
    # 3. 저장
    save_path = cfg.PROCESSED_DATA_DIR / "re288_table.csv"
    re288_df.to_csv(save_path, index=False)
    print(f"✅ RE288 Table Saved to {save_path}")
    
    # 4. 시각화 (검증용)
    # 볼카운트(Ball-Strike)에 따른 평균 RE 변화 (주자 없을 때, 0아웃 기준)
    # State 포맷: B-S-O-1-2-3
    print(">>> [Visualization] Generating RE288 Heatmap...")
    
    # 주자 없고, 0아웃인 상황만 필터링
    # 패턴: "B-S-0-0-0-0"
    heatmap_data = np.zeros((3, 4)) # 3 Strikes x 4 Balls
    
    for b in range(4):
        for s in range(3):
            key = f"{b}-{s}-0-0-0-0"
            row = re288_df[re288_df['state'] == key]
            if not row.empty:
                heatmap_data[s, b] = row['re_value'].values[0]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=['0B', '1B', '2B', '3B'],
                yticklabels=['0S', '1S', '2S'])
    plt.title("RE288 Map (No Runners, 0 Out)\nRun Expectancy by Count")
    plt.xlabel("Balls")
    plt.ylabel("Strikes")
    plt.show()

if __name__ == "__main__":
    main()