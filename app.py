# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import platform

# 설정 모듈 임포트
import config.config as cfg
from src.recommendation.recommender import PitchRecommender

# --- [설정] ---
st.set_page_config(page_title="Pitch Recommender", layout="wide")

# 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

PITCH_MAP = {
    0: 'CU (Curveball)', 1: 'FC (Cutter)', 2: 'FF (4-Seam Fastball)',
    3: 'FS (Splitter)', 4: 'SI (Sinker)', 5: 'SL (Slider)', 6: 'ST (Sweeper)'
}
PITCH_MAP_REVERSE = {v: k for k, v in PITCH_MAP.items()}

# --- [함수] ---
@st.cache_resource
def load_models():
    """구종 모델과 위치 모델 로드"""
    try:
        # 1. 구종 모델
        path_rl = cfg.MODEL_DIR / "pitch_model_rl.joblib"
        path_base = cfg.MODEL_DIR / "pitch_model.joblib"
        model_path = path_rl if path_rl.exists() else path_base
        recommender = PitchRecommender(model_path=model_path)
        
        # 2. 위치 모델
        loc_path = cfg.MODEL_DIR / "location_model.joblib"
        loc_model = None
        if loc_path.exists():
            loc_model = joblib.load(loc_path)
            
        return recommender, loc_model
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None

def plot_dual_heatmap(pred_loc1, name1, pred_loc2, name2):
    """1순위(Red)와 2순위(Blue) 추천 위치를 좌우로 분할하여 시각화합니다."""
    # 1행 2열의 그래프 생성 (가로로 길게)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 공통 설정: 스트라이크 존 그리기
    def draw_zone(ax):
        # 스트라이크 존 사각형
        rect = patches.Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=2, edgecolor='black', facecolor='none', zorder=5)
        ax.add_patch(rect)
        
        # 내부 가이드라인
        for x in [-0.27, 0.27]: ax.plot([x, x], [1.5, 3.5], 'k--', alpha=0.1, zorder=1)
        for y in [2.16, 2.83]: ax.plot([-0.83, 0.83], [y, y], 'k--', alpha=0.1, zorder=1)
        
        ax.text(0, 0.5, "Catcher View", ha='center', color='gray')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5.0)
        ax.axis('off')

    # 가상 데이터 생성용 공분산 (탄착군 퍼짐 정도)
    cov = [[0.1, 0], [0, 0.1]]

    # --- [왼쪽 그래프] 1순위 (Red) ---
    ax1 = axes[0]
    draw_zone(ax1)
    
    x1, y1 = np.random.multivariate_normal(pred_loc1, cov, 1000).T
    sns.kdeplot(x=x1, y=y1, fill=True, cmap="Reds", alpha=0.7, levels=10, thresh=0.05, ax=ax1, zorder=2)
    ax1.scatter(pred_loc1[0], pred_loc1[1], color='red', s=200, marker='X', edgecolors='white', linewidth=2, zorder=10)
    ax1.set_title(f"🥇 1st Recommendation\n{name1}", fontsize=14, fontweight='bold', color='#D32F2F')

    # --- [오른쪽 그래프] 2순위 (Blue) ---
    ax2 = axes[1]
    draw_zone(ax2)
    
    x2, y2 = np.random.multivariate_normal(pred_loc2, cov, 1000).T
    sns.kdeplot(x=x2, y=y2, fill=True, cmap="Blues", alpha=0.7, levels=10, thresh=0.05, ax=ax2, zorder=2)
    ax2.scatter(pred_loc2[0], pred_loc2[1], color='blue', s=200, marker='o', edgecolors='white', linewidth=2, zorder=10)
    ax2.set_title(f"🥈 2nd Recommendation\n{name2}", fontsize=14, fontweight='bold', color='#1976D2')

    plt.tight_layout()
    return fig

def get_pitch_name_and_id(res):
    """결과 딕셔너리에서 이름과 ID를 추출"""
    name = str(res['pitch_type'])
    # 이름이 숫자면 변환
    if name.isdigit():
        p_id = int(name)
        p_name = PITCH_MAP.get(p_id, str(p_id))
    else:
        p_name = name
        # ID 찾기
        p_id = PITCH_MAP_REVERSE.get(name, 2) # 기본값 FF
        # 괄호 포함 이름 처리
        for k, v in PITCH_MAP.items():
            if v == name:
                p_id = k
                break
    return p_name, p_id

# --- [UI 구성] ---
def main():
    st.title("Pitcheezy")
    st.markdown("### 오타니 쇼헤이 투구 추천 시스템")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("1. 상황 (Context)")
        with st.container():
            c1, c2 = st.columns(2)
            balls = c1.number_input("Balls", 0, 3, 0)
            strikes = c2.number_input("Strikes", 0, 2, 2)
            outs = st.selectbox("아웃", [0, 1, 2])
            on_1b = st.checkbox("1루 주자")
            on_2b = st.checkbox("2루 주자")
            on_3b = st.checkbox("3루 주자")
            score_diff = 0
            inning = 5
            is_batter_lefty = 1
            
            st.info("💡 **Tip:** 2순위 추천 구종(파란색)과 위치도 함께 비교해보세요!")

            prev_pitch_name = st.selectbox("직전 구종", ["No_Pitch (초구)"] + list(PITCH_MAP.values()))
            if prev_pitch_name == "No_Pitch (초구)":
                prev_pitch_code = 2 
            else:
                prev_pitch_code = PITCH_MAP_REVERSE[prev_pitch_name]

    with col2:
        st.header("2. AI 전략 수립 (Strategy)")
        
        if st.button("전략 생성 (Generate Strategy)", type="primary"):
            recommender, loc_model = load_models()
            
            if recommender and loc_model:
                input_data = {
                    'inning': inning, 'balls': balls, 'strikes': strikes, 'outs_when_up': outs,
                    'score_diff': score_diff, 'on_1b': int(on_1b), 'on_2b': int(on_2b), 'on_3b': int(on_3b),
                    'is_batter_lefty': is_batter_lefty, 'pitcher_throws_left': 0,
                    'prev_pitch_type_code': prev_pitch_code
                }
                
                # 1) 구종 추천 (Top 3)
                results = recommender.recommend(input_data, top_k=3)
                
                # 1순위 데이터 준비
                res1 = results[0]
                name1, id1 = get_pitch_name_and_id(res1)
                
                # 2순위 데이터 준비
                res2 = results[1]
                name2, id2 = get_pitch_name_and_id(res2)
                
                # 2) 위치 예측을 위한 입력 데이터프레임 생성 함수
                def predict_location(p_id):
                    df = pd.DataFrame([input_data])
                    for col in recommender.feature_names:
                        if col not in df.columns: df[col] = 0
                    df = df[recommender.feature_names]
                    df['pitch_type_code'] = p_id
                    return loc_model.predict(df)[0]

                pred_loc1 = predict_location(id1)
                pred_loc2 = predict_location(id2)
                
                # --- 시각화 영역 ---
                c_res1, c_res2 = st.columns([1, 1])
                
                with c_res1:
                    st.subheader("🎯 추천 분석")
                    
                    # 1순위 카드
                    st.success(f"**1순위: {name1}**")
                    st.caption(f"확률: {res1['probability']*100:.1f}% | 목표: X={pred_loc1[0]:.2f}, Z={pred_loc1[1]:.2f}")
                    
                    st.write("") # 공백
                    
                    # 2순위 카드
                    st.info(f"**2순위: {name2}**")
                    st.caption(f"확률: {res2['probability']*100:.1f}% | 목표: X={pred_loc2[0]:.2f}, Z={pred_loc2[1]:.2f}")
                    
                    st.write("---")
                    st.write("**분석 코멘트:**")
                    if res1['probability'] - res2['probability'] < 0.1:
                        st.write("👉 두 구종의 확률 차이가 크지 않습니다. 타자의 반응에 따라 **2순위 구종**을 섞어 던지는 것이 효과적일 수 있습니다.")
                    else:
                        st.write("👉 **1순위 구종**이 압도적으로 추천됩니다. 확실한 결정구로 사용하는 것을 권장합니다.")

                with c_res2:
                    st.subheader("📍 멀티 타겟 로케이션")
                    # [수정] 두 개의 위치를 동시에 그리는 함수 호출
                    fig = plot_dual_heatmap(pred_loc1, name1, pred_loc2, name2)
                    st.pyplot(fig)

if __name__ == "__main__":
    main()