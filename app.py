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
import plotly.express as px
import plotly.graph_objects as go

# 설정 모듈 임포트
import config.config as cfg
from src.recommendation.recommender import PitchRecommender
from src.data.video_data import load_video_analysis, find_similar_pitches
from src.utils.video_loader import display_video_safe

# --- [페이지 설정] ---
st.set_page_config(
    page_title="Pitcheezy - MLB 투구 분석 대시보드",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [커스텀 CSS 스타일] ---
st.markdown("""
<style>
    /* 전체 테마 색상 */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #FF6F00;
        --success-color: #43A047;
        --warning-color: #FB8C00;
        --danger-color: #E53935;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 히어로 영역 스타일 */
    .hero-section {
        background: linear-gradient(135deg, #1E88E5 0%, #1976D2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* KPI 카드 스타일 */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        transition: transform 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* 섹션 카드 스타일 */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* 배지 스타일 */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-primary {
        background-color: #E3F2FD;
        color: #1976D2;
    }
    
    .badge-success {
        background-color: #E8F5E9;
        color: #43A047;
    }
    
    .badge-warning {
        background-color: #FFF3E0;
        color: #FB8C00;
    }
    
    /* 비디오 카드 스타일 */
    .video-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .video-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 2px 8px rgba(30, 136, 229, 0.2);
    }
    
    .video-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    /* 필터 패널 스타일 */
    .filter-panel {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        height: 100%;
    }
    
    /* 메트릭 카드 개선 */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

# 구종별 색상 매핑
PITCH_COLORS = {
    'FF': '#E53935',  # Red
    'ST': '#1E88E5',  # Blue
    'CU': '#43A047',  # Green
    'SL': '#FB8C00',  # Orange
    'FS': '#8E24AA',  # Purple
    'FC': '#00ACC1',  # Cyan
    'SI': '#FDD835',  # Yellow
}

# --- [함수] ---
@st.cache_resource
def load_models():
    """구종 모델과 위치 모델 로드"""
    try:
        path_rl = cfg.MODEL_DIR / "pitch_model_rl.joblib"
        path_base = cfg.MODEL_DIR / "pitch_model.joblib"
        model_path = path_rl if path_rl.exists() else path_base
        recommender = PitchRecommender(model_path=model_path)
        
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    def draw_zone(ax):
        rect = patches.Rectangle((-0.83, 1.5), 1.66, 2.0, linewidth=2, edgecolor='black', facecolor='none', zorder=5)
        ax.add_patch(rect)
        for x in [-0.27, 0.27]: ax.plot([x, x], [1.5, 3.5], 'k--', alpha=0.1, zorder=1)
        for y in [2.16, 2.83]: ax.plot([-0.83, 0.83], [y, y], 'k--', alpha=0.1, zorder=1)
        ax.text(0, 0.5, "Catcher View", ha='center', color='gray')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5.0)
        ax.axis('off')

    cov = [[0.1, 0], [0, 0.1]]

    ax1 = axes[0]
    draw_zone(ax1)
    x1, y1 = np.random.multivariate_normal(pred_loc1, cov, 1000).T
    sns.kdeplot(x=x1, y=y1, fill=True, cmap="Reds", alpha=0.7, levels=10, thresh=0.05, ax=ax1, zorder=2)
    ax1.scatter(pred_loc1[0], pred_loc1[1], color='red', s=200, marker='X', edgecolors='white', linewidth=2, zorder=10)
    ax1.set_title(f"🥇 1st Recommendation\n{name1}", fontsize=14, fontweight='bold', color='#D32F2F')

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
    if name.isdigit():
        p_id = int(name)
        p_name = PITCH_MAP.get(p_id, str(p_id))
    else:
        p_name = name
        p_id = PITCH_MAP_REVERSE.get(name, 2)
        for k, v in PITCH_MAP.items():
            if v == name:
                p_id = k
                break
    return p_name, p_id

def render_kpi_card(label, value, icon="📊", delta=None):
    """KPI 카드 렌더링"""
    delta_html = f'<span style="color: #43A047; font-size: 0.9rem;">{delta}</span>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span class="kpi-label">{label}</span>
        </div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def show_hero_section(df: pd.DataFrame):
    """히어로 영역 표시"""
    if df is None or len(df) == 0:
        return
    
    total_pitches = len(df)
    avg_angle = df["calculated_release_angle"].mean() if "calculated_release_angle" in df.columns else 0
    avg_detection = df["detection_rate"].mean() if "detection_rate" in df.columns else 0
    pitch_types = df["pitch_type_extracted"].nunique() if "pitch_type_extracted" in df.columns else 0
    
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">⚾ Pitcheezy</div>
        <div class="hero-subtitle">Shohei Ohtani 투구 분석 대시보드</div>
        <div style="margin-top: 1.5rem; display: flex; gap: 2rem; flex-wrap: wrap;">
            <div><strong>{total_pitches:,}</strong> 개의 투구 분석</div>
            <div><strong>{pitch_types}</strong> 가지 구종</div>
            <div><strong>{avg_detection:.1f}%</strong> 평균 탐지율</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_analysis_home(df: pd.DataFrame):
    """영상/팔각도 요약 대시보드"""
    show_hero_section(df)
    
    st.markdown("## 📊 주요 지표")
    
    # KPI 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi_card("총 분석 투구", f"{len(df):,}개", "⚾")
    
    with col2:
        if "calculated_release_angle" in df.columns:
            avg_angle = df["calculated_release_angle"].mean()
            render_kpi_card("평균 릴리스 각도", f"{avg_angle:.1f}°", "📐")
    
    with col3:
        if "detection_rate" in df.columns:
            avg_detection = df["detection_rate"].mean()
            render_kpi_card("평균 탐지율", f"{avg_detection:.1f}%", "🎯")
    
    with col4:
        if "pitch_type_extracted" in df.columns:
            pitch_types = df["pitch_type_extracted"].nunique()
            render_kpi_card("구종 종류", f"{pitch_types}개", "🎨")
    
    st.markdown("---")
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("### 📈 릴리스 각도 분포")
            if "calculated_release_angle" in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df["calculated_release_angle"].dropna(), bins=30, edgecolor="black", alpha=0.7, color="#1E88E5")
                ax.set_xlabel("릴리스 각도 (°)", fontsize=12)
                ax.set_ylabel("빈도", fontsize=12)
                ax.set_title("릴리스 각도 분포", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    with col2:
        with st.container(border=True):
            st.markdown("### 🎯 구종별 분포")
            if "pitch_type_extracted" in df.columns:
                pitch_counts = df["pitch_type_extracted"].value_counts()
                colors = [PITCH_COLORS.get(pt[:2], '#95a5a6') for pt in pitch_counts.index]
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(range(len(pitch_counts)), pitch_counts.values, color=colors, edgecolor='white', linewidth=2)
                ax.set_xlabel("구종", fontsize=12)
                ax.set_ylabel("개수", fontsize=12)
                ax.set_title("구종별 투구 수", fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(pitch_counts)))
                ax.set_xticklabels(pitch_counts.index, rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.3)
                st.pyplot(fig)

def show_overall_stats_page(df: pd.DataFrame):
    """전체 통계 페이지"""
    show_hero_section(df)
    
    st.markdown("## 📈 전체 통계 분석")
    
    stats_cols = [
        "calculated_release_angle", "calculated_avg_angle", "release_frame",
        "max_wrist_velocity", "detection_rate", "angle_range",
    ]
    stats_cols = [c for c in stats_cols if c in df.columns]
    
    if not stats_cols:
        st.warning("통계에 사용할 컬럼이 없습니다.")
        return
    
    tab1, tab2 = st.tabs(["📊 주요 지표", "🔗 상관관계"])
    
    with tab1:
        with st.container(border=True):
            st.markdown("### 주요 지표 통계")
            stats_df = df[stats_cols].describe()
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True, height=400)
    
    with tab2:
        if len(stats_cols) >= 2:
            with st.container(border=True):
                st.markdown("### 변수 간 상관관계")
                corr_matrix = df[stats_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
                ax.set_title("상관관계 히트맵", fontsize=14, fontweight='bold', pad=20)
                st.pyplot(fig)

def show_pitch_type_analysis_page(df: pd.DataFrame):
    """구종별 분석 페이지"""
    show_hero_section(df)
    
    if "pitch_type_extracted" not in df.columns:
        st.warning("구종 정보가 없어 구종별 분석을 할 수 없습니다.")
        return
    
    st.markdown("## 🎯 구종별 분석")
    
    pitch_types = sorted(df["pitch_type_extracted"].dropna().unique())
    selected_types = st.multiselect("구종 선택", pitch_types, default=pitch_types, key="pitch_type_select")
    
    if not selected_types:
        st.warning("구종을 선택해주세요.")
        return
    
    filtered_df = df[df["pitch_type_extracted"].isin(selected_types)]
    
    tab1, tab2 = st.tabs(["📊 통계", "📈 시각화"])
    
    with tab1:
        with st.container(border=True):
            st.markdown("### 구종별 통계")
            pitch_stats = (
                filtered_df.groupby("pitch_type_extracted")
                .agg({
                    "calculated_release_angle": ["mean", "std", "count"],
                    "calculated_avg_angle": "mean",
                    "max_wrist_velocity": "mean",
                    "detection_rate": "mean",
                })
                .round(2)
            )
            st.dataframe(pitch_stats, use_container_width=True)
    
    with tab2:
        with st.container(border=True):
            st.markdown("### 구종별 릴리스 각도 비교")
            fig, ax = plt.subplots(figsize=(12, 6))
            data_to_plot = [
                filtered_df[filtered_df["pitch_type_extracted"] == pt]["calculated_release_angle"].dropna()
                for pt in selected_types
            ]
            bp = ax.boxplot(data_to_plot, labels=selected_types, patch_artist=True)
            for patch, pt in zip(bp['boxes'], selected_types):
                patch.set_facecolor(PITCH_COLORS.get(pt[:2], '#95a5a6'))
                patch.set_alpha(0.7)
            ax.set_xlabel("구종", fontsize=12)
            ax.set_ylabel("릴리스 각도 (°)", fontsize=12)
            ax.set_title("구종별 릴리스 각도 분포", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

def show_pitch_phase_analysis_page(df: pd.DataFrame):
    """투구 단계 분석 페이지"""
    show_hero_section(df)
    
    st.markdown("## 🔄 투구 단계별 분석")
    
    phases = {
        "준비 단계": "setup_angle",
        "코킹 단계": "cocking_avg",
        "가속 단계": "accel_avg",
        "감속 단계": "decel_avg",
        "팔로스루 단계": "follow_avg",
    }
    
    phase_data = []
    phase_names = []
    for phase_name, col in phases.items():
        if col not in df.columns:
            continue
        angles = df[col].dropna()
        if len(angles) > 0:
            phase_data.append(angles)
            phase_names.append(phase_name)
    
    if len(phase_data) > 0:
        tab1, tab2 = st.tabs(["📈 시각화", "📊 통계"])
        
        with tab1:
            with st.container(border=True):
                st.markdown("### 투구 단계별 각도 분포")
                fig, ax = plt.subplots(figsize=(12, 6))
                bp = ax.boxplot(phase_data, labels=phase_names, patch_artist=True)
                colors = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_xlabel("투구 단계", fontsize=12)
                ax.set_ylabel("각도 (°)", fontsize=12)
                ax.set_title("투구 단계별 각도 분포", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with tab2:
            with st.container(border=True):
                st.markdown("### 단계별 통계")
                phase_stats = []
                for phase_name, col in phases.items():
                    if col not in df.columns:
                        continue
                    angles = df[col].dropna()
                    if len(angles) > 0:
                        phase_stats.append({
                            "단계": phase_name,
                            "평균 각도": angles.mean(),
                            "중앙값": angles.median(),
                            "표준편차": angles.std(),
                            "샘플 수": len(angles),
                        })
                if phase_stats:
                    phase_stats_df = pd.DataFrame(phase_stats)
                    format_dict = {col: "{:.2f}" for col in phase_stats_df.columns if phase_stats_df[col].dtype in ['float64', 'float32', 'int64', 'int32']}
                    if format_dict:
                        st.dataframe(phase_stats_df.style.format(format_dict), use_container_width=True)
                    else:
                        st.dataframe(phase_stats_df, use_container_width=True)

def show_temporal_analysis_page(df: pd.DataFrame):
    """시간 경향성 분석 페이지"""
    show_hero_section(df)
    
    st.markdown("## 📅 시간 경향성 분석")
    
    if "date" not in df.columns:
        st.warning("날짜 정보가 없어 시간 경향성 분석을 할 수 없습니다.")
        return
    
    df_with_date = df[df["date"].notna()].copy()
    if len(df_with_date) == 0:
        st.warning("날짜 정보가 없어 시간 경향성 분석을 할 수 없습니다.")
        return
    
    df_with_date = df_with_date.sort_values("date")
    
    with st.container(border=True):
        st.markdown("### 시간에 따른 릴리스 각도 추이")
        if "calculated_release_angle" in df_with_date.columns:
            daily_avg = df_with_date.groupby("date")["calculated_release_angle"].mean().reset_index()
            fig = px.line(
                daily_avg,
                x="date",
                y="calculated_release_angle",
                title="날짜별 평균 릴리스 각도",
                markers=True,
                line_shape='spline'
            )
            fig.update_layout(
                xaxis_title="날짜",
                yaxis_title="평균 릴리스 각도 (°)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            fig.update_traces(line_color='#1E88E5', line_width=3)
            st.plotly_chart(fig, use_container_width=True)

def show_video_explorer_page(df: pd.DataFrame):
    """동영상 탐색 페이지 - 3단 레이아웃"""
    show_hero_section(df)
    
    st.markdown("## 🎬 동영상 탐색")
    
    if "calculated_release_angle" not in df.columns:
        st.warning("릴리스 각도 정보가 없어 동영상 탐색을 할 수 없습니다.")
        return
    
    # 3단 레이아웃: 필터 | 결과 | 상세
    col_filter, col_results, col_detail = st.columns([1, 2, 2])
    
    with col_filter:
        st.markdown("### 🔍 필터")
        with st.container(border=True):
            if "pitch_type_extracted" in df.columns:
                pitch_types = ["전체"] + sorted(df["pitch_type_extracted"].dropna().unique().tolist())
                selected_pitch_type = st.selectbox("구종", pitch_types, key="video_pitch_filter")
            else:
                selected_pitch_type = "전체"
            
            min_angle = float(df["calculated_release_angle"].min())
            max_angle = float(df["calculated_release_angle"].max())
            min_sel, max_sel = st.slider(
                "릴리스 각도 범위",
                min_value=min_angle,
                max_value=max_angle,
                value=(min_angle, max_angle),
                key="video_angle_filter"
            )
            
            st.markdown("---")
            st.caption(f"**총 {len(df)}개** 투구 중")
    
    # 필터링
    filtered_df = df[
        (df["calculated_release_angle"] >= min_sel) &
        (df["calculated_release_angle"] <= max_sel)
    ]
    
    if "pitch_type_extracted" in df.columns and selected_pitch_type != "전체":
        filtered_df = filtered_df[filtered_df["pitch_type_extracted"] == selected_pitch_type]
    
    with col_results:
        st.markdown(f"### 📋 검색 결과 ({len(filtered_df)}개)")
        
        if len(filtered_df) == 0:
            st.info("조건에 맞는 투구가 없습니다.")
        else:
            # 결과 리스트를 카드 형태로 표시
            display_df = filtered_df.head(50).copy()  # 최대 50개만 표시
            
            selected_idx = st.selectbox(
                "투구 선택",
                range(len(display_df)),
                format_func=lambda x: f"#{x+1} | {display_df.iloc[x].get('pitch_type_extracted', 'N/A')} | 각도: {display_df.iloc[x].get('calculated_release_angle', 0):.1f}°",
                key="video_select"
            )
            
            selected_pitch = display_df.iloc[selected_idx]
            
            # 선택된 투구 정보 카드
            with st.container(border=True):
                pitch_type = selected_pitch.get("pitch_type_extracted", "N/A")
                pitch_color = PITCH_COLORS.get(pitch_type[:2] if pitch_type != "N/A" else "", "#95a5a6")
                
                st.markdown(f"""
                <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem;">
                    <span class="badge" style="background-color: {pitch_color}20; color: {pitch_color}; border: 1px solid {pitch_color};">
                        {pitch_type}
                    </span>
                    <span style="font-weight: 600;">각도: {selected_pitch.get('calculated_release_angle', 0):.1f}°</span>
                    <span style="color: #666;">탐지율: {selected_pitch.get('detection_rate', 0):.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                # 기본 정보 테이블
                info_data = {
                    "항목": ["게임 ID", "타석", "투구", "구종", "릴리스 각도", "평균 각도", "탐지율"],
                    "값": [
                        selected_pitch.get("game_pk", "N/A"),
                        selected_pitch.get("at_bat_number", "N/A"),
                        selected_pitch.get("pitch_number", "N/A"),
                        pitch_type,
                        f"{selected_pitch.get('calculated_release_angle', 0):.2f}°",
                        f"{selected_pitch.get('calculated_avg_angle', 0):.2f}°",
                        f"{selected_pitch.get('detection_rate', 0):.1f}%"
                    ]
                }
                info_df = pd.DataFrame(info_data)
                st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    with col_detail:
        st.markdown("### 🎥 영상 재생")
        
        if pd.notna(selected_pitch.get("output_video_path", None)):
            video_path = selected_pitch["output_video_path"]
            with st.container(border=True):
                display_video_safe(video_path, max_size_mb=100.0, show_debug=False)
        else:
            st.info("이 투구에 대한 영상이 없습니다.")

# --- [UI 구성] ---
def main():
    # 히어로 영역 (전역)
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ⚾ Pitcheezy
        </h1>
        <p style="font-size: 1.1rem; color: #666; margin-top: 0.5rem;">
            Shohei Ohtani 투구 분석 대시보드
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    file_mtime = cfg.FINAL_MERGED_CSV.stat().st_mtime if cfg.FINAL_MERGED_CSV.exists() else 0
    video_df = load_video_analysis(_file_mtime=file_mtime)
    
    # 사이드바
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #e0e0e0; margin-bottom: 1rem;">
            <h2 style="margin: 0; color: #1E88E5;">📊 메뉴</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "페이지 선택",
            [
                "🏠 분석 홈",
                "📈 전체 통계",
                "🎯 구종별 분석",
                "🔄 투구 단계 분석",
                "📅 시간 경향성",
                "🎬 동영상 탐색",
                "🤖 추천 시스템",
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 데이터 정보 카드
        st.markdown("### 📊 데이터 정보")
        if video_df is not None and len(video_df) > 0:
            st.metric("총 투구 수", f"{len(video_df):,}개")
            if "year" in video_df.columns:
                years = sorted(video_df["year"].dropna().unique())
                if len(years) > 0:
                    st.caption(f"📅 연도: {', '.join(map(str, map(int, years)))}")
            elif "game_date" in video_df.columns:
                video_df["game_date"] = pd.to_datetime(video_df["game_date"], errors="coerce")
                years = sorted(video_df["game_date"].dt.year.dropna().unique())
                if len(years) > 0:
                    st.caption(f"📅 연도: {', '.join(map(str, map(int, years)))}")
        else:
            st.caption("데이터 로드 중...")
    
    # 페이지 라우팅
    if "추천 시스템" in page:
        st.markdown("## 🤖 AI 투구 추천 시스템")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.container(border=True):
                st.markdown("### 1️⃣ 상황 설정")
                c1, c2 = st.columns(2)
                balls = c1.number_input("Balls", 0, 3, 0, key="balls_input")
                strikes = c2.number_input("Strikes", 0, 2, 2, key="strikes_input")
                outs = st.selectbox("아웃", [0, 1, 2], key="outs_input")
                on_1b = st.checkbox("1루 주자", key="on_1b")
                on_2b = st.checkbox("2루 주자", key="on_2b")
                on_3b = st.checkbox("3루 주자", key="on_3b")
                score_diff = 0
                inning = 5
                is_batter_lefty = 1
                
                st.info("💡 **Tip:** 2순위 추천 구종과 위치도 함께 비교해보세요!")
                
                prev_pitch_name = st.selectbox(
                    "직전 구종", ["No_Pitch (초구)"] + list(PITCH_MAP.values()), key="prev_pitch"
                )
                if prev_pitch_name == "No_Pitch (초구)":
                    prev_pitch_code = 2
                else:
                    prev_pitch_code = PITCH_MAP_REVERSE[prev_pitch_name]
        
        with col2:
            with st.container(border=True):
                st.markdown("### 2️⃣ AI 전략 수립")
                
                if st.button("🚀 전략 생성", type="primary", use_container_width=True):
                    recommender, loc_model = load_models()
                    
                    if recommender and loc_model:
                        input_data = {
                            "inning": inning, "balls": balls, "strikes": strikes,
                            "outs_when_up": outs, "score_diff": score_diff,
                            "on_1b": int(on_1b), "on_2b": int(on_2b), "on_3b": int(on_3b),
                            "is_batter_lefty": is_batter_lefty, "pitcher_throws_left": 0,
                            "prev_pitch_type_code": prev_pitch_code,
                        }
                        
                        results = recommender.recommend(input_data, top_k=3)
                        res1, res2 = results[0], results[1]
                        name1, id1 = get_pitch_name_and_id(res1)
                        name2, id2 = get_pitch_name_and_id(res2)
                        
                        def predict_location(p_id):
                            df_temp = pd.DataFrame([input_data])
                            for col in recommender.feature_names:
                                if col not in df_temp.columns:
                                    df_temp[col] = 0
                            df_temp = df_temp[recommender.feature_names]
                            df_temp["pitch_type_code"] = p_id
                            return loc_model.predict(df_temp)[0]
                        
                        pred_loc1 = predict_location(id1)
                        pred_loc2 = predict_location(id2)
                        
                        c_res1, c_res2 = st.columns([1, 1])
                        
                        with c_res1:
                            st.markdown("#### 🎯 추천 분석")
                            
                            st.success(f"**1순위: {name1}**")
                            st.caption(f"확률: {res1['probability']*100:.1f}% | 목표: X={pred_loc1[0]:.2f}, Z={pred_loc1[1]:.2f}")
                            
                            st.info(f"**2순위: {name2}**")
                            st.caption(f"확률: {res2['probability']*100:.1f}% | 목표: X={pred_loc2[0]:.2f}, Z={pred_loc2[1]:.2f}")
                            
                            st.markdown("---")
                            st.markdown("**분석 코멘트:**")
                            if res1["probability"] - res2["probability"] < 0.1:
                                st.write("👉 두 구종의 확률 차이가 크지 않습니다. 타자의 반응에 따라 **2순위 구종**을 섞어 던지는 것이 효과적일 수 있습니다.")
                            else:
                                st.write("👉 **1순위 구종**이 압도적으로 추천됩니다. 확실한 결정구로 사용하는 것을 권장합니다.")
                            
                            st.markdown("---")
                            st.markdown("#### 🎬 관련 실제 투구 영상")
                            
                            if video_df is None or len(video_df) == 0:
                                st.caption("영상 분석 데이터셋을 찾을 수 없습니다.")
                            else:
                                st.markdown(f"**1순위 {name1} 예시**")
                                vids1 = find_similar_pitches(video_df, input_data, id1, max_results=2)
                                if not vids1:
                                    st.caption("해당 상황과 비슷한 1순위 예시 영상을 찾지 못했습니다.")
                                else:
                                    for i, v in enumerate(vids1, start=1):
                                        caption = f"{i}. {v.get('game_date', '')} | 카운트 {v.get('balls', '')}-{v.get('strikes', '')}"
                                        st.caption(caption)
                                        video_path = v.get("output_video_path")
                                        if video_path:
                                            display_video_safe(video_path, max_size_mb=100.0, show_debug=False)
                                
                                st.markdown(f"**2순위 {name2} 예시**")
                                vids2 = find_similar_pitches(video_df, input_data, id2, max_results=2)
                                if not vids2:
                                    st.caption("해당 상황과 비슷한 2순위 예시 영상을 찾지 못했습니다.")
                                else:
                                    for i, v in enumerate(vids2, start=1):
                                        caption = f"{i}. {v.get('game_date', '')} | 카운트 {v.get('balls', '')}-{v.get('strikes', '')}"
                                        st.caption(caption)
                                        video_path = v.get("output_video_path")
                                        if video_path:
                                            display_video_safe(video_path, max_size_mb=100.0, show_debug=False)
                        
                        with c_res2:
                            st.markdown("#### 📍 멀티 타겟 로케이션")
                            fig = plot_dual_heatmap(pred_loc1, name1, pred_loc2, name2)
                            st.pyplot(fig)
    
    else:
        if video_df is None or len(video_df) == 0:
            st.warning("영상 분석 데이터를 찾을 수 없습니다.")
            return
        
        if "분석 홈" in page:
            show_analysis_home(video_df)
        elif "전체 통계" in page:
            show_overall_stats_page(video_df)
        elif "구종별 분석" in page:
            show_pitch_type_analysis_page(video_df)
        elif "투구 단계 분석" in page:
            show_pitch_phase_analysis_page(video_df)
        elif "시간 경향성" in page:
            show_temporal_analysis_page(video_df)
        elif "동영상 탐색" in page:
            show_video_explorer_page(video_df)

if __name__ == "__main__":
    main()
