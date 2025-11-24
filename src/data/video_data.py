# src/data/video_data.py
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import streamlit as st

import config.config as cfg

# 구종 ID -> Statcast pitch_type 코드 매핑
PITCH_SHORT_CODE = {
    0: "CU",
    1: "FC",
    2: "FF",
    3: "FS",
    4: "SI",
    5: "SL",
    6: "ST",
}


@st.cache_data
def load_video_analysis(_file_mtime: float = 0) -> pd.DataFrame | None:
    """
    data_extraction_mlb에서 생성한 통합 분석 CSV 로드
    (Statcast + 영상 분석 + 팔각도 + 영상 경로)
    
    Args:
        _file_mtime: 파일 수정 시간 (캐시 키로 사용, 파일 변경 시 자동 갱신)
    """
    csv_path: Path = cfg.FINAL_MERGED_CSV

    if not csv_path.exists():
        st.warning(f"영상 분석 통합 CSV를 찾을 수 없습니다: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)

    # 날짜 컬럼이 있으면 datetime으로 (먼저 변환하여 필터링에 사용)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["year"] = df["game_date"].dt.year
    
    # 2025년 영상이 있는 데이터만 필터링
    # output_video_path가 있고, 날짜가 2025년인 데이터만 사용
    if "output_video_path" in df.columns:
        # output_video_path가 있는 행만 필터링
        has_video = df["output_video_path"].notna()
        
        # 2025년 데이터만 필터링 (영상 경로에서 날짜 추출 또는 game_date 사용)
        if "year" in df.columns:
            is_2025 = df["year"] == 2025
        else:
            # output_video_path에서 2025 추출
            import re
            is_2025 = df["output_video_path"].apply(
                lambda x: bool(re.search(r"2025", str(x))) if pd.notna(x) else False
            )
        
        # 2025년 영상이 있는 데이터만 필터링
        df = df[has_video & is_2025].copy()
        
        if len(df) == 0:
            st.warning("2025년 영상 데이터를 찾을 수 없습니다.")
            return None
        
        # 중복 제거 (game_pk, at_bat_number, pitch_number 기준)
        df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="first")
        
        st.info(f"📹 2025년 영상 데이터 {len(df)}개를 로드했습니다.")

    # 자주 쓰는 컬럼 타입 정리
    int_cols = [
        "balls",
        "strikes",
        "outs_when_up",
        "on_1b",
        "on_2b",
        "on_3b",
        "inning",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # 날짜/구종 코드 파생 (data_extraction_mlb/app.py 와 동일 로직)
    import re

    if "output_video_path" in df.columns:
        # 상대 경로를 절대 경로로 변환
        def normalize_video_path(path_str):
            if pd.isna(path_str) or not path_str:
                return path_str
            path_str = str(path_str).strip()
            
            # 파일명 추출 (경로 구분자 처리)
            if "/" in path_str:
                filename = path_str.split("/")[-1]
            elif "\\" in path_str:
                filename = path_str.split("\\")[-1]
            else:
                filename = path_str
            
            # 파일명이 비어있으면 원래 경로 반환
            if not filename:
                return path_str
            
            # 1순위: analyzed_videos 폴더 확인 (분석된 영상)
            absolute_path = cfg.ANALYZED_VIDEOS_DIR / filename
            if absolute_path.exists() and absolute_path.is_file():
                return str(absolute_path.resolve())
            
            # 2순위: 2025_data/videos 폴더 확인
            videos_2025_dir = cfg.DE_RESULTS_DIR / "2025_data" / "videos"
            if videos_2025_dir.exists():
                video_2025_path = videos_2025_dir / filename
                if video_2025_path.exists() and video_2025_path.is_file():
                    return str(video_2025_path.resolve())
            
            # 3순위: 원본 영상 폴더 확인 (ohtani_videos/2025/)
            # 파일명에서 _analyzed 제거하여 원본 파일명 찾기
            original_filename = filename.replace("_analyzed.mp4", ".mp4")
            
            # 날짜에서 연도 추출 (예: 2025-10-28 -> 2025)
            import re
            date_match = re.search(r"(\d{4})-\d{2}-\d{2}", original_filename)
            if date_match:
                year = date_match.group(1)
                # 연도 폴더 내에서 검색 (예: ohtani_videos/2025/)
                if cfg.ORIGINAL_VIDEOS_DIR.exists():
                    original_video_path = cfg.ORIGINAL_VIDEOS_DIR / year / original_filename
                    if original_video_path.exists() and original_video_path.is_file():
                        return str(original_video_path.resolve())
                    
                    # 연도 폴더 없이 직접 확인
                    original_video_path_direct = cfg.ORIGINAL_VIDEOS_DIR / original_filename
                    if original_video_path_direct.exists() and original_video_path_direct.is_file():
                        return str(original_video_path_direct.resolve())
            
            # 4순위: 모든 폴더에서 재귀적으로 검색 (마지막 수단)
            search_dirs = []
            if cfg.ORIGINAL_VIDEOS_DIR.exists():
                search_dirs.append(cfg.ORIGINAL_VIDEOS_DIR)
            if cfg.ANALYZED_VIDEOS_DIR.exists():
                search_dirs.append(cfg.ANALYZED_VIDEOS_DIR)
            if videos_2025_dir.exists():
                search_dirs.append(videos_2025_dir)
            
            for search_dir in search_dirs:
                # 재귀적으로 원본 파일명 검색
                found_files = list(search_dir.rglob(original_filename))
                if found_files:
                    return str(found_files[0].resolve())
                # _analyzed 버전도 검색
                found_files = list(search_dir.rglob(filename))
                if found_files:
                    return str(found_files[0].resolve())
            
            # 파일을 찾지 못한 경우에도 올바른 경로 구조 반환 (나중에 경고 표시)
            # 원본 영상 경로를 우선 반환 (가장 가능성 높음)
            if date_match:
                year = date_match.group(1)
                fallback_path = cfg.ORIGINAL_VIDEOS_DIR / year / original_filename
                return str(fallback_path.resolve())
            
            return str(absolute_path.resolve())
        
        df["output_video_path"] = df["output_video_path"].apply(normalize_video_path)
        
        df["pitch_type_extracted"] = df["output_video_path"].apply(
            lambda x: re.search(r"_(FF|ST|CU|SL|FS|FC|SI|CH|KN)_", str(x)).group(1)
            if x and re.search(r"_(FF|ST|CU|SL|FS|FC|SI|CH|KN)_", str(x))
            else None
        )
        df["date_extracted"] = df["output_video_path"].apply(
            lambda x: re.search(r"(\d{4}-\d{2}-\d{2})", str(x)).group(1)
            if x and re.search(r"(\d{4}-\d{2}-\d{2})", str(x))
            else None
        )
        df["date"] = pd.to_datetime(df["date_extracted"], errors="coerce")
    else:
        df["pitch_type_extracted"] = None
        df["date"] = pd.NaT

    return df


def find_similar_pitches(
    video_df: pd.DataFrame,
    input_data: Dict[str, Any],
    pitch_id: int,
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    현재 상황(input_data) + 추천 구종(pitch_id)에 비슷한 과거 투구들을 찾아
    영상/팔각도/탐지율 정보를 반환.
    """
    if video_df is None or len(video_df) == 0:
        return []

    if "pitch_type" not in video_df.columns:
        return []

    pitch_code = PITCH_SHORT_CODE.get(pitch_id)
    if pitch_code is None:
        return []

    df = video_df.copy()

    # 1차 필터: 구종 + 카운트 + 아웃 + 주자 + 이닝
    cond = (df["pitch_type"] == pitch_code)

    # input_data에 있는 경우에만 필터 적용
    def _safe_eq(col: str, key: str):
        nonlocal cond
        if col in df.columns and key in input_data:
            cond = cond & (df[col] == input_data[key])

    _safe_eq("balls", "balls")
    _safe_eq("strikes", "strikes")
    _safe_eq("outs_when_up", "outs_when_up")
    _safe_eq("on_1b", "on_1b")
    _safe_eq("on_2b", "on_2b")
    _safe_eq("on_3b", "on_3b")
    _safe_eq("inning", "inning")

    filtered = df[cond]

    # 결과가 너무 적으면 조건 완화: 카운트만 맞추고 나머지는 풀어줌
    if len(filtered) < max_results:
        cond_relaxed = df["pitch_type"] == pitch_code
        if "balls" in df.columns and "balls" in input_data:
            cond_relaxed &= df["balls"] == input_data["balls"]
        if "strikes" in df.columns and "strikes" in input_data:
            cond_relaxed &= df["strikes"] == input_data["strikes"]
        filtered = df[cond_relaxed]

    if len(filtered) == 0:
        return []

    # 탐지율 높은 순으로 정렬 (없으면 원래 순서 유지)
    if "detection_rate" in filtered.columns:
        filtered = filtered.sort_values("detection_rate", ascending=False)

    filtered = filtered.head(max_results)

    results: List[Dict[str, Any]] = []
    for _, row in filtered.iterrows():
        results.append(
            {
                "game_date": row.get("game_date"),
                "balls": row.get("balls"),
                "strikes": row.get("strikes"),
                "outs": row.get("outs_when_up"),
                "on_1b": row.get("on_1b"),
                "on_2b": row.get("on_2b"),
                "on_3b": row.get("on_3b"),
                "pitch_type": row.get("pitch_type"),
                "output_video_path": row.get("output_video_path"),
                "calculated_release_angle": row.get("calculated_release_angle"),
                "detection_rate": row.get("detection_rate"),
                "description": row.get("description") or row.get("des"),
            }
        )

    return results