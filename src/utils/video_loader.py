"""
비디오 파일 로딩 및 표시 유틸리티
Streamlit에서 안정적으로 비디오를 표시하기 위한 헬퍼 함수들
"""
from pathlib import Path
from typing import Optional, Tuple
import streamlit as st
import config.config as cfg


def find_video_file(video_path: str | Path) -> Tuple[Optional[Path], bool, dict]:
    """
    비디오 파일을 찾고 검증합니다.
    
    Args:
        video_path: 비디오 파일 경로 (상대/절대 경로 또는 파일명)
    
    Returns:
        Tuple[찾은_파일경로, 파일존재여부, 디버깅정보딕셔너리]
    """
    debug_info = {
        "original_path": str(video_path),
        "file_exists": False,
        "file_size_mb": None,
        "file_extension": None,
        "searched_paths": [],
        "errors": []
    }
    
    video_file = Path(str(video_path))
    
    # 1. 경로 정규화 (절대 경로로 변환)
    try:
        if not video_file.is_absolute():
            video_file = video_file.resolve()
    except Exception as e:
        debug_info["errors"].append(f"경로 정규화 실패: {e}")
    
    # 2. 파일 존재 확인
    file_exists = video_file.exists() and video_file.is_file()
    debug_info["file_exists"] = file_exists
    debug_info["searched_paths"].append(str(video_file))
    
    if file_exists:
        try:
            file_size = video_file.stat().st_size / (1024 * 1024)  # MB
            debug_info["file_size_mb"] = round(file_size, 2)
            debug_info["file_extension"] = video_file.suffix.lower()
        except Exception as e:
            debug_info["errors"].append(f"파일 정보 읽기 실패: {e}")
        return video_file, True, debug_info
    
    # 3. 파일이 없으면 원본 영상 경로에서 찾기 시도
    filename = video_file.name
    original_filename = filename.replace("_analyzed.mp4", ".mp4")
    
    search_paths = [
        cfg.ANALYZED_VIDEOS_DIR,
        cfg.DE_RESULTS_DIR / "2025_data" / "videos",
        cfg.ORIGINAL_VIDEOS_DIR,
    ]
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        # 연도 폴더에서 검색 시도
        import re
        date_match = re.search(r"(\d{4})-\d{2}-\d{2}", original_filename)
        if date_match:
            year = date_match.group(1)
            year_path = base_path / year / original_filename
            debug_info["searched_paths"].append(str(year_path))
            if year_path.exists() and year_path.is_file():
                return year_path.resolve(), True, debug_info
        
        # 재귀 검색
        found_files = list(base_path.rglob(original_filename))
        if found_files:
            found_file = found_files[0].resolve()
            debug_info["searched_paths"].append(str(found_file))
            return found_file, True, debug_info
    
    return None, False, debug_info


def load_video_bytes(video_file: Path, max_size_mb: float = 100.0) -> Tuple[Optional[bytes], Optional[str]]:
    """
    비디오 파일을 bytes로 읽어옵니다.
    
    Args:
        video_file: 비디오 파일 경로
        max_size_mb: 최대 파일 크기 (MB), 초과 시 None 반환
    
    Returns:
        Tuple[비디오바이트데이터, 오류메시지]
    """
    try:
        file_size_mb = video_file.stat().st_size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            return None, f"파일 크기가 너무 큽니다 ({file_size_mb:.1f}MB > {max_size_mb}MB)"
        
        with open(video_file, 'rb') as f:
            video_bytes = f.read()
        
        return video_bytes, None
    except Exception as e:
        return None, f"파일 읽기 실패: {str(e)}"


def display_video_safe(video_path: str | Path, max_size_mb: float = 100.0, show_debug: bool = False) -> bool:
    """
    Streamlit에서 안정적으로 비디오를 표시합니다.
    
    Args:
        video_path: 비디오 파일 경로
        max_size_mb: 최대 파일 크기 (MB)
        show_debug: 디버깅 정보 표시 여부
    
    Returns:
        성공 여부
    """
    # 1. 파일 찾기
    video_file, file_exists, debug_info = find_video_file(video_path)
    
    if not file_exists or video_file is None:
        st.warning("⚠️ 영상 파일을 찾을 수 없습니다")
        if show_debug:
            with st.expander("🔍 디버깅 정보"):
                st.json(debug_info)
        return False
    
    # 2. 파일 정보 표시
    file_size_mb = debug_info.get("file_size_mb", 0)
    if file_size_mb:
        st.caption(f"📁 파일 크기: {file_size_mb:.1f}MB | 경로: `{video_file.name}`")
    
    # 3. 파일 크기 체크
    if file_size_mb and file_size_mb > max_size_mb:
        st.error(f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.1f}MB > {max_size_mb}MB)")
        st.info("💡 파일 크기를 줄이거나 다른 영상을 선택해주세요.")
        return False
    
    # 4. 비디오 bytes 로드
    video_bytes, error_msg = load_video_bytes(video_file, max_size_mb)
    
    if video_bytes is None:
        st.error(f"❌ 영상 로드 실패: {error_msg}")
        if show_debug:
            with st.expander("🔍 상세 오류 정보"):
                st.write(f"**오류**: {error_msg}")
                st.write(f"**경로**: `{video_file}`")
                st.write(f"**파일 존재**: {file_exists}")
                st.json(debug_info)
        return False
    
    # 5. Streamlit에 비디오 표시
    try:
        # bytes를 직접 전달 (가장 안정적인 방법)
        st.video(video_bytes)
        return True
    except Exception as e:
        st.error(f"❌ 영상 재생 오류: {str(e)}")
        if show_debug:
            with st.expander("🔍 상세 오류 정보"):
                st.write(f"**오류**: {e}")
                st.write(f"**경로**: `{video_file}`")
                st.write(f"**파일 크기**: {file_size_mb:.1f}MB")
                st.write(f"**파일 존재**: {file_exists}")
                st.json(debug_info)
        return False


