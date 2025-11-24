# 비디오 표시 문제 해결 요약

## 🔍 문제 분석

### 원인 우선순위

1. **가장 가능성 높음**: Streamlit이 로컬 파일 경로 문자열을 직접 처리하지 못함
   - `st.video(str(path))`는 URL이나 bytes를 기대함
   - 로컬 경로 문자열은 브라우저에서 접근 불가
   - **해결**: 파일을 bytes로 읽어서 전달

2. **중간**: 경로 문제
   - 절대 경로가 올바르지 않거나 접근 불가
   - Windows 경로 형식 문제
   - **해결**: Path 객체로 정규화 후 bytes로 읽기

3. **낮음**: 코덱/파일 크기 문제
   - 브라우저 미지원 코덱(H.265 등)
   - 파일이 너무 큼
   - **해결**: 파일 크기 체크 및 경고, 코덱 정보 로깅

## ✅ 적용된 해결책

### 선택한 방법: **bytes로 읽어서 전달**

**장점**:
- ✅ 로컬/배포 환경 모두 동작
- ✅ 가장 안정적이고 범용적
- ✅ Streamlit 공식 권장 방식

**단점**:
- ⚠️ 대용량 파일 시 메모리 사용 증가 (100MB 제한 추가)

## 📝 변경 사항

### 1. 새로운 유틸리티 모듈 생성

**파일**: `src/utils/video_loader.py`

주요 함수:
- `find_video_file()`: 비디오 파일 찾기 및 검증
- `load_video_bytes()`: 비디오 파일을 bytes로 읽기
- `display_video_safe()`: Streamlit에서 안정적으로 비디오 표시

### 2. app.py 수정

**변경 위치**:
1. `show_video_explorer_page()` 함수 (동영상 탐색 페이지)
2. 추천 시스템 페이지의 1순위 영상
3. 추천 시스템 페이지의 2순위 영상

**변경 내용**:
- 기존: `st.video(str(video_file))` (로컬 경로 문자열 전달)
- 변경: `display_video_safe(video_path)` (bytes 기반 안정적 로딩)

## 🎯 개선 사항

### 1. 안정성 향상
- bytes 기반 전달로 로컬/배포 환경 모두 지원
- 파일 크기 체크 (100MB 제한)
- 자동 경로 검색 및 fallback

### 2. 디버깅 개선
- 상세한 디버깅 정보 제공
- 파일 존재 여부, 크기, 경로 정보 표시
- 오류 발생 시 자세한 로그 제공

### 3. 코드 중복 제거
- 중복된 비디오 로딩 로직을 유틸리티 함수로 통합
- 유지보수성 향상

## 🚀 사용 방법

### 기본 사용

```python
from src.utils.video_loader import display_video_safe

# 간단한 사용
display_video_safe(video_path)

# 디버깅 정보 포함
display_video_safe(video_path, max_size_mb=100.0, show_debug=True)
```

### 파라미터

- `video_path`: 비디오 파일 경로 (str 또는 Path)
- `max_size_mb`: 최대 파일 크기 (MB, 기본값: 100.0)
- `show_debug`: 디버깅 정보 표시 여부 (기본값: False)

## 📊 성능 고려사항

### 메모리 사용
- 파일 크기 제한: 100MB (기본값)
- 대용량 파일의 경우 메모리 사용량 증가
- 필요시 `max_size_mb` 파라미터 조정 가능

### 파일 검색
- 자동으로 여러 경로에서 파일 검색
- 원본 영상 경로 fallback 지원
- 재귀 검색으로 파일 찾기

## 🔧 문제 해결

### 영상이 여전히 표시되지 않는 경우

1. **파일 존재 확인**
   - `show_debug=True`로 설정하여 디버깅 정보 확인
   - 파일 경로, 크기, 존재 여부 확인

2. **파일 크기 확인**
   - 100MB 이상인 경우 `max_size_mb` 파라미터 증가
   - 또는 파일 크기 줄이기

3. **코덱 문제**
   - 브라우저가 지원하지 않는 코덱일 수 있음
   - H.264 코덱으로 재인코딩 권장

4. **경로 문제**
   - 절대 경로 확인
   - 파일 접근 권한 확인

## 📚 참고 자료

- [Streamlit Video Documentation](https://docs.streamlit.io/library/api-reference/media/st.video)
- [Pathlib Documentation](https://docs.python.org/3/library/pathlib.html)

---

**수정 완료일**: 2025년
**수정 파일**: 
- `src/utils/video_loader.py` (신규)
- `src/utils/__init__.py` (신규)
- `app.py` (수정)


