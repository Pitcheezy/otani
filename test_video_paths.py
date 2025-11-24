"""영상 경로 및 재생 가능 여부 테스트 스크립트"""
import sys
from pathlib import Path
import pandas as pd

# 프로젝트 경로 추가
sys.path.insert(0, '.')

import config.config as cfg
from src.data.video_data import load_video_analysis, find_similar_pitches

def test_video_paths():
    """영상 경로 테스트"""
    print("=" * 60)
    print("영상 경로 테스트 시작")
    print("=" * 60)
    
    # 1. 설정 경로 확인
    print("\n1. 설정 경로 확인:")
    print(f"   ANALYZED_VIDEOS_DIR: {cfg.ANALYZED_VIDEOS_DIR}")
    print(f"   존재 여부: {cfg.ANALYZED_VIDEOS_DIR.exists()}")
    
    videos_2025_dir = cfg.DE_RESULTS_DIR / "2025_data" / "videos"
    print(f"\n   2025_data/videos: {videos_2025_dir}")
    print(f"   존재 여부: {videos_2025_dir.exists()}")
    
    # 2. CSV 파일 로드 테스트
    print("\n2. CSV 파일 로드 테스트:")
    csv_path = cfg.FINAL_MERGED_CSV
    print(f"   CSV 경로: {csv_path}")
    print(f"   존재 여부: {csv_path.exists()}")
    
    if not csv_path.exists():
        print("   ❌ CSV 파일을 찾을 수 없습니다!")
        return False
    
    # 3. 데이터 로드 테스트
    print("\n3. 데이터 로드 테스트:")
    try:
        video_df = load_video_analysis(_file_mtime=0)
        if video_df is None or len(video_df) == 0:
            print("   ❌ 데이터를 로드할 수 없습니다!")
            return False
        
        print(f"   ✅ 데이터 로드 성공: {len(video_df)}행")
        print(f"   컬럼 수: {len(video_df.columns)}개")
        
        # output_video_path 컬럼 확인
        if "output_video_path" not in video_df.columns:
            print("   ❌ output_video_path 컬럼이 없습니다!")
            return False
        
        print(f"   ✅ output_video_path 컬럼 존재")
        
    except Exception as e:
        print(f"   ❌ 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 샘플 영상 경로 확인
    print("\n4. 샘플 영상 경로 확인:")
    sample_size = min(10, len(video_df))
    sample_df = video_df.head(sample_size)
    
    found_count = 0
    not_found_count = 0
    not_found_files = []
    
    for idx, row in sample_df.iterrows():
        video_path = row.get("output_video_path")
        if pd.isna(video_path) or not video_path:
            continue
        
        video_file = Path(str(video_path))
        
        # 경로 정규화
        if not video_file.is_absolute():
            video_file = video_file.resolve()
        
        if video_file.exists() and video_file.is_file():
            found_count += 1
            if found_count <= 3:  # 처음 3개만 출력
                print(f"   ✅ [{idx}] {video_file.name}")
                print(f"      경로: {video_file}")
        else:
            not_found_count += 1
            not_found_files.append({
                'index': idx,
                'filename': video_file.name,
                'path': str(video_file),
                'parent_exists': video_file.parent.exists()
            })
            if not_found_count <= 3:  # 처음 3개만 출력
                print(f"   ❌ [{idx}] {video_file.name}")
                print(f"      경로: {video_file}")
                print(f"      부모 폴더 존재: {video_file.parent.exists()}")
    
    print(f"\n   총 확인: {sample_size}개")
    print(f"   ✅ 찾음: {found_count}개")
    print(f"   ❌ 없음: {not_found_count}개")
    
    if not_found_count > 0:
        print(f"\n   ⚠️  찾을 수 없는 파일들:")
        for item in not_found_files[:5]:  # 최대 5개만 표시
            print(f"      - {item['filename']}")
            print(f"        경로: {item['path']}")
            print(f"        부모 폴더 존재: {item['parent_exists']}")
    
    # 5. find_similar_pitches 함수 테스트
    print("\n5. find_similar_pitches 함수 테스트:")
    test_input = {
        "balls": 0,
        "strikes": 2,
        "outs_when_up": 0,
        "on_1b": 0,
        "on_2b": 0,
        "on_3b": 0,
        "inning": 5,
    }
    
    # SL (Slider) 테스트
    try:
        results = find_similar_pitches(video_df, test_input, pitch_id=5, max_results=3)
        print(f"   ✅ SL (Slider) 검색 성공: {len(results)}개 결과")
        
        for i, result in enumerate(results[:3], 1):
            video_path = result.get("output_video_path")
            if video_path:
                video_file = Path(str(video_path))
                if not video_file.is_absolute():
                    video_file = video_file.resolve()
                
                exists = video_file.exists() and video_file.is_file()
                status = "✅" if exists else "❌"
                print(f"      {status} [{i}] {video_file.name}")
                if not exists:
                    print(f"         경로: {video_path}")
    except Exception as e:
        print(f"   ❌ 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 최종 결과
    print("\n" + "=" * 60)
    success_rate = (found_count / sample_size * 100) if sample_size > 0 else 0
    print(f"테스트 결과: {found_count}/{sample_size} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ 테스트 통과: 대부분의 영상 파일을 찾을 수 있습니다.")
        return True
    elif success_rate >= 50:
        print("⚠️  테스트 부분 통과: 일부 영상 파일을 찾을 수 없습니다.")
        return True
    else:
        print("❌ 테스트 실패: 대부분의 영상 파일을 찾을 수 없습니다.")
        return False

if __name__ == "__main__":
    success = test_video_paths()
    sys.exit(0 if success else 1)

