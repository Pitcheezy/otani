otani_pitching_project/
├── config/             # 설정 파일 관리
│   ├── __init__.py
│   └── config.py       # 경로, 모델 하이퍼파라미터, 랜덤 시드 등 상수 정의
├── data/               # 데이터 저장소
│   ├── raw/            # 수집된 원본 데이터 (pitch_data.csv)
│   └── processed/      # 전처리된 데이터
├── models/             # 학습된 모델 저장소 (.joblib 파일)
├── src/                # 소스 코드
│   ├── __init__.py
│   ├── data/           # 데이터 로드 및 전처리
│   │   ├── __init__.py
│   │   ├── loader.py       # 데이터 로딩 및 저장 함수
│   │   └── preprocessor.py # 결측치 처리, 기본 전처리
│   ├── features/       # 피처 엔지니어링
│   │   ├── __init__.py
│   │   └── engineering.py  # 상황 정보 추출 (Leakage 제거된 피처 생성)
│   ├── models/         # 모델 학습 및 평가
│   │   ├── __init__.py
│   │   └── trainer.py      # 학습 클래스 정의
│   └── recommendation/ # 추천 서비스 로직
│       ├── __init__.py
│       └── recommender.py  # 실제 추천 함수
├── notebooks/          # 실험용 주피터 노트북
│   └── 01_data_collection_test.ipynb
├── requirements.txt    # 라이브러리 명세서
├── main.py             # (New) 전체 파이프라인 실행 스크립트
└── README.md

개발 순서 요약

1. 환경 설정 (venv, config.py)
2. 데이터 로더 구현 (src/data/loader.py)
3. 피처 엔지니어링 구현 (src/features/engineering.py) - Data Leakage 해결
4. 모델 학습 스크립트 실행 (src/models/trainer.py) -> 모델 파일 생성
5. 추천기 구현 (src/recommendation/recommender.py)
6. 메인 실행 파일 작성 (main.py or app.py)