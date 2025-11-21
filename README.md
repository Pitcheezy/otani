# ⚾️ MLB Pitch Recommendation System (Ohtani Shohei Edition)

이 프로젝트는 MLB 레전드 **오타니 쇼헤이(Ohtani Shohei)** 선수의 투구 데이터를 기반으로, 특정 경기 상황(볼카운트, 주자 상황, 점수 차 등)에 최적화된 **투구 구종(Pitch Type)과 위치(Location)**를 추천하는 AI 시스템입니다.

머신러닝(Random Forest)과 강화학습 개념(Behavioral Cloning)을 활용하여 실제 경기에서 투수에게 도움을 줄 수 있는 인사이트를 제공하는 것을 목표로 합니다.

---

## 🚀 주요 기능 (Features)

*   **데이터 자동 수집**: `pybaseball` 라이브러리를 활용하여 MLB Statcast 데이터를 실시간으로 수집합니다.
*   **상황별 최적 구종 추천**: 볼카운트, 주자, 이닝, 타자 유형(좌/우) 등을 고려하여 가장 승률이 높은 구종을 추천합니다.
*   **투구 위치 시각화**: 추천 구종과 함께 스트라이크 존 내 최적의 투구 위치를 히트맵으로 시각화합니다.
*   **웹 대시보드**: `Streamlit` 기반의 웹 인터페이스를 통해 누구나 쉽게 상황을 설정하고 추천 결과를 확인할 수 있습니다.

---

## 🛠 설치 및 실행 (Installation & Usage)

### 1. 환경 설정
Python 3.8 이상 환경에서 실행하는 것을 권장합니다.

```bash
# 저장소 클론
git clone https://github.com/Pitcheezy/ubiquitous-train.git
cd ubiquitous-train

# 가상환경 생성 (선택 사항)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2. 애플리케이션 실행 (Web App)
가장 간편하게 추천 시스템을 경험할 수 있는 방법입니다.

```bash
streamlit run app.py
```
브라우저가 열리면 사이드바에서 경기 상황을 입력하고 **"투구 추천 받기"** 버튼을 누르세요.

### 3. 단계별 스크립트 실행 (Pipeline)
전체 데이터 파이프라인을 직접 실행해보고 싶다면 아래 순서대로 스크립트를 실행하세요.

*   **환경 점검**: `python check_env.py`
*   **데이터 수집**: `python run_phase2_1.py`
*   **피처 엔지니어링**: `python run_phase2_2.py`
*   **모델 학습**: `python run_phase3.py`
*   **추천 성능 테스트**: `python run_phase4.py`
*   **통합 파이프라인(노트북)**: `notebooks/combined_pipeline.ipynb` 실행

---

## 📂 프로젝트 구조 (Project Structure)

```
otani_pitching_project/
├── config/             # 설정 파일 관리
│   ├── __init__.py
│   └── config.py       # 경로, 모델 하이퍼파라미터, 선수 ID 설정 등
├── data/               # 데이터 저장소
│   ├── raw/            # 수집된 원본 데이터 (pitch_data.csv)
│   └── processed/      # 전처리된 데이터 (re288_table.csv 등)
├── models/             # 학습된 모델 저장소 (.joblib 파일)
├── src/                # 소스 코드 (핵심 로직)
│   ├── data/           # 데이터 로드 (loader.py)
│   ├── features/       # 피처 엔지니어링 (engineering.py) - Data Leakage 방지 로직 포함
│   ├── models/         # 모델 학습 (trainer.py, location_trainer.py)
│   └── recommendation/ # 추천 엔진 (recommender.py)
├── notebooks/          # 실험 및 분석용 주피터 노트북
├── tools/              # 유틸리티 스크립트 (build_re288.py)
├── app.py              # Streamlit 웹 애플리케이션 메인 파일
├── main.py             # 전체 파이프라인 실행 스크립트
└── requirements.txt    # 의존성 패키지 목록
```

---

## 🧠 모델 및 알고리즘

### 1. 구종 추천 (Pitch Type Classification)
*   **알고리즘**: Random Forest Classifier
*   **입력 특성**: 볼카운트(Balls, Strikes), 주자 상황(On 1B/2B/3B), 이닝, 점수 차, 타자/투수 주손(Handedness), 직전 구종 등
*   **특이 사항**: Data Leakage를 방지하기 위해 투구 후 결과(구속, 회전수 등)는 학습 피처에서 제외했습니다.

### 2. 위치 추천 (Location Prediction)
*   **접근 방식**: Behavioral Cloning (강화학습 기초)
*   **Reward Engineering**: 상황별 기대 득점 변화량(RE24/RE288)을 기반으로 투수에게 유리한 결과(스트라이크, 아웃)를 낸 투구만 필터링하여 학습합니다.
*   **출력**: 스트라이크 존(Plate X, Z) 좌표 예측 및 KDE(Kernel Density Estimation) 시각화

---

## 📝 개발 히스토리

1.  **환경 설정**: 프로젝트 구조 및 라이브러리 의존성 정의
2.  **데이터 수집**: `pybaseball`을 이용한 오타니 쇼헤이 투구 데이터 확보 (2021~2023)
3.  **피처 엔지니어링**: 경기 상황 변수 추출 및 데이터 누수(Leakage) 제거
4.  **모델 학습**: 구종 분류 모델(Accuracy 약 45~50%) 및 위치 예측 모델 개발
5.  **서비스 구현**: `Streamlit`을 이용한 인터랙티브 웹 애플리케이션 배포

---

## ⚠️ 주의사항
*   `pybaseball` 데이터 수집 시 네트워크 환경에 따라 시간이 소요될 수 있습니다.
*   2024년 오타니 선수는 타자로만 활동했으므로, 투구 데이터 수집 시 2023년 이전 데이터를 사용하는 것이 좋습니다. (Config 설정 확인 필요)