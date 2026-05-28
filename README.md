# Machine Learning Project - Formular 1 - 맑은 날 대비 비가 오는 환경에서의 랩타임 비율을 예측하여 레인 마스터 찾기

기상 악화(우천) 조건에서 F1 차량의 랩타임(Lap Time) 저하 비율을 기계학습으로 예측하고, 각 선수가 이 예측 대비 얼마나 잘 달렸는지를 기준으로 삼아, 팀 간 기계적 성능 차이를 최대한 배제하며 좋은 주행 능력을 보여준 **'레인 마스터(Rain Master)'**를 정량적으로 선별하는 프로젝트입니다.

## 데이터셋 구조
코드를 실행하기 전에 Kaggle 원본 데이터셋을 다운로드하여 아래와 같은 폴더 구조로 배치해야 합니다.

* [Formula 1 World Championship (1950 - 2024) 데이터셋](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
* [F1 Weather Dataset (2018-2023) 데이터셋](https://www.kaggle.com/datasets/quantumkaze/f1-weather-dataset-2018-2023)

```text
ML_Project_f1/
│
├── championship/                   # Formula 1 World Championship 원본 데이터
│   ├── constructors.csv
│   ├── constructor_standings.csv
│   ├── drivers.csv
│   ├── lap_times.csv
│   ├── pit_stops.csv
│   ├── qualifying.csv
│   ├── races.csv
│   ├── results.csv
│   └── ...
│
├── weather/                        # F1 Weather 원본 데이터
│   └── F1 Weather(2023-2018).csv
│
├── 001_data_preprocessing.py       # 데이터 전처리
├── 002_data_split_analysis.py      # GroupShuffleSplit을 이용하여 경기 단위 데이터 분할을 했을 때의 분포 확인
├── 003_baseline.py                 # 베이스라인 모델 실험
├── 004_1_tuning_1.py               # 개별 모델 하이퍼파라미터 튜닝
├── 004_2_tuning_2.py               # 스태킹 앙상블 구축 및 MAE, RMSE, R2 Score 평가 / 비교
├── 005_analysis.py                 # 순열 피처 중요도 및 레인 마스터 순위 산출, 시각화
├── requirements.txt                # 필요 패키지 목록
└── README.md                       # 가이드
```

## 재현 및 실행 방법

**1. 패키지 설치**  
Python 3.9 이상의 환경에서 아래 명령어를 통해 필수 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

**2. 코드 실행 순서**  
데이터 전처리부터 최종 분석 및 시각화까지, 번호 순서대로 실행합니다.

1. `python 001_data_preprocessing.py`
   - 서로 다른 기록 주기를 가진 경기 기록과 1분 단위 기상 데이터를 병합(`merge_asof`)하고, '연료 소모율', '클린 에어' 등의 파생 변수를 추가하여 최종 분석용 `f1_dataset.csv`를 생성합니다.
2. `python 002_data_split_analysis.py`
   - 모델 학습 시 미래 데이터가 유출되는 데이터 누수(Data Leakage)를 방지하기 위해, 경기(RaceId) 단위로 Train/Test 세트가 올바르게 분할되는지 검증합니다.
3. `python 003_baseline.py`
   - 단순 중앙값 예측과 다중 선형 회귀 모델을 구축하여, 비선형적인 빗길 랩타임 예측에 대한 베이스라인 성능 한계를 확인합니다.
4. `python 004_1_tuning_1.py`
   - 비가 온 날(Is_Wet=1)의 데이터에 10배의 가중치를 부여한 상태로 Ridge, XGBoost, HistGradientBoosting 모델의 최적 하이퍼파라미터를 탐색합니다.
5. `python 004_2_tuning_2.py`
   - 튜닝된 XGBoost와 HGB를 베이스 러너로 두고, Ridge를 메타 모델로 사용하는 스태킹 회귀(Stacking Regressor) 앙상블 아키텍처를 학습합니다.
6. `python 005_analysis.py`
   - 최종 스태킹 모델을 기반으로 순열 피처 중요도(Permutation Importance)를 추출하고, 예측 오차에 기반한 '한계 돌파력', '팀 동료 우위', '악천후 생존율'을 40/20/40 비율로 결합하여 레인 마스터 랭킹을 산출합니다.

## 결과물

`005_analysis.py` 실행을 완료하면 프로젝트 경로에 아래 3개의 분석 이미지가 자동 저장됩니다.

1. **`feature_importance.png`**
   - 예측 모델에 가장 큰 영향을 미치는 주요 환경 변수 Top 7 분석 그래프 (우천 시 습도 및 노면 온도의 중요성 확인 가능).
2. **`rain_master_stacked.png`**
   - 3가지 지표(한계 돌파력 40%, 팀 동료 우위 20%, 악천후 생존율 40%)의 비율과 점수를 누적 막대로 시각화한 레인 마스터 랭킹 (Top 20).
3. **`rain_master_heatmap.png`**
   - Top 20 드라이버들이 각각의 세부 지표에서 강점을 가지는 영역을 100점 만점으로 환산하여 보여주는 히트맵.
