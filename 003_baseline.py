import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


print("=" * 70)
print("베이스라인(Baseline) 모델 평가")
print("=" * 70)

# 1. 데이터 로드
df = pd.read_csv('f1_dataset.csv')

# 2. X, y 분리 및 미래 정보 차단
groups = df['raceId']
drop_cols = ['raceId', 'driverId', 'constructorId', 'driverRef', 'Team',
             'positionOrder', 'Lap_Time_sec', 'year', 'round',
             'Circuit_Dry_Median', 'Target_Pace_Ratio', 'Is_Survived']

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Target_Pace_Ratio']
is_wet_flag = df['Is_Wet']

# 3. 그룹 분할 (새로운 경기 환경 시뮬레이션)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
is_wet_test = is_wet_flag.iloc[test_idx].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 초(sec) 단위 오차 복원용
baseline_test = df['Circuit_Dry_Median'].iloc[test_idx].values
y_test_sec = df['Lap_Time_sec'].iloc[test_idx].values


# 평가 함수 (비가 온 Test 데이터에 대해서만)
def evaluate_baseline(model, name):
    model.fit(X_train_scaled, y_train)
    y_pred_ratio = model.predict(X_test_scaled)

    wet_indices = np.where(is_wet_test == 1)[0]

    if len(wet_indices) == 0:
        print("Test 셋에 비 온 날 데이터가 없습니다. random_state를 변경해보세요.")
        return

    y_test_wet_ratio = y_test.iloc[wet_indices]
    y_pred_wet_ratio = y_pred_ratio[wet_indices]

    y_test_wet_sec = y_test_sec[wet_indices]
    y_pred_wet_sec = y_pred_ratio[wet_indices] * baseline_test[wet_indices]

    rmse = np.sqrt(mean_squared_error(y_test_wet_sec, y_pred_wet_sec))
    r2 = r2_score(y_test_wet_ratio, y_pred_wet_ratio)
    print(f"{name:<25} | 빗길 RMSE: {rmse:>6.3f}초 | 빗길 R²: {r2:>6.2f}")


# 4. 베이스라인
print("\n[Baseline] 빗길(Wet) 환경에서의 성능")

# Baseline 1: 단순 중앙값 찍기
dummy = DummyRegressor(strategy='median')
evaluate_baseline(dummy, "1. Simple Median")

# Baseline 2: 다중 선형 회귀
lr = LinearRegression()
evaluate_baseline(lr, "2. Linear Regression")
