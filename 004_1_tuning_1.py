import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


print("=" * 80)
print("모델 튜닝")
print("=" * 80)

df = pd.read_csv('f1_dataset.csv')
groups = df['raceId']
drop_cols = ['raceId', 'driverId', 'constructorId', 'driverRef', 'Team',
             'positionOrder', 'Lap_Time_sec', 'year', 'round',
             'Circuit_Dry_Median', 'Target_Pace_Ratio', 'Is_Survived']

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Target_Pace_Ratio']

weights = np.where(df['Is_Wet'] == 1, 10.0, 1.0)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
weights_train = weights[train_idx]
groups_train = groups.iloc[train_idx]

is_wet_test = df['Is_Wet'].iloc[test_idx].values
baseline_test = df['Circuit_Dry_Median'].iloc[test_idx].values
y_test_sec = df['Lap_Time_sec'].iloc[test_idx].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 하이퍼파라미터 탐색 범위
ridge_grid = {'alpha': [0.1, 1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 200.0]}

xgb_grid = {
    'n_estimators': [100, 150, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'reg_lambda': [0.0, 1.0, 5.0, 10.0, 20.0]
}

hgb_grid = {
    'max_iter': [100, 150, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_regularization': [0.0, 1.0, 5.0, 10.0, 20.0]
}

gkf = GroupKFold(n_splits=3)

def print_dual_evaluation(y_pred_ratio, name, best_params):
    y_pred_sec = y_pred_ratio * baseline_test
    wet_indices = np.where(is_wet_test == 1)[0]
    dry_indices = np.where(is_wet_test == 0)[0]

    print(f"[최적 파라미터] {best_params}")
    if len(wet_indices) > 0:
        rmse_wet = np.sqrt(mean_squared_error(y_test_sec[wet_indices], y_pred_sec[wet_indices]))
        mae_wet = mean_absolute_error(y_test_sec[wet_indices], y_pred_sec[wet_indices])
        r2_wet = r2_score(y_test.iloc[wet_indices], y_pred_ratio[wet_indices])
        print(f"[빗길] RMSE: {rmse_wet:>6.3f}초 | MAE: {mae_wet:>6.3f}초 | R²: {r2_wet:>6.2f}")
    if len(dry_indices) > 0:
        rmse_dry = np.sqrt(mean_squared_error(y_test_sec[dry_indices], y_pred_sec[dry_indices]))
        mae_dry = mean_absolute_error(y_test_sec[dry_indices], y_pred_sec[dry_indices])
        r2_dry = r2_score(y_test.iloc[dry_indices], y_pred_ratio[dry_indices])
        print(f"[맑은 길] RMSE: {rmse_dry:>6.3f}초 | MAE: {mae_dry:>6.3f}초 | R²: {r2_dry:>6.2f}\n")

def tune_and_eval(model, param_grid, name, search_type='grid'):
    print(f"{name} 최적화 탐색 중...")
    if search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, n_iter=25, cv=gkf,
                                    scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
    else:
        search = GridSearchCV(model, param_grid, cv=gkf,
                              scoring='neg_mean_squared_error', n_jobs=-1)

    search.fit(X_train_scaled, y_train, groups=groups_train, sample_weight=weights_train)
    best_model = search.best_estimator_
    best_params = search.best_params_

    y_pred_ratio = best_model.predict(X_test_scaled)
    print_dual_evaluation(y_pred_ratio, name, best_params)
    return best_model

print("\n--- 개별 모델 튜닝 시작 ---")
best_ridge = tune_and_eval(Ridge(random_state=42), ridge_grid, "1. Ridge Regression", search_type='grid')
best_xgb = tune_and_eval(xgb.XGBRegressor(random_state=42, n_jobs=-1), xgb_grid, "2. XGBoost", search_type='random')
best_hgb = tune_and_eval(HistGradientBoostingRegressor(random_state=42), hgb_grid, "3. HistGradientBoosting", search_type='random')
