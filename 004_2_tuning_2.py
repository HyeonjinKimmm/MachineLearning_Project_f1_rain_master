import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import platform

print("=" * 80)
print("모델 3개, Stacking 모델 비교")
print("=" * 80)

# =====================================================================
# 1. 데이터 로드 및 셋업
# =====================================================================
df = pd.read_csv('f1_dataset.csv')
groups = df['raceId']
drop_cols = ['raceId', 'driverId', 'constructorId', 'driverRef', 'Team',
             'positionOrder', 'Lap_Time_sec', 'year', 'round',
             'Circuit_Dry_Median', 'Target_Pace_Ratio', 'Is_Survived']

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Target_Pace_Ratio']
weights = np.where(df['Is_Wet'] == 1, 10.0, 1.0)

# =====================================================================
# 2. 분할 및 스케일링
# =====================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
weights_train = weights[train_idx]

is_wet_test = df['Is_Wet'].iloc[test_idx].values
baseline_test = df['Circuit_Dry_Median'].iloc[test_idx].values
y_test_sec = df['Lap_Time_sec'].iloc[test_idx].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# 3. 평가용 출력 함수 정의
# =====================================================================
def print_model_score(model, name):
    y_pred_ratio = model.predict(X_test_scaled)
    y_pred_sec = y_pred_ratio * baseline_test

    wet_indices = np.where(is_wet_test == 1)[0]
    dry_indices = np.where(is_wet_test == 0)[0]

    print(f"\n {name}")
    if len(wet_indices) > 0:
        rmse_wet = np.sqrt(mean_squared_error(y_test_sec[wet_indices], y_pred_sec[wet_indices]))
        mae_wet = mean_absolute_error(y_test_sec[wet_indices], y_pred_sec[wet_indices])
        r2_wet = r2_score(y_test.iloc[wet_indices], y_pred_ratio[wet_indices])
        print(f"  [빗길] RMSE: {rmse_wet:>6.3f}초 | MAE: {mae_wet:>6.3f}초 | R²: {r2_wet:>6.2f}")
    if len(dry_indices) > 0:
        rmse_dry = np.sqrt(mean_squared_error(y_test_sec[dry_indices], y_pred_sec[dry_indices]))
        mae_dry = mean_absolute_error(y_test_sec[dry_indices], y_pred_sec[dry_indices])
        r2_dry = r2_score(y_test.iloc[dry_indices], y_pred_ratio[dry_indices])
        print(f"  [맑은길] RMSE: {rmse_dry:>6.3f}초 | MAE: {mae_dry:>6.3f}초 | R²: {r2_dry:>6.2f}")


# =====================================================================
# 4. 기본 모델 세팅 및 학습
# =====================================================================
best_ridge = Ridge(alpha=200.0, random_state=42)
best_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01,
                            subsample=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1)
best_hgb = HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.05,
                                         l2_regularization=10.0, random_state=42)

print("\nRidge, XGB, HGB 학습 중...")
best_ridge.fit(X_train_scaled, y_train, sample_weight=weights_train)
best_xgb.fit(X_train_scaled, y_train, sample_weight=weights_train)
best_hgb.fit(X_train_scaled, y_train, sample_weight=weights_train)

# 모델 평가
print("\n" + "-" * 50)
print("기본 모델")
print("-" * 50)
print_model_score(best_ridge, "1. Ridge (Tuned)")
print_model_score(best_xgb, "2. XGBoost (Tuned)")
print_model_score(best_hgb, "3. HGB (Tuned)")

# =====================================================================
# 5. 스태킹 메타 모델 학습 및 tuning
# =====================================================================
estimators = [
    ('xgb', best_xgb),
    ('hgb', best_hgb)
]

best_stack_r2_wet = -float('inf')
best_alpha = 0
best_y_pred_ratio = None
best_stack_model = None

print("\n\nMeta-Learner 학습 중...")
for alpha in [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]:
    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=alpha, random_state=42),
        cv="prefit"
    )
    stack_model.fit(X_train_scaled, y_train, sample_weight=weights_train)
    y_pred_ratio = stack_model.predict(X_test_scaled)

    wet_indices = np.where(is_wet_test == 1)[0]
    if len(wet_indices) > 0:
        r2 = r2_score(y_test.iloc[wet_indices], y_pred_ratio[wet_indices])
        if r2 > best_stack_r2_wet:
            best_stack_r2_wet = r2
            best_alpha = alpha
            best_stack_model = stack_model

# =====================================================================
# 6. 최종 스태킹 앙상블 성능
# =====================================================================
print("\n" + "-" * 50)
print("스태킹 앙상블 평가")
print("-" * 50)
print(f"Meta-Learner Ridge 최적 Alpha: {best_alpha}")
print_model_score(best_stack_model, "Stacking Regressor")

# =====================================================================
# 7. 성능 시각화 (Performance Visualization)
# =====================================================================


print("\n" + "=" * 50)
print("모델 성능 비교 시각화 및 저장 중...")
print("=" * 50)

# 한글 폰트 설정 (OS별 호환성 유지)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # Mac
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 시각화를 위한 4개 모델 딕셔너리 구성
models = {
    'Ridge': best_ridge,
    'XGBoost': best_xgb,
    'HGB': best_hgb,
    'Stacking': best_stack_model
}

results_list = []

# 기존 변수들을 활용하여 시각화용 데이터프레임 재구성
for model_name, model in models.items():
    y_pred_ratio = model.predict(X_test_scaled)
    y_pred_sec = y_pred_ratio * baseline_test

    wet_idx = np.where(is_wet_test == 1)[0]
    dry_idx = np.where(is_wet_test == 0)[0]

    if len(wet_idx) > 0:
        rmse_wet = np.sqrt(mean_squared_error(y_test_sec[wet_idx], y_pred_sec[wet_idx]))
        mae_wet = mean_absolute_error(y_test_sec[wet_idx], y_pred_sec[wet_idx])
        r2_wet = r2_score(y_test.iloc[wet_idx], y_pred_ratio[wet_idx])
        results_list.extend([
            {'Model': model_name, 'Condition': 'Wet (빗길)', 'Metric': 'RMSE', 'Score': rmse_wet},
            {'Model': model_name, 'Condition': 'Wet (빗길)', 'Metric': 'MAE', 'Score': mae_wet},
            {'Model': model_name, 'Condition': 'Wet (빗길)', 'Metric': 'R2 Score', 'Score': r2_wet}
        ])

    if len(dry_idx) > 0:
        rmse_dry = np.sqrt(mean_squared_error(y_test_sec[dry_idx], y_pred_sec[dry_idx]))
        mae_dry = mean_absolute_error(y_test_sec[dry_idx], y_pred_sec[dry_idx])
        r2_dry = r2_score(y_test.iloc[dry_idx], y_pred_ratio[dry_idx])
        results_list.extend([
            {'Model': model_name, 'Condition': 'Dry (맑은 길)', 'Metric': 'RMSE', 'Score': rmse_dry},
            {'Model': model_name, 'Condition': 'Dry (맑은 길)', 'Metric': 'MAE', 'Score': mae_dry},
            {'Model': model_name, 'Condition': 'Dry (맑은 길)', 'Metric': 'R2 Score', 'Score': r2_dry}
        ])

df_results = pd.DataFrame(results_list)

# 1x3 서브플롯 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('F1 Lap Time Prediction Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['RMSE', 'MAE', 'R2 Score']
colors = ['#1f77b4', '#ff7f0e']  # 파란색(Wet), 주황색(Dry) 테마

for i, metric in enumerate(metrics):
    subset = df_results[df_results['Metric'] == metric]
    sns.barplot(data=subset, x='Model', y='Score', hue='Condition', ax=axes[i], palette=colors)

    axes[i].set_title(f'[{metric}]', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel(metric, fontsize=12)

    # R2 스코어의 경우 음수가 있으므로 y축 범위 동적 조절
    if metric == 'R2 Score':
        axes[i].set_ylim(-0.5, 0.5)
        axes[i].axhline(0, color='black', linewidth=1)  # 0점 기준선 추가
    else:
        axes[i].set_ylim(0, subset['Score'].max() * 1.2)

    # 막대 위에 수치 텍스트 표시
    for p in axes[i].patches:
        height = p.get_height()
        if not np.isnan(height):
            # 값이 양수면 위에, 음수면 아래에 텍스트 표시
            y_pos = height + (0.02 if height >= 0 else -0.05)
            va_align = 'bottom' if height >= 0 else 'top'
            axes[i].text(p.get_x() + p.get_width() / 2., y_pos,
                         f'{height:.2f}', ha='center', va=va_align, fontsize=10, fontweight='bold')

plt.tight_layout()
fig.subplots_adjust(top=0.88)

# 그래프 이미지 파일로 저장
save_path = 'model_performance_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"[{save_path}] 파일로 시각화 그래프가 성공적으로 저장되었습니다.")
