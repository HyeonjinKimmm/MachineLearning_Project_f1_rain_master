import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use('Agg') # GUI 충돌 에러 방지
import matplotlib.pyplot as plt
import seaborn as sns
import platform

print("=" * 95)
print("F1 우천 경기 - 40/20/40 랭킹 및 예측/실제 분석")
print("=" * 95)

# =====================================================================
# 1. 데이터 로드 및 셋업
# =====================================================================
df = pd.read_csv('f1_dataset.csv')
groups = df['raceId']

meta_data = df[['raceId', 'driverRef', 'Team', 'Is_Wet', 'Is_Survived', 'lap', 'Lap_Time_sec']].copy()
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)  # 전체 데이터 랭킹 복원용 스케일러

# =====================================================================
# 3. 스태킹 모델 (최적 파라미터 적용)
# =====================================================================
print("최적 파라미터 기반 Stacking 모델 학습 중")
best_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01,
                            subsample=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1)
best_hgb = HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.05,
                                         l2_regularization=10.0, random_state=42)

best_xgb.fit(X_train_scaled, y_train, sample_weight=weights_train)
best_hgb.fit(X_train_scaled, y_train, sample_weight=weights_train)

ultimate_model = StackingRegressor(
    estimators=[('xgb', best_xgb), ('hgb', best_hgb)],
    final_estimator=Ridge(alpha=1.0, random_state=42),
    cv="prefit"
)
ultimate_model.fit(X_train_scaled, y_train, sample_weight=weights_train)

# =====================================================================
# 4. Feature Importance (스태킹 앙상블)
# =====================================================================
print("\n스태킹 앙상블의 Feature Importance (Permutation 기반)")
wet_test_idx = np.where(df['Is_Wet'].iloc[test_idx] == 1)[0]
X_test_wet_scaled = X_test_scaled[wet_test_idx]
y_test_wet = y_test.iloc[wet_test_idx]

result = permutation_importance(ultimate_model, X_test_wet_scaled, y_test_wet,
                                scoring='r2', n_repeats=5, random_state=42)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Score': result.importances_mean
}).sort_values(by='Importance_Score', ascending=False)

for i, (idx, row) in enumerate(importance_df.head(5).iterrows()):
    print(f"  {i + 1}위. {row['Feature']:<20} (손실 스코어: {row['Importance_Score']:.4f})")

# =====================================================================
# 5. 모델 예측 vs 실제 기록 계산 및 지표 산출 (전체 데이터 기반)
# =====================================================================
print("\n스태킹 모델 기반 예측 랩타임 복원 및 상대 평가 진행 중...")
meta_data['Predicted_Ratio'] = ultimate_model.predict(X_all_scaled)
meta_data['Predicted_Lap_Time'] = meta_data['Predicted_Ratio'] * df['Circuit_Dry_Median']

# 오직 비가 온 데이터만 슬라이싱
wet_all_df = meta_data[meta_data['Is_Wet'] == 1].copy()

# 지표 1: 모델 예측 - 실제 랩타임
wet_all_df['Model_Advantage'] = wet_all_df['Predicted_Lap_Time'] - wet_all_df['Lap_Time_sec']

# 지표 2: 동일 레이스 소속 팀 평균 - 해당 선수 실제 랩타임
wet_all_df['Team_Race_Avg'] = wet_all_df.groupby(['raceId', 'Team'])['Lap_Time_sec'].transform('mean')
wet_all_df['Teammate_Advantage'] = wet_all_df['Team_Race_Avg'] - wet_all_df['Lap_Time_sec']

# 선수별 최종 스탯 집계
driver_stats = wet_all_df.groupby('driverRef').agg({
    'Predicted_Lap_Time': 'mean',  # 모델 예측 평균
    'Lap_Time_sec': 'mean',  # 실제 기록 평균
    'Model_Advantage': 'mean',  # 차이 (한계 돌파력)
    'Teammate_Advantage': 'mean',  # 소속 팀 동료와 비교
    'Is_Survived': 'mean',  # 생존율(리타이어하지 않은 비율)
    'lap': 'count'  # 빗길 주행 수
}).reset_index()

# 필터링: 통계적 신뢰성을 위해 커리어 통산 빗길 경험 50바퀴 이상인 선수만
valid_drivers = driver_stats[driver_stats['lap'] >= 50].copy()

# =====================================================================
# 6. 40/20/40 종합 점수 환산 및 정렬
# =====================================================================
min_max = MinMaxScaler((0, 100))
valid_drivers[['Score_LimitBreak', 'Score_TeamKill', 'Score_Survival']] = min_max.fit_transform(
    valid_drivers[['Model_Advantage', 'Teammate_Advantage', 'Is_Survived']]
)

# 최종 40 / 20 / 40 밸런스 공식 적용
valid_drivers['Final_Rain_Master_Score'] = (
        (valid_drivers['Score_LimitBreak'] * 0.40) +
        (valid_drivers['Score_TeamKill'] * 0.20) +
        (valid_drivers['Score_Survival'] * 0.40)
)

all_masters = valid_drivers.sort_values(by='Final_Rain_Master_Score', ascending=False)

# =====================================================================
# 7. 최종 결과 출력
# =====================================================================
print("\n" + "=" * 115)
print("모델 예측 기반 '전체 레인 마스터 랭킹 및 랩타임 세부 분석'")
print("=" * 115)
print(
    f"{'Rank':<4} | {'Driver':<15} | {'Total':<6} | {'Predict(Avg)':<12} | {'Actual(Avg)':<11} | {'Diff(Avg)':<9} | {'Break(40%)':<10} | {'Team(20%)':<10} | {'Surv(40%)'}")
print("-" * 115)

for rank, (_, row) in enumerate(all_masters.iterrows(), 1):
    diff_sec = row['Lap_Time_sec'] - row['Predicted_Lap_Time']
    diff_str = f"{diff_sec:+.3f}초"

    print(
        f" {rank:<3} | {row['driverRef']:<15} | {row['Final_Rain_Master_Score']:>5.1f}점 | {row['Predicted_Lap_Time']:>10.2f}초 | {row['Lap_Time_sec']:>9.2f}초 | {diff_str:>9} | {row['Score_LimitBreak']:>8.1f}점 | {row['Score_TeamKill']:>8.1f}점 | {row['Score_Survival']:>8.1f}점")
print("=" * 115)


# =====================================================================
# 8. 결과 시각화 (Feature Importance & 4-in-1 종합 비교)
# =====================================================================
print("\n" + "=" * 115)
print("피처 중요도 및 랭킹 통합 시각화 저장 중...")
print("=" * 115)

# OS별 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # Mac
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------
# [그래프 1] 순열 피처 중요도 (Top 7)
# -----------------------------------------------------------
plt.figure(figsize=(10, 6))
top_features = importance_df.head(7)

ax_feat = sns.barplot(
    data=top_features, x='Importance_Score', y='Feature', palette='viridis'
)
plt.title('우천 환경 랩타임 예측 주요 피처 중요도 (Top 7)', fontsize=16, fontweight='bold')
plt.xlabel('순열 피처 중요도', fontsize=12)
plt.ylabel('')

for p in ax_feat.patches:
    width = p.get_width()
    plt.text(width + 0.005, p.get_y() + p.get_height() / 2, f'{width:.4f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("[feature_importance.png] 피처 중요도 저장 완료.")
plt.close()

# -----------------------------------------------------------
# [그래프 2] 40/20/40 누적 막대 그래프 (한눈에 보는 종합 순위 & 구성 비율)
# -----------------------------------------------------------
plt.figure(figsize=(14, 10))
top_20 = valid_drivers.sort_values(by='Final_Rain_Master_Score', ascending=False).head(20).copy()

# 누적을 위해 각 지표에 가중치를 곱한 '실제 반영 점수' 계산
top_20['Break_40'] = top_20['Score_LimitBreak'] * 0.40
top_20['Team_20'] = top_20['Score_TeamKill'] * 0.20
top_20['Surv_40'] = top_20['Score_Survival'] * 0.40

y_pos = np.arange(len(top_20))
driver_names = top_20['driverRef']

# 3개의 막대를 겹쳐서(left 속성 활용) 하나의 막대로 합침
plt.barh(y_pos, top_20['Break_40'], color='#ff6f69', edgecolor='white', label='한계 돌파력 (40%)')
plt.barh(y_pos, top_20['Team_20'], left=top_20['Break_40'], color='#ffcc5c', edgecolor='white', label='팀 동료 우위 (20%)')
plt.barh(y_pos, top_20['Surv_40'], left=top_20['Break_40'] + top_20['Team_20'], color='#88d8b0', edgecolor='white', label='악천후 생존율 (40%)')

plt.yticks(y_pos, driver_names, fontsize=12)
plt.gca().invert_yaxis()  # 1위가 맨 위로 오도록 반전

plt.title('F1 레인 마스터 40/20/40 종합 점수 구성 분석 (Top 20)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('종합 점수 (Total Score, Max 100)', fontsize=13)
plt.xlim(0, 115)

# 막대 끝에 총점 및 최종 순위 텍스트 추가
for i, total in enumerate(top_20['Final_Rain_Master_Score']):
    plt.text(total + 1.5, i, f'{i+1}위 ({total:.1f}점)', va='center', fontsize=12, fontweight='bold')

plt.legend(loc='lower right', fontsize=12, title='평가 지표 (가중치)', title_fontsize=13)
plt.tight_layout()
plt.savefig('rain_master_stacked.png', dpi=300, bbox_inches='tight')
print("[rain_master_stacked.png] 누적 막대 그래프(종합 비교) 저장 완료.")
plt.close()

# -----------------------------------------------------------
# [그래프 3] 세부 지표 히트맵 (각 항목별 순수 강점 비교)
# -----------------------------------------------------------
plt.figure(figsize=(10, 8))

# 히트맵용 데이터프레임 구성 (가중치 곱하기 전 순수 100점 만점 기준 점수들)
heatmap_data = top_20.set_index('driverRef')[['Score_LimitBreak', 'Score_TeamKill', 'Score_Survival', 'Final_Rain_Master_Score']]
heatmap_data.columns = ['한계 돌파력', '팀 동료 우위', '악천후 생존율', '종합 점수']

# 컬러맵(cmap)으로 점수가 높을수록 진한 색상으로 표현
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': '환산 점수 (0~100)'})

plt.title('Top 20 드라이버 세부 지표 100점 환산 스탯 보드', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('')
plt.tight_layout()
plt.savefig('rain_master_heatmap.png', dpi=300, bbox_inches='tight')
print("[rain_master_heatmap.png] 세부 스탯 히트맵 저장 완료.")
plt.close()

print("=" * 115)
