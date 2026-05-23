import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


print("=" * 60)
print("Train / Test 데이터셋 구성 정밀 분석")
print("=" * 60)

# 1. 데이터 로드
df = pd.read_csv('f1_dataset.csv')
groups = df['raceId']

X = df.drop(columns=['Target_Pace_Ratio', 'Lap_Time_sec'], errors='ignore')
y = df['Target_Pace_Ratio'] if 'Target_Pace_Ratio' in df.columns else df['Is_Wet']

# 2. 데이터 분할
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]


# 3. 통계 계산 함수
def print_split_stats(name, data_subset):
    total = len(data_subset)
    wet_count = data_subset['Is_Wet'].sum()
    dry_count = total - wet_count

    print(f"{name} 데이터셋 (총 {total:,}건)")
    print(f"맑은 날 (Dry): {dry_count:,}건 ({dry_count / total * 100:.1f}%)")
    print(f"비 온 날 (Wet): {wet_count:,}건 ({wet_count / total * 100:.1f}%)\n")


# 4. 결과 출력
print(f"전체 원본 데이터: 총 {len(df):,}건\n")
print_split_stats("학습용 (Train)", train_df)
print_split_stats("평가용 (Test)", test_df)
print("=" * 60)
