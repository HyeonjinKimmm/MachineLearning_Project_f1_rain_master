import pandas as pd
import numpy as np

print("="*70)
print("F1 데이터 전처리")
print("="*70)

# =====================================================================
# 1. 원본 데이터 로드
# =====================================================================
print("CSV 데이터 로드 중...")
races = pd.read_csv('championship/races.csv')
lap_times = pd.read_csv('championship/lap_times.csv')
results = pd.read_csv('championship/results.csv')
drivers = pd.read_csv('championship/drivers.csv')
constructors = pd.read_csv('championship/constructors.csv')
qualifying = pd.read_csv('championship/qualifying.csv')
pit_stops = pd.read_csv('championship/pit_stops.csv')
constructor_standings = pd.read_csv('championship/constructor_standings.csv')
weather = pd.read_csv('weather/F1 Weather(2023-2018).csv')

# =====================================================================
# 2. 날씨 데이터 정제
# =====================================================================
print("기상 데이터 정제 및 Timedelta 변환 중...")
# 문자열을 Datetime이 아닌 Timedelta(경과 시간)로 변환
weather['Time'] = pd.to_timedelta(weather['Time'], errors='coerce')
weather.dropna(subset=['Time'], inplace=True)

weather_clean = weather[['Time', 'Year', 'Round Number', 'Rainfall',
                         'TrackTemp', 'AirTemp', 'Humidity', 'Pressure', 'WindSpeed']].copy()
weather_clean.rename(columns={
    'Year': 'year',
    'Round Number': 'round',
    'Rainfall': 'Is_Wet'
}, inplace=True)
weather_clean['Is_Wet'] = weather_clean['Is_Wet'].astype(int)

# merge_asof를 위해 Time 기준으로 정렬
weather_clean = weather_clean.sort_values('Time')

# =====================================================================
# 3. 랩타임 누적 경과 시간 계산 및 기상 매핑
# =====================================================================
print("랩(Lap) 누적 경과 시간(Elapsed Time) 계산 및 기상 매핑 중...")
recent_races = races[(races['year'] >= 2018) & (races['year'] <= 2023)].copy()
df = pd.merge(lap_times, recent_races[['raceId', 'year', 'round', 'circuitId']], on='raceId', how='inner')

# 드라이버별로 랩타임(밀리초)을 누적 합산(cumsum)하여 레이스 경과 시간을 계산
df = df.sort_values(['raceId', 'driverId', 'lap'])
df['Elapsed_Time'] = pd.to_timedelta(df.groupby(['raceId', 'driverId'])['milliseconds'].cumsum(), unit='ms')

# merge_asof는 두 데이터 모두 기준 키로 정렬되어 있어야 함
df = df.dropna(subset=['Elapsed_Time']).sort_values('Elapsed_Time')

# 연도(year)와 라운드(round)가 같은 그룹 안에서 가장 가까운 경과 시간(backward) 매핑
df = pd.merge_asof(
    df,
    weather_clean,
    left_on='Elapsed_Time',
    right_on='Time',
    by=['year', 'round'],
    direction='backward'
)

# =====================================================================
# 4. 도메인 지식 테이블 병합
# =====================================================================
print("드라이버/팀/퀄리파잉 정보 병합 및 피트스탑 제어 중...")
df = pd.merge(df, results[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder']], on=['raceId', 'driverId'], how='left')
df = pd.merge(df, drivers[['driverId', 'driverRef']], on='driverId', how='left')
df = pd.merge(df, constructors[['constructorId', 'name']], on='constructorId', how='left')
df.rename(columns={'name': 'Team'}, inplace=True)

quali_sub = qualifying[['raceId', 'driverId', 'position']].rename(columns={'position': 'Quali_Pos'})
df = pd.merge(df, quali_sub, on=['raceId', 'driverId'], how='left')

cs_sub = constructor_standings[['raceId', 'constructorId', 'points']].rename(columns={'points': 'Constructor_Points'})
df = pd.merge(df, cs_sub, on=['raceId', 'constructorId'], how='left')
df['Constructor_Points'] = df['Constructor_Points'].fillna(0)

pit_sub = pit_stops[['raceId', 'driverId', 'lap', 'stop']].rename(columns={'stop': 'Pit_Stop_Count'})
df = pd.merge(df, pit_sub, on=['raceId', 'driverId', 'lap'], how='left')

# 피트스탑 전방 채우기
df = df.sort_values(by=['raceId', 'driverId', 'lap'])
df['Pit_Stop_Count'] = df.groupby(['raceId', 'driverId'])['Pit_Stop_Count'].ffill().fillna(0)

# =====================================================================
# 5. 파생 변수 생성
# =====================================================================
print("파생 변수 생성 중...")
race_max_lap = df.groupby('raceId')['lap'].transform('max')
driver_max_lap = df.groupby(['raceId', 'driverId'])['lap'].transform('max')
df['Is_Survived'] = np.where(driver_max_lap >= race_max_lap * 0.9, 1, 0)

df['is_2022_regulation'] = np.where(df['year'] >= 2022, 1, 0)
street_circuits = [6, 14, 73, 77, 79, 80]
df['is_street_circuit'] = np.where(df['circuitId'].isin(street_circuits), 1, 0)
df['Fuel_Burn_Ratio'] = df['lap'] / race_max_lap
df['Clean_Air'] = np.where(df['position'] <= 5, 1, 0)

# =====================================================================
# 6. 아웃라이어 필터링 및 타겟 스케일링
# =====================================================================
print("이상치 제거 및 Target 생성 중...")
df['Lap_Time_sec'] = df['milliseconds'] / 1000.0

# 불필요한 임시 시간 변수 삭제
drop_temp_cols = ['time', 'milliseconds', 'Time', 'Elapsed_Time']
df.drop(columns=[col for col in drop_temp_cols if col in df.columns], inplace=True, errors='ignore')
df.dropna(inplace=True)

# 맑은 날/비 온 날 분리하여 115% 이상치 컷오프
median_lap_times = df.groupby(['raceId', 'Is_Wet'])['Lap_Time_sec'].transform('median')
df_cleaned = df[df['Lap_Time_sec'] <= median_lap_times * 1.15].copy()

# 서킷 단위 타겟 스케일링
dry_baseline = df_cleaned[df_cleaned['Is_Wet'] == 0].groupby('circuitId')['Lap_Time_sec'].median().reset_index()
dry_baseline.rename(columns={'Lap_Time_sec': 'Circuit_Dry_Median'}, inplace=True)
df_cleaned = pd.merge(df_cleaned, dry_baseline, on='circuitId', how='left')

df_cleaned['Circuit_Dry_Median'] = df_cleaned['Circuit_Dry_Median'].fillna(df_cleaned['Lap_Time_sec'].median())
df_cleaned['Target_Pace_Ratio'] = df_cleaned['Lap_Time_sec'] / df_cleaned['Circuit_Dry_Median']

# =====================================================================
# 7. 최종 데이터셋 저장
# =====================================================================
print("최종 데이터셋 CSV 저장 중...")
output_filename = 'f1_dataset.csv'
df_cleaned.to_csv(output_filename, index=False)

print("\n" + "="*70)
print(f"[완료] 전처리 파일 저장: '{output_filename}' (Shape: {df_cleaned.shape})")
print("="*70)
