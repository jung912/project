import pandas as pd
import pingouin as pg
import numpy as np
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


plt.rcParams['font.family'] = 'Malgun Gothic'  
plt.rcParams['axes.unicode_minus'] = False  

csv_path = 'outlier_removed.csv'    

df = pd.read_csv(
    csv_path,
    header=0,        
    index_col='id',  
    encoding='utf-8-sig')
df


import time
import json
import requests
import geopandas as gpd
from shapely.geometry import Point
# ─────────────────────────────────────────────────────────────────────────────
# 0) 원본 df, poi_tags, 그리고 bbox 계산
  # latitude, longitude 칼럼이 있어야 함
poi_tags = {
    'transport': {
        'amenity': ['bus_station','taxi'],
        'railway': ['station']
    },
    'infrastructure': {
        'amenity': ['police','hospital','pharmacy','restaurant','supermarket']
    },
    'tourism': {
        'tourism': ['viewpoint','museum','attraction'],
        'leisure': ['park']
    }
}
pad = 0.01
minx, maxx = df.longitude.min()-pad, df.longitude.max()+pad
miny, maxy = df.latitude.min()-pad, df.latitude.max()+pad
# ─────────────────────────────────────────────────────────────────────────────
# 1) 한 번에 bbox 내 모든 POI 내려받기 (Overpass bbox 쿼리)
OVERPASS_URL = "http://overpass-api.de/api/interpreter"
# build filters for bbox query
filters = ""
for grp in poi_tags.values():
    for key, vals in grp.items():
        for v in vals:
            filters += f'node["{key}"="{v}"]({miny},{minx},{maxy},{maxx});\n'
# full query
query = f"""
[out:json][timeout:180];
(
{filters}
);
out body;
"""
resp = requests.post(OVERPASS_URL, data={'data': query}, timeout=(5,300))
resp.raise_for_status()
data = resp.json().get('elements', [])
# ─────────────────────────────────────────────────────────────────────────────
# 2) GeoDataFrame 생성
pois = pd.DataFrame([
    {
      'lon': el['lon'],
      'lat': el['lat'],
      **el.get('tags',{})
    }
    for el in data
    if el['type']=='node' and 'lon' in el
])
gdf_pois = gpd.GeoDataFrame(
    pois,
    geometry=gpd.points_from_xy(pois.lon, pois.lat),
    crs="EPSG:4326"
).to_crs(epsg=3857)
# 원본 좌표도 GeoDataFrame
gdf_pts = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
).to_crs(epsg=3857)
sindex = gdf_pois.sindex
# ─────────────────────────────────────────────────────────────────────────────
# 3) 그룹별 카운트 함수
def count_group(pt, grp_map, radius=1000):
    buf = pt.buffer(radius)
    candidates = gdf_pois.iloc[list(sindex.intersection(buf.bounds))]
    cnt = 0
    for key, vals in grp_map.items():
        cnt += candidates[candidates[key].isin(vals)].shape[0]
    return cnt
# 4) 각 포인트별 count, df에 붙이기
for grp, tags in poi_tags.items():
    df[f"{grp}_count"] = [
        count_group(pt, tags, radius=1000)
        for pt in gdf_pts.geometry
    ]
# 5) 결과 확인
print(df[['transport_count','infrastructure_count','tourism_count']].head())


from sklearn.decomposition import PCA
poi_cols = ['transport_count','infrastructure_count','tourism_count']
pca = PCA(n_components=1)
# PCA fit → PC1 점수 생성
df['poi_pca1'] = pca.fit_transform(df[poi_cols].fillna(0))
# 설명 분산 비율 확인 (얼마나 데이터의 변동성을 담았는지)
print("Explained variance ratio (PC1):", pca.explained_variance_ratio_[0])
#poi_pca1 <0 poi 희박 지역, poi_pca1 > 0 poi 밀집지역

'''
room_new_type 기준으로 필수 amenity와 필요 amenity 갖추고 있는 지수(점수로 표현)

공통 amenity (필수):
['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']

high 특화 amenity:
['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo']

low-mid 특화 amenity:
['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking', 'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave']

mid 특화 amenity:
['Cooking basics', 'Kitchen', 'Oven']

upper-mid 특화 amenity:
['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']
'''


import ast

# 기준 Amenity 딕셔너리 정의
common_amenities = ['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']

type_amenity_dict = {
    'high': ['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo'],
    'low-mid': ['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking', 
                'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave'],
    'mid': ['Cooking basics', 'Kitchen', 'Oven'],
    'upper-mid': ['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']}

# amenities 문자열 → 리스트로 파싱
def parse_amenities(row):
    try:
        return ast.literal_eval(row)
    except:
        return []

df['parsed_amenities'] = df['amenities'].apply(parse_amenities)

# amenity 매칭 점수 계산 함수
def calc_match_score(row):
    amenities = row['parsed_amenities']
    room_type = row['room_new_type']  
    
    # 공통 어매니티 일치 비율
    common_match = sum(1 for a in amenities if a in common_amenities) / len(common_amenities)
    
    # room type 별 특화 어매니티 일치 비율
    type_amenities = type_amenity_dict.get(room_type, [])
    if type_amenities:
        type_match = sum(1 for a in amenities if a in type_amenities) / len(type_amenities)
    else:
        type_match = 0.0
    
    return pd.Series({
        'common_amenity_score': round(common_match, 3),
        'type_amenity_score': round(type_match, 3)})

# 점수 컬럼 추가
df[['common_amenity_score', 'type_amenity_score']] = df.apply(calc_match_score, axis=1)

# 점수 해석을 위한 요약 출력
print(df[['room_new_type', 'common_amenity_score', 'type_amenity_score']].groupby('room_new_type').mean().round(3))




# 위치데이터 카운트 변수들 정규성/등분산성

from scipy.stats import shapiro, levene

Location = ['transport_count', 'infrastructure_count', 'tourism_count', 'poi_pca1']
TARGET = 'host_is_superhost'

for col in Location:
    print(f"\n 변수: {col}")

    # 정규성 검정 (랜덤 샘플링)
    group1 = df[df[TARGET]==1][col].dropna()
    group0 = df[df[TARGET]==0][col].dropna()
    
    n1 = min(5000, len(group1))
    n0 = min(5000, len(group0))

    stat1, p1 = shapiro(group1.sample(n1, random_state=42))
    stat0, p0 = shapiro(group0.sample(n0, random_state=42))

    print(f"정규성 p값 (group1): {p1:.4f}, (group0): {p0:.4f}")

    # 등분산성 검정
    stat, p = levene(group1, group0)
    print(f"등분산성 p값: {p:.4f}")

    # 장소 변수별 비모수 검정 
from scipy.stats import mannwhitneyu

for col in Location:
    group1 = df[df[TARGET]==1][col].dropna()
    group0 = df[df[TARGET]==0][col].dropna()

    stat, p = mannwhitneyu(group1, group0, alternative='two-sided')
    print(f"{col} - Mann-Whitney U p값: {p:.4f}")


    
#수치형 변수/ 이진형/ 범주형 각각 t검정, 비모수검정, 카이제곱 검정 

from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
import pingouin as pg   # 카이-제곱용

TARGET = 'host_is_superhost'

# 수치형 변수 리스트 (위도·경도·식별자 제외)
raw_num = [c for c in df.select_dtypes(include=['int64','float64']).columns
           if c not in ['latitude','longitude','host_id','id','host_is_superhost','Unnamed: 0']]

# 이진 수치형(0/1)만 골라내기
binary_num = [c for c in raw_num if df[c].dropna().isin([0,1]).all()]
continuous_num = [c for c in raw_num if c not in binary_num]

# 범주형 변수
cat_cols = df.select_dtypes(include=['object','category']).columns

results = []

# 연속형: 정규성 → t vs Mann-Whitney
def check_normality(series):
    return shapiro(series.dropna())[1] >= 0.05

for col in continuous_num:
    super = df[df[TARGET]==1][col].dropna()
    non   = df[df[TARGET]==0][col].dropna()
    
    if check_normality(super) and check_normality(non):
        stat, p = ttest_ind(super, non, equal_var=False)
        test = 't-test'
    else:
        stat, p = mannwhitneyu(super, non, alternative='two-sided')
        test = 'Mann-Whitney U'
    
    results.append({'variable':col, 'test':test, 'p':round(p,4)})

# 이진 수치형 & 범주형 → 카이제곱
for col in binary_num + cat_cols.tolist():
    ct = pd.crosstab(df[col], df[TARGET])
    chi2, p, _, _ = chi2_contingency(ct)
    results.append({'variable':col, 'test':'chi2', 'p':round(p,4)})

# 결과 정리
stat_df = pd.DataFrame(results).sort_values('p')
stat_df


# 연관성탐색 모델링 랜덤포레스트
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# 제외할 컬럼
exclude_cols = ['host_is_superhost', 'amenities', 'host_id', 'longitude', 'latitude','parsed_amenities']

# 설명 변수 설정 (원본 df에서 제외 컬럼 제외)
cols = [c for c in df.columns if c not in exclude_cols]
X = df[cols]

# 원핫인코딩 (범주형 변수 처리)
X = pd.get_dummies(X, drop_first=True)

# 타겟 변수
y = df['host_is_superhost']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 랜덤포레스트 모델 학습
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

# 예측
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# 평가 결과 출력
print("\n=== 테스트셋 평가 결과 ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("AUC:", round(roc_auc_score(y_test, y_proba), 3))

# 변수 중요도 출력
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\n=== 변수 중요도 ===")
print(importances.sort_values(ascending=False).round(3))



# 슈퍼호스트여부 판별 예측 모델링 

# 1. 목표 변수 설정
TARGET = 'host_is_superhost'
y = df[TARGET].astype(int)

# === 변수 목록 정의 ===

strategy_cols = ['amenities_cnt', 'availability_365', 'price', 'host_about_length_group', 'room_type','name_length_group', 'description_length_group',
                 'host_has_profile_pic', 'host_response_time_score','type_amenity_score','common_amenity_score',
                 'host_acceptance_rate_score', 'host_identity_verified','is_long_term', 'accommodates']

# === 데이터셋 준비 ===
X = df[strategy_cols]
y = df['host_is_superhost'].astype(int)

# 원핫 인코딩
X_encoded = pd.get_dummies(X, drop_first=False)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# === 랜덤포레스트 모델 정의 ===
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=15,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced')

# 모델 학습
rf.fit(X_train, y_train)

# 예측
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# === 평가 지표 출력 ===
print("\n=== 랜덤포레스트 전략모델 성능 평가 ===")
print(classification_report(y_test, y_pred))
print("AUC:", round(roc_auc_score(y_test, y_proba), 4))

# === 변수 중요도 출력 ===
importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
print("\n=== 변수 중요도 ===")
print(importances.sort_values(ascending=False).round(3))


# 수치형데이터 중앙값 평균값 비교 
continuous_cols = [
    c for c in df.select_dtypes(include=['int64', 'float64']).columns
    if c not in [
        'host_is_superhost', # 종속변수
        'latitude', 'longitude', 'host_id', 'id', 'Unnamed: 0',
        # 이진 0/1 변수들 추가
        'host_identity_verified', 'host_location_boolean', 'host_location_ny',
        'neighborhood_overview_exists', 'is_long_term', 'instant_bookable',
        'is_activate', 'host_has_profile_pic','accommodates']]

# 중앙값 테이블
median_table = pd.DataFrame({
    'variable': continuous_cols,
    'superhost_median': [df[df['host_is_superhost'] == 1][col].median() for col in continuous_cols],
    'non_superhost_median': [df[df['host_is_superhost'] == 0][col].median() for col in continuous_cols]})

# 평균값 테이블 
avg_table = pd.DataFrame({
    'variable': continuous_cols,
    'superhost_avg': [df[df['host_is_superhost'] == 1][col].mean().round(2) for col in continuous_cols],
    'non_superhost_avg': [df[df['host_is_superhost'] == 0][col].mean().round(2) for col in continuous_cols]})

# 평균 + 중앙값 테이블 합치기
merged_table = pd.merge(avg_table,median_table,on='variable')

# 차이 컬럼 추가
merged_table['mean_diff'] = (merged_table['superhost_avg'] - merged_table['non_superhost_avg']).round(2)
merged_table['median_diff'] = (merged_table['superhost_median'] - merged_table['non_superhost_median']).round(2)

# 차이 기준 정렬 
merged_table.sort_values('mean_diff', ascending=False)


# 수치형 데이터(이진제외)시각화
import seaborn as sns
import matplotlib.pyplot as plt

continuous_cols = [
    'amenities_cnt', 'availability_365', 'price', 'log_price',
    'accommodates', 'host_acceptance_rate_score', 'host_response_time_score'
]

plt.figure(figsize=(14, 10))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='host_is_superhost', y=col, data=df)
    plt.title(col)
    plt.xticks([0, 1], ['Not', 'Super'])
plt.tight_layout()
plt.show()


# 이진수치형 데이터 변수
bin_vars = ['host_location_boolean', 'host_location_ny',
        'neighborhood_overview_exists', 'is_long_term', 'instant_bookable',
        'is_activate', 'host_has_profile_pic'] 

bin_table = pd.DataFrame({'variable': bin_vars,
    'superhost_1(%)': [df[df['host_is_superhost']==1][col].mean().round(2)*100 for col in bin_vars],
    'superhost_0(%)': [(1-df[df['host_is_superhost']==1][col]).mean().round(2)*100 for col in bin_vars],
    'non_superhost_1(%)': [df[df['host_is_superhost']==0][col].mean().round(2)*100 for col in bin_vars],
    'non_superhost_0(%)': [(1-df[df['host_is_superhost']==0][col]).mean().round(2)*100 for col in bin_vars],
    'diff_1(%)': [df[df['host_is_superhost']==1][col].mean().round(2)*100 -
                 df[df['host_is_superhost']==0][col].mean().round(2)*100 for col in bin_vars]})

bin_table = bin_table.sort_values('diff_1(%)', ascending=False)
bin_table



# 범주형 변수 room_new_type
cat_var_room_new_type = 'room_new_type'
ct = pd.crosstab(df[cat_var_room_new_type], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

# 룸타입 
cat_var_room = 'room_type'
ct = pd.crosstab(df[cat_var_room], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)
# 비슈퍼호스트일때와 슈퍼호스트일때 룸타입별 비율 

cat_var_name = 'name_length_group'
ct = pd.crosstab(df[cat_var_name], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_description = 'description_length_group'
ct = pd.crosstab(df[cat_var_description], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_hostabout = 'host_about_length_group'
ct = pd.crosstab(df[cat_var_hostabout], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_structure = 'room_structure_type'
ct = pd.crosstab(df[cat_var_structure], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_neighbourhood = 'neighbourhood_cleansed'
ct = pd.crosstab(df[cat_var_neighbourhood], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_group = 'neighbourhood_group_cleansed'
ct = pd.crosstab(df[cat_var_group], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

# 범주형/ 이진형 데이터 시각화 
cat_cols = [
    'is_long_term', 'instant_bookable', 'neighborhood_overview_exists',
    'neighbourhood_group_cleansed', 'host_identity_verified',
    'room_type', 'host_has_profile_pic', 'room_new_type']

n_cols = 4
n_rows = (len(cat_cols) + n_cols - 1) // n_cols
plt.figure(figsize=(16, 3 * n_rows))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(x=col, hue='host_is_superhost', data=df)
    plt.title(col)
    plt.legend(title=None, labels=['Not', 'Super'])
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Location 중앙값, 평균차

# 중앙값 테이블
Location_median = pd.DataFrame({
    'variable': Location,
    'superhost_median': [df[df['host_is_superhost'] == 1][col].median() for col in Location],
    'non_superhost_median': [df[df['host_is_superhost'] == 0][col].median() for col in Location]})

# 평균값 테이블 
Location_median_avg = pd.DataFrame({
    'variable': Location,
    'superhost_avg': [df[df['host_is_superhost'] == 1][col].mean().round(2) for col in Location],
    'non_superhost_avg': [df[df['host_is_superhost'] == 0][col].mean().round(2) for col in Location]})
 
# 평균 + 중앙값 테이블 합치기
Location_merged = pd.merge(Location_median_avg,Location_median,on='variable')

# 차이 컬럼 추가
Location_merged['mean_diff'] = (Location_merged['superhost_avg'] - Location_merged['non_superhost_avg']).round(2)
Location_merged['median_diff'] = (Location_merged['superhost_median'] - Location_merged['non_superhost_median']).round(2)

# 차이 기준 정렬
Location_merged.sort_values('mean_diff', ascending=False)

# location 과 review_scores_rating
# median 기준 High/Low로 나눔
location_vars = ['transport_count', 'infrastructure_count', 'tourism_count', 'poi_pca1']
review_median = df['review_scores_rating'].median()

for loc_var in location_vars:
    loc_median = df[loc_var].median()
    
    loc_level = df[loc_var].apply(lambda x: 'Low' if x < loc_median else 'High')
    review_level = df['review_scores_rating'].apply(lambda x: 'Low' if x < review_median else 'High')
    
    group = loc_level + f' {loc_var} & ' + review_level + ' Review'
    
    group_summary = pd.crosstab(
        group,
        df['host_is_superhost'],
        normalize='index'
    ).round(3)
    
    group_summary.columns = ['Not Superhost', 'Superhost']
    group_summary = group_summary.sort_values(by='Superhost', ascending=False)
    
    print(f'\n Superhost Ratio for Location Variable: **{loc_var}**')
    print(group_summary)

# location과 amenities_cnt
location_vars = ['transport_count', 'infrastructure_count', 'tourism_count', 'poi_pca1']
nities_median = df['amenities_cnt'].median()

for loc_var in location_vars:
    loc_median = df[loc_var].median()
    
    loc_level = df[loc_var].apply(lambda x: 'Low' if x < loc_median else 'High')
    amenities_level = df['amenities_cnt'].apply(lambda x: 'Low' if x < nities_median else 'High')
    
    group = loc_level + f' {loc_var} & ' + amenities_level + ' amenities_cnt'
    
    group_summary = pd.crosstab(
        group,
        df['host_is_superhost'],
        normalize='index'
    ).round(3)
    
    group_summary.columns = ['Not Superhost', 'Superhost']
    group_summary = group_summary.sort_values(by='Superhost', ascending=False)
    
    print(f'\n Superhost Ratio for Location Variable: **{loc_var}** (grouped by amenities_cnt)')
    print(group_summary)

# 박스플롯 & 커널밀도 히스토그램 (각 변수별 슈퍼호스트 유무 분포)
import seaborn as sns
import matplotlib.pyplot as plt
location_vars = ['transport_count', 'infrastructure_count', 'tourism_count','poi_pca1']

for var in location_vars:
    plt.figure(figsize=(12, 5))

    # 박스플롯
    plt.subplot(1, 2, 1)
    sns.boxplot(x='host_is_superhost', y=var, data=df)
    plt.title(f'{var} by Superhost (Boxplot)')
    plt.xlabel('Superhost')
    plt.ylabel(var)

    # 히스토그램 + 커널밀도
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df[df['host_is_superhost'] == 1], x=var, label='Superhost', fill=True)
    sns.kdeplot(data=df[df['host_is_superhost'] == 0], x=var, label='Not Superhost', fill=True)
    plt.title(f'{var} Distribution by Superhost (KDE)')
    plt.xlabel(var)
    plt.legend()

    plt.tight_layout()
    plt.show()


# 슈퍼호스트 여부 예측 모델링 로지스틱/랜덤포레스트 앙상블 인사이트용
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 변수 리스트 정의 (신뢰 변수는 별도 처리) VIF가 높은 host변수들 trust_cols로 그룹지어 사용 
strategy_cols = [
    'amenities_cnt', 'availability_365', 'price',
    'instant_bookable', 'host_about_length_group', 'room_type',
    'neighbourhood_group_cleansed', 'name_length_group', 'description_length_group',
    'is_long_term', 'accommodates']

trust_cols = ['host_has_profile_pic', 'host_response_time_score', 'host_acceptance_rate_score', 'host_identity_verified']

# 2. 모델링용 데이터 복사
df_model = df.copy()

# 3. 신뢰 변수들 결측치는 0으로 채우고 host_trust_score 생성 (평균)
df_model[trust_cols] = df_model[trust_cols].fillna(0)
df_model['host_trust_score'] = df_model[trust_cols].mean(axis=1)

# 4. 모델링에 사용할 변수 리스트에 host_trust_score 추가
model_cols = strategy_cols + ['host_trust_score']

# 5. 설명 변수 준비 (원래 신뢰 변수는 제외)
X_raw = df_model[model_cols].copy()

# 6. 리스트형 변수 있으면 제거 (예외 처리)
for col in X_raw.columns:
    if X_raw[col].apply(lambda x: isinstance(x, list)).any():
        print(f"[제거] 리스트형 컬럼: {col}")
        X_raw.drop(columns=[col], inplace=True)

# 7. 원핫 인코딩
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# 8. 결측치 확인 및 제거 (NaN 0으로 대체)
print("결측치 합계:\n", X_encoded.isnull().sum()[X_encoded.isnull().sum() > 0])
X_encoded = X_encoded.fillna(0)

# 9. 목표 변수 설정
y = df_model['host_is_superhost']

# 10. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 11. 모델 정의 및 학습
log_reg = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')

ensemble = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf)], voting='soft')
ensemble.fit(X_train, y_train)

# 12. 예측 및 평가
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print("\n=== 소프트 보팅 앙상블 평가 결과 ===")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# 13. 로지스틱 회귀 계수 분석
log_reg.fit(X_train, y_train)
coeff_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\n=== 로지스틱 회귀 계수 상위 변수 ===")
print(coeff_df.round(3).head(10))

print("\n=== 로지스틱 회귀 계수 하위 변수 ===")
print(coeff_df.round(3).tail(15))




# 예측 돌리기 (랜덤포레스트)

# 1. host_response_time → 점수 변환 함수
def response_time_to_score(response_time_str):
    mapping = {
        'within an hour': 4,
        'within a few hours': 3,
        'within a day': 2,
        'a few days or more': 1
    }
    return mapping.get(response_time_str.lower(), 0)  # 기본 0점

# 2. host_response_rate(0~100) → 점수 변환 함수
def response_rate_to_score(rate_percent):
    rate = rate_percent / 100
    if rate <= 0.25:
        return 1
    elif rate <= 0.5:
        return 2
    elif rate <= 0.75:
        return 3
    else:
        return 4

# 3. host_acceptance_rate(0~100) → 점수 변환 함수
def acceptance_rate_to_score(rate_percent):
    rate = rate_percent / 100
    if rate <= 0.25:
        return 1
    elif rate <= 0.5:
        return 2
    elif rate <= 0.75:
        return 3
    else:
        return 4

# 4. amenities 점수 계산 함수
common_amenities = ['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']

type_amenity_dict = {
    'high': ['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo'],
    'low-mid': ['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking', 
                'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave'],
    'mid': ['Cooking basics', 'Kitchen', 'Oven'],
    'upper-mid': ['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']
}

def calc_amenity_scores(amenities_list, room_new_type):
    # 공통 amenity 점수
    common_match = sum(1 for a in amenities_list if a in common_amenities) / len(common_amenities) if common_amenities else 0

    # 타입별 amenity 점수
    type_amenities = type_amenity_dict.get(room_new_type, [])
    type_match = sum(1 for a in amenities_list if a in type_amenities) / len(type_amenities) if type_amenities else 0

    return round(common_match, 3), round(type_match, 3)

# 점수변환 예시 입력값
# 사용자 입력 예시(입력값만 변경)
user_input = {
    'host_response_time': 'within an hour',
    'host_response_rate': 85,  # %
    'host_acceptance_rate': 78,  # %
    'amenities': ['Wifi', 'Essentials', 'Hangers', 'Oven', 'Kitchen'],
    'room_new_type': 'mid'
}

# 점수 계산
response_time_score = response_time_to_score(user_input['host_response_time'])
response_rate_score = response_rate_to_score(user_input['host_response_rate'])
acceptance_rate_score = acceptance_rate_to_score(user_input['host_acceptance_rate'])
common_amenity_score, type_amenity_score = calc_amenity_scores(user_input['amenities'], user_input['room_new_type'])

# 결과 출력
print(f"host_response_time_score: {response_time_score}")
print(f"host_response_rate_score: {response_rate_score}")
print(f"host_acceptance_rate_score: {acceptance_rate_score}")
print(f"common_amenity_score: {common_amenity_score}")
print(f"type_amenity_score: {type_amenity_score}")



# 점수계산 함수값 이용 
# 새로운 데이터 예시 (입력값만 변경)
new_data = pd.DataFrame([{
    'amenities_cnt': 12,
    'availability_365': 200,
    'price': 150,
    'host_about_length_group': 'medium',  # 범주형
    'room_type': 'Entire home/apt',       # 범주형
    'name_length_group': 'short',         # 범주형
    'description_length_group': 'long',   # 범주형
    'host_has_profile_pic': 1,
    'host_response_time_score': 0.9,
    'type_amenity_score': 0.7,
    'common_amenity_score': 0.6,
    'host_acceptance_rate_score': 0.95,
    'host_identity_verified': 1,
    'is_long_term': 0,
    'accommodates': 3
}])

# 모델 학습 때 썼던 컬럼명 저장
train_columns = X_encoded.columns

# 입력 데이터 전처리 함수
def preprocess_input(new_df, train_cols):
    new_encoded = pd.get_dummies(new_df, drop_first=False)
    # 학습 시 없던 컬럼 채우기 (0으로)
    missing_cols = set(train_cols) - set(new_encoded.columns)
    for c in missing_cols:
        new_encoded[c] = 0
    # 순서 맞추기
    new_encoded = new_encoded[train_cols]
    return new_encoded

# 전처리
X_new = preprocess_input(new_data, train_columns)

# 예측
pred = rf.predict(X_new)
proba = rf.predict_proba(X_new)[:, 1]

print("예측 결과 (슈퍼호스트 여부):", pred[0])  # 1이면 슈퍼호스트, 0이면 아님
print("슈퍼호스트 확률:", round(proba[0], 3))




# 슈퍼호스트 변수 그룹별 평균비교 

# 슈퍼호스트만 필터링
df_super = df[df['host_is_superhost'] == 1].copy()

# 범주형 변수 숫자 인코딩
cat_cols = ['host_about_length_group', 'name_length_group', 'description_length_group']
for col in cat_cols:
    if df_super[col].dtype == 'object':
        le = LabelEncoder()
        df_super[col] = le.fit_transform(df_super[col].astype(str))

def plot_radar_chart(df, group_col, value_cols, title):
    group_means = df.groupby(group_col)[value_cols].mean()
    group_means_norm = (group_means - group_means.min()) / (group_means.max() - group_means.min())
    
    labels = value_cols
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.Set2.colors
    
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    for i, (idx, row) in enumerate(group_means_norm.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=idx, color=colors[i % len(colors)], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

# 그룹

facility_info_cols = ['amenities_cnt', 'common_amenity_score', 'type_amenity_score', 'accommodates', 'neighborhood_overview_exists', 'price']
plot_radar_chart(df_super, 'room_type', facility_info_cols, 'Room Type별 시설/정보 그룹 평균 비교 (Superhost)')

facility_info_cols = ['amenities_cnt', 'common_amenity_score', 'type_amenity_score', 'accommodates', 'neighborhood_overview_exists', 'price']
plot_radar_chart(df_super, 'room_new_type', facility_info_cols, 'Room_new_type별 시설/정보 그룹 평균 비교 (Superhost)')


value_cols = ['amenities_cnt','common_amenity_score','type_amenity_score','accommodates','neighborhood_overview_exists','price']
plot_radar_chart(df_super, 'neighbourhood_group_cleansed', value_cols, '지역 그룹별 시설/정보/가격 평균 비교 (Superhost)')


host_profile_cols = ['host_has_profile_pic', 'host_identity_verified', 'host_about_length_group', 'name_length_group', 'description_length_group','neighborhood_overview_exists']
plot_radar_chart(df_super, 'neighbourhood_group_cleansed', host_profile_cols, '지역 그룹별 호스트 프로필 그룹 평균 비교 (Superhost)')
