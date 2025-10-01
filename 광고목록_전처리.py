import pandas as pd
import numpy as np

df = pd.read_csv('IVE_광고목록.csv', encoding='utf-8', low_memory=False)

df = df.drop(['ads_search','ads_icon_img','ads_summary','ads_guide','delyn','adv_idx','sch_idx','ads_limit','ads_sex_type','ads_day_cap','aff_idx','ads_payment','ads_package','ads_order','regdate','ads_age_min','ads_age_max','ads_require_adid'], axis=1)

# ads_category 에서 9번 카테고리 제거 
df = df[df["ads_category"] != 9]

# ads_sdate, ads_edate 컬럼 0000-00-00 00:00:00 제거 
df = df[df["ads_sdate"] != "0000-00-00 00:00:00"]
df = df[df["ads_edate"] != "0000-00-00 00:00:00"]

# ads_contract_price 컬럼 1~4값 제거
# 1~4 사이 값 삭제, 0과 5 이상은 남김
df = df[(df["ads_contract_price"] == 0) | (df["ads_contract_price"] > 4)]

# ads_reward_price 컬럼 1~4값 제거
# 1~4 사이 값 삭제, 0과 5 이상은 남김   
df = df[(df["ads_reward_price"] == 0) | (df["ads_reward_price"] > 4)]

# ads_save_way 결측치 "없음"으로 대체(참고용컬럼)
df["ads_save_way"] = df["ads_save_way"].fillna("없음")

# ads_sdate datetime 형식으로 변환
df["ads_sdate"] = pd.to_datetime(df["ads_sdate"], format="%Y-%m-%d %H:%M:%S", errors='coerce')

# ads_edate 형식 변환
# ads_edate 컬럼을 문자열로 변환 후 "9999"를 "2261"로 교체 후 datetime 형식으로 변환
df['ads_edate'] = df['ads_edate'].astype(str).str.replace("9999", "2261", regex=False)
df["ads_edate"] = pd.to_datetime(df["ads_edate"], format="%Y-%m-%d %H:%M:%S", errors='coerce')

# csv파일로 저장 
#df.to_csv("광고목록(전처리).csv", index=False, encoding="utf-8-sig")
