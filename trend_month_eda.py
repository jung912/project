import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


df= pd.read_csv("목록+시간대_0915_1236.csv")

df['rpt_time_date'] = pd.to_datetime(df['rpt_time_date'])
df = df[
    (df['rpt_time_date'].dt.year == 2025) &
    (df['rpt_time_date'].dt.month.isin([7, 8]))
]
df.drop(columns=['mda_ads','rpt_time_time'], inplace=True)

ads_type_map = {
    1: "설치형", 2: "실행형", 3: "참여형", 4: "클릭형",
    5: "페북", 6: "트위터", 7: "인스타", 8: "노출형",
    9: "퀘스트", 10: "유튜브", 11: "네이버", 12: "CPS"
}
ads_category_map = {
    0: "선택안함", 1: "앱(간편적립)", 2: "경험하기-CPI/CPE",
    3: "구독(간편적립)", 4: "간편미션-퀴즈", 5: "경험하기-CPA",
    6: "멀티보상", 7: "금융", 8: "무료참여",
    10: "유료참여", 11: "쇼핑-상품", 12: "제휴몰", 13: "간편미션"
}
df["ads_type_nm"] = df["ads_type"].map(ads_type_map)
df["ads_category_nm"] = df["ads_category"].map(ads_category_map)

df = df[df["rpt_time_turn"] <= df["rpt_time_clk"]].copy()

df = df[~((df["rpt_time_turn"] == 0) & (df["rpt_time_clk"] == 0))]

df = df[df["rpt_time_acost"] != 0].copy()

df.groupby("ads_type_nm")["rpt_time_clk"].median()
# 타입별 클릭수 중앙값
type_mean = df.groupby("ads_type_nm")["rpt_time_clk"].transform("median")

# 자신의 타입 중앙값 보다 클릭수가 낮은 행 제거
df2 = df[df["rpt_time_clk"] >= type_mean].copy()


# 클릭수를 타입별 중앙값이상으로 제한
# 타입-카테고리-매체사별 전환율 클릭수를 타입별 중앙값이상으로 제한
def calc_CVR_rate(df2, group_cols):
    agg = (
        df2.groupby(group_cols, dropna=False)[["rpt_time_clk", "rpt_time_turn"]]
        .sum()
        .reset_index()
    )
    agg["CVR_rate"] = (agg['rpt_time_turn'] / agg['rpt_time_clk']).replace([np.inf, -np.inf], np.nan)
    agg["CVR"] = (agg["CVR_rate"] * 100).round(1)
    return agg
clkCVR= calc_CVR_rate(df2, ["ads_type_nm","ads_category_nm","mda_idx"]).sort_values("CVR", ascending=False)

N_total = 10  # 상위/하위 개수


# 필수 포함 상위 조합: clkCVR에서 직접 추출
required_combos = clkCVR[
    ((clkCVR["ads_type_nm"]=="인스타") & (clkCVR["ads_category_nm"]=="구독(간편적립)") & (clkCVR["mda_idx"]==1031)) |
    ((clkCVR["ads_type_nm"]=="실행형") & (clkCVR["ads_category_nm"]=="경험하기-CPI/CPE") & (clkCVR["mda_idx"]==795)) |
    ((clkCVR["ads_type_nm"]=="네이버") & (clkCVR["ads_category_nm"]=="구독(간편적립)") & (clkCVR["mda_idx"]==828))
].copy()


# merge용 키만 따로 준비
required_keys = required_combos[["ads_type_nm","ads_category_nm","mda_idx"]]


# 상위 10개 조합 선택
clkCVR_sorted = clkCVR.sort_values("CVR", ascending=False).reset_index(drop=True)

# 필수 조합 제외 후 나머지 상위 선택
top_remaining = clkCVR_sorted.merge(
    required_keys, 
    on=["ads_type_nm","ads_category_nm","mda_idx"], 
    how="left", indicator=True
)
top_remaining = top_remaining[top_remaining["_merge"] == "left_only"].drop(columns="_merge")

# 상위 N_total - 필수개수 만큼 추가
top_additional = top_remaining.head(N_total - len(required_keys))

# 최종 상위 10개 (필수 + 추가)
top_rows = pd.concat([required_combos, top_additional], axis=0).reset_index(drop=True)

# 상위 조합(타입+카테고리 단위, 중복 제거)
top_combos = top_rows[["ads_type_nm","ads_category_nm"]].drop_duplicates().reset_index(drop=True)

# 상위 조합에 속한 모든 매체사 상세
top_combo_mdas = clkCVR.merge(top_combos, on=["ads_type_nm","ads_category_nm"], how="inner")
top_combo_mdas = top_combo_mdas.sort_values(
    ["ads_type_nm","ads_category_nm","CVR"], ascending=[True, True, False]
).reset_index(drop=True)
top_combo_mdas["rank_in_combo"] = top_combo_mdas.groupby(
    ["ads_type_nm","ads_category_nm"]
)["CVR"].rank(method="dense", ascending=False).astype(int)



# 하위 10개 조합 선택
bottom_rows = clkCVR.sort_values("CVR", ascending=True).head(N_total).reset_index(drop=True)

bottom_combos = bottom_rows[["ads_type_nm","ads_category_nm"]].drop_duplicates().reset_index(drop=True)

bottom_combo_mdas = clkCVR.merge(bottom_combos, on=["ads_type_nm","ads_category_nm"], how="inner")
bottom_combo_mdas = bottom_combo_mdas.sort_values(
    ["ads_type_nm","ads_category_nm","CVR"], ascending=[True, True, False]
).reset_index(drop=True)
bottom_combo_mdas["rank_in_combo"] = bottom_combo_mdas.groupby(
    ["ads_type_nm","ads_category_nm"]
)["CVR"].rank(method="dense", ascending=False).astype(int)


# 결과 출력
print("=== clkCVR 행 기준 상위 {} ===".format(N_total))
display(top_rows)

print("\n=== 상위 행에서 추출된 타입+카테고리 조합 ===")
display(top_combos)

print("\n=== 해당 조합들에 속한 모든 매체사 (상세) ===")
display(top_combo_mdas)

print("\n\n=== clkCVR 행 기준 하위 {} ===".format(N_total))
display(bottom_rows)

print("\n=== 하위 행에서 추출된 타입+카테고리 조합 ===")
display(bottom_combos)

print("\n=== 해당 조합들에 속한 모든 매체사 (상세) ===")
display(bottom_combo_mdas)


# 타입-카테고리-매체사별 마진율 클릭수중앙값제한 df2 기준
def calc_margin_rate(df2, group_cols):
    agg = (
        df2.groupby(group_cols, dropna=False)[["rpt_time_acost", "rpt_time_earn"]]
        .sum()
        .reset_index()
    )
    agg["margin"] = agg["rpt_time_acost"] - agg["rpt_time_earn"]
    agg["margin_rate"] = (agg["margin"] / agg["rpt_time_acost"]).replace([np.inf, -np.inf], np.nan)
    agg["margin_rate_pct"] = (agg["margin_rate"] * 100).round(1)
    return agg
clkmargin = calc_margin_rate(df2, ["ads_type_nm","ads_category_nm","mda_idx"]).dropna().sort_values("margin_rate_pct", ascending=False)


N = 10  # 상/하위 개수

# 1) clkCVR 행 기준으로 상위/하위 N개(타입+카테고리+매체 단위)
top_rows2 = clkmargin.sort_values("margin_rate_pct", ascending=False).head(N).reset_index(drop=True)
bottom_rows2 = clkmargin.sort_values("margin_rate_pct", ascending=True).head(N).reset_index(drop=True)

# 2) 그 행들에서 타입+카테고리 조합 추출 (중복 제거)
top_combos2 = top_rows2[["ads_type_nm", "ads_category_nm"]].drop_duplicates().reset_index(drop=True)
bottom_combos2 = bottom_rows2[["ads_type_nm", "ads_category_nm"]].drop_duplicates().reset_index(drop=True)

# 3) 원본 clkCVR에서 해당 조합들에 속한 모든 매체사 행을 뽑기
top_combo_mdas2 = clkmargin.merge(top_combos2, on=["ads_type_nm", "ads_category_nm"], how="inner")
bottom_combo_mdas2 = clkmargin.merge(bottom_combos2, on=["ads_type_nm", "ads_category_nm"], how="inner")

# 4) 보기 좋게 정렬하고, 조합 내에서 CVR 순위 추가 (선택적)
top_combo_mdas2 = top_combo_mdas2.sort_values(["ads_type_nm","ads_category_nm","margin_rate_pct"],
                                            ascending=[True, True, False]).reset_index(drop=True)
bottom_combo_mdas2 = bottom_combo_mdas2.sort_values(["ads_type_nm","ads_category_nm","margin_rate_pct"],
                                                  ascending=[True, True, False]).reset_index(drop=True)

top_combo_mdas2["rank_in_combo"] = top_combo_mdas2.groupby(["ads_type_nm","ads_category_nm"])["margin_rate_pct"] \
                                               .rank(method="dense", ascending=False).astype(int)
bottom_combo_mdas2["rank_in_combo"] = bottom_combo_mdas2.groupby(["ads_type_nm","ads_category_nm"])["margin_rate_pct"] \
                                                     .rank(method="dense", ascending=False).astype(int)

# 5) 출력 (Jupyter 환경 가정)
print("=== clkCVR 행 기준 상위 {} ===".format(N))
display(top_rows2)

print("\n=== 상위 행에서 추출된 타입+카테고리 조합 ===")
display(top_combos2)

print("\n=== 해당 조합들에 속한 모든 매체사 (상세) ===")
display(top_combo_mdas2)

print("\n\n=== clkCVR 행 기준 하위 {} ===".format(N))
display(bottom_rows2)

print("\n=== 하위 행에서 추출된 타입+카테고리 조합 ===")
display(bottom_combos2)

print("\n=== 해당 조합들에 속한 모든 매체사 (상세) ===")
display(bottom_combo_mdas2)


# 전환율 전체 데이터
cvr_all = clkCVR.copy()

# mda_idx × 타입-카테고리 별 pivot 생성
pivot_cvr = cvr_all.pivot_table(
    index="mda_idx",
    columns=["ads_type_nm", "ads_category_nm"],
    values="CVR",
    aggfunc="mean"
)

# 매체별 요약 통계
summary_cvr = pd.DataFrame({
    "조합 수": pivot_cvr.notna().sum(axis=1),
    "최대 CVR": pivot_cvr.max(axis=1, skipna=True),
    "최소 CVR": pivot_cvr.min(axis=1, skipna=True),
    "평균 CVR": pivot_cvr.mean(axis=1, skipna=True),
    "표준편차": pivot_cvr.std(axis=1, skipna=True)
})

# 유형 분류 함수
def classify(row):
    if row["조합 수"] < 2:
        return "단일조합"
    if row["최소 CVR"] >= 70:
        return "일관 고효율형"
    if row["최대 CVR"] < 30:
        return "일관 저효율형"
    if row["최소 CVR"] < 10 and row["최대 CVR"] > 85:
        return "조합 편차형"
    return "중간형"

summary_cvr["유형"] = summary_cvr.apply(classify, axis=1)

# 유형별 매체 리스트
high_eff = summary_cvr[summary_cvr["유형"]=="일관 고효율형"].index.tolist()
low_eff = summary_cvr[summary_cvr["유형"]=="일관 저효율형"].index.tolist()
varied  = summary_cvr[summary_cvr["유형"]=="조합 편차형"].index.tolist()

print("일관 고효율형 매체:", high_eff)
print("일관 저효율형 매체:", low_eff)
print("조합 편차형 매체:", varied)

# 필요 시 각 유형별로 전체 조합별 CVR 확인
low_eff_combos = cvr_all[cvr_all["mda_idx"].isin(low_eff)][
    ["mda_idx","ads_type_nm","ads_category_nm","CVR"]
].drop_duplicates().sort_values(["mda_idx","ads_type_nm","ads_category_nm"])

high_eff_combos = cvr_all[cvr_all["mda_idx"].isin(high_eff)][
    ["mda_idx","ads_type_nm","ads_category_nm","CVR"]
].drop_duplicates().sort_values(["mda_idx","ads_type_nm","ads_category_nm"])

varied_combos = cvr_all[cvr_all["mda_idx"].isin(varied)][
    ["mda_idx","ads_type_nm","ads_category_nm","CVR"]
].drop_duplicates().sort_values(["mda_idx","ads_type_nm","ads_category_nm"])

# 유형별 매체 리스트 가져오기
high_eff = summary_cvr[summary_cvr["유형"]=="일관 고효율형"].index.tolist()
low_eff  = summary_cvr[summary_cvr["유형"]=="일관 저효율형"].index.tolist()
varied   = summary_cvr[summary_cvr["유형"]=="조합 편차형"].index.tolist()

# 각 유형별 전체 조합별 CVR 정리
def extract_combos(mda_list, label):
    return (
        cvr_all[cvr_all["mda_idx"].isin(mda_list)]
        [["mda_idx","ads_type_nm","ads_category_nm","CVR"]]
        .drop_duplicates()
        .sort_values(["mda_idx","ads_type_nm","ads_category_nm"])
        .assign(유형=label)
    )

low_eff_combos  = extract_combos(low_eff, "일관 저효율형")
high_eff_combos = extract_combos(high_eff, "일관 고효율형")
varied_combos   = extract_combos(varied, "조합 편차형")

# 최종
all_profiles = pd.concat([low_eff_combos, high_eff_combos, varied_combos], ignore_index=True)



# 조합 찾아보기
cvr_top = top_combo_mdas.copy()
cvr_bottom = bottom_combo_mdas.copy()
margin_top = top_combo_mdas2.copy()
margin_bottom = bottom_combo_mdas2.copy()

# 위/아래 합치기
cvr_all = pd.concat([cvr_top, cvr_bottom])
margin_all = pd.concat([margin_top, margin_bottom])

# 필요한 열만 남기기
cvr_all = cvr_all[["mda_idx","ads_type_nm","ads_category_nm","CVR"]].drop_duplicates()
margin_all = margin_all[["mda_idx","ads_type_nm","ads_category_nm","margin_rate_pct"]].drop_duplicates()

# 두 데이터 합치기 (매체 + 타입 + 카테고리 기준)
merged = pd.merge(cvr_all, margin_all,
                  on=["mda_idx","ads_type_nm","ads_category_nm"],
                  how="inner")

# 조건 맞는 애들만 고르기
result = merged[(merged["CVR"] > 90) & (merged["margin_rate_pct"] > 70)] # 고전환-고마진
result2 = merged[(merged["CVR"] > 90) & (merged["margin_rate_pct"] < 10)] # 고전환-저마진
result3 = merged[(merged["CVR"] < 20) & (merged["margin_rate_pct"] > 70)] # 저전환-고마진
result4 = merged[(merged["CVR"] < 20) & (merged["margin_rate_pct"] < 10)] # 저전환-저마진



# 시각화 

BINS   = [0, 20, 40, 60, 80, 100]
LABELS = ["0~20%", "20~40%", "40~60%", "60~80%", "80~100%"]

def add_combo_col(df):
    """ads_type_nm + ads_category_nm -> '조합' 컬럼 생성"""
    df = df.copy()
    df["조합"] = df["ads_type_nm"].astype(str) + " | " + df["ads_category_nm"].astype(str)
    return df


# 1) 히스토그램(막대) + 막대 위에 '개수(비율%)' 라벨
def plot_distribution_by_group(df, value_col, title_prefix=""):
    """
    df: 대상 DF
    value_col: 'CVR' 또는 'margin_rate_pct' 처럼 퍼센트(0~100) 값이 있는 컬럼명
    title_prefix: 그래프 타이틀 접두사
    """
    if df is None or len(df) == 0:
        print(f"[WARN] 입력 DF 비어있음: {title_prefix}")
        return

    df = add_combo_col(df)
    # 혹시 모를 문자열 -> 숫자 변환
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    for comb in df["조합"].dropna().unique():
        subset = df[df["조합"] == comb].copy()
        subset = subset[subset[value_col].notna()]
        if len(subset) == 0:
            print(f"[INFO] '{comb}' 조합에 {value_col} 값이 없음. 스킵.")
            continue

        subset["구간"] = pd.cut(subset[value_col], bins=BINS, labels=LABELS, include_lowest=True, right=True)
        dist  = subset["구간"].value_counts().reindex(LABELS, fill_value=0)
        total = dist.sum()
        share = (dist / total * 100).round(1)

        # 그리기
        plt.figure(figsize=(6,4))
        bars = plt.bar(dist.index, dist.values, edgecolor="black")

        # y-리밋 여유(라벨이 잘 보이도록)
        top = dist.max()
        ymax = top + (top * 0.15 + 0.5)  # 여백
        plt.ylim(0, ymax)

        # 막대 위 라벨: "개수 (비율%)"
        for i, b in enumerate(bars):
            v = dist.values[i]
            p = share.values[i]
            plt.text(
                b.get_x() + b.get_width()/2, 
                b.get_height() + max(0.02*top, 0.2), 
                f"{int(v)} ({p:.1f}%)",
                ha="center", va="bottom", fontsize=9
            )

        plt.title(f"{title_prefix} {comb} {value_col} 분포")
        plt.xlabel(f"{value_col} 구간")
        plt.ylabel("매체 수")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


# 2) 화이트리스트(유지/배제) 테이블
def make_whitelist_table(df, value_col, threshold=70):
    """
    threshold는 퍼센트 단위(예: 70 -> 70%)
    반환: [조합, mda_idx, value_col, 등급] 정렬 테이블
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["조합", "mda_idx", value_col, "등급"])

    df = add_combo_col(df).copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df["등급"] = df[value_col].apply(lambda x: "✅ 유지·확대" if pd.notnull(x) and x >= threshold else "❌ 축소·배제")
    out = df[["조합", "mda_idx", value_col, "등급"]].sort_values(by=["조합", value_col], ascending=[True, False])
    return out


# 3) 구간 요약표(매체수/비율) 생성
def make_bin_summary(df, value_col, combo_name=None):
    """
    combo_name을 None으로 두면 DF 안의 모든 '조합'에 대해 요약 반환.
    특정 조합만 원하면 combo_name="네이버 | 구독(간편적립)" 처럼 지정.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["조합", "구간", "매체수", "비율(%)"])

    df = add_combo_col(df).copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df[value_col].notna()]

    targets = [combo_name] if combo_name else df["조합"].dropna().unique().tolist()
    frames = []
    for comb in targets:
        sub = df[df["조합"] == comb].copy()
        if len(sub) == 0:
            continue
        sub["구간"] = pd.cut(sub[value_col], bins=BINS, labels=LABELS, include_lowest=True, right=True)
        counts = sub["구간"].value_counts().reindex(LABELS, fill_value=0)
        share  = (counts / counts.sum() * 100).round(1)
        tmp = pd.DataFrame({"구간": LABELS, "매체수": counts.values, "비율(%)": share.values})
        tmp.insert(0, "조합", comb)
        frames.append(tmp)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["조합", "구간", "매체수", "비율(%)"])


# 4) 원본 DF 확보 (덮어쓰기 방지)
df_cvr_all    = top_combo_mdas.copy()    # 전환율 전체(상위에서 사용)
df_margin_all = top_combo_mdas2.copy()   # 마진율 전체(상위에서 사용)

# 5) 상위/하위 조합 필터링
# 전환율 상위: 네이버·인스타 | 구독(간편적립)
cvr_top_df = df_cvr_all[
    (df_cvr_all["ads_type_nm"].isin(["네이버", "인스타"])) &
    (df_cvr_all["ads_category_nm"] == "구독(간편적립)")
]

# 전환율 하위: 참여형 | 경험하기-CPA, 유료참여
cvr_bottom_df = df_cvr_all[
    (df_cvr_all["ads_type_nm"] == "참여형") &
    (df_cvr_all["ads_category_nm"].isin(["경험하기-CPA", "유료참여"]))
]

# 마진율 상위: CPS | 쇼핑-상품 + 설치형 | 경험하기-CPI/CPE
margin_top_df = df_margin_all[
    ((df_margin_all["ads_type_nm"] == "CPS") & (df_margin_all["ads_category_nm"] == "쇼핑-상품")) |
    ((df_margin_all["ads_type_nm"] == "설치형") & (df_margin_all["ads_category_nm"] == "경험하기-CPI/CPE"))
]

# 마진율 하위: 설치형 | 앱(간편적립) + 참여형 | 간편미션, 무료참여
margin_bottom_df = df_margin_all[
    ((df_margin_all["ads_type_nm"] == "설치형") & (df_margin_all["ads_category_nm"] == "앱(간편적립)")) |
    ((df_margin_all["ads_type_nm"] == "참여형") & (df_margin_all["ads_category_nm"].isin(["간편미션", "무료참여"])))
]

# 6) 그래프 + 화이트리스트 + 요약표 실행
# 전환율 상위
plot_distribution_by_group(cvr_top_df, value_col="CVR", title_prefix="[전환율 상위]")
table_cvr_top = make_whitelist_table(cvr_top_df, value_col="CVR", threshold=70)
summary_cvr_top_all = make_bin_summary(cvr_top_df, value_col="CVR")  # 모든 조합 요약
summary_cvr_top_nav = make_bin_summary(cvr_top_df, value_col="CVR", combo_name="네이버 | 구독(간편적립)")  # 특정 조합 요약

# 전환율 하위
plot_distribution_by_group(cvr_bottom_df, value_col="CVR", title_prefix="[전환율 하위]")
table_cvr_bottom = make_whitelist_table(cvr_bottom_df, value_col="CVR", threshold=70)
summary_cvr_bottom_all = make_bin_summary(cvr_bottom_df, value_col="CVR")

# 마진율 상위
plot_distribution_by_group(margin_top_df, value_col="margin_rate_pct", title_prefix="[마진율 상위]")
table_margin_top = make_whitelist_table(margin_top_df, value_col="margin_rate_pct", threshold=50)
summary_margin_top_all = make_bin_summary(margin_top_df, value_col="margin_rate_pct")

# 마진율 하위
plot_distribution_by_group(margin_bottom_df, value_col="margin_rate_pct", title_prefix="[마진율 하위]")
table_margin_bottom = make_whitelist_table(margin_bottom_df, value_col="margin_rate_pct", threshold=50)
summary_margin_bottom_all = make_bin_summary(margin_bottom_df, value_col="margin_rate_pct")

# 7) 결과 확인
print("=== 전환율 상위(화이트리스트 미리보기) ===")
print(table_cvr_top.head(), "\n")

print("=== 전환율 상위(구간 요약: 전체) ===")
print(summary_cvr_top_all.head(10), "\n")

print("=== 전환율 상위(구간 요약: 네이버 | 구독(간편적립)) ===")
print(summary_cvr_top_nav, "\n")

print("=== 전환율 하위(화이트리스트 미리보기) ===")
print(table_cvr_bottom.head(), "\n")

print("=== 마진율 상위(화이트리스트 미리보기) ===")
print(table_margin_top.head(), "\n")

print("=== 마진율 하위(화이트리스트 미리보기) ===")
print(table_margin_bottom.head(), "\n")

