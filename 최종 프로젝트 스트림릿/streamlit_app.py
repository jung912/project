import streamlit as st
import pandas as pd
import os
import altair as alt
DATA_DIR = "."
GROUP_FILES = {
    "A_high_perf_big": "dataset_A_high_perf_big.parquet",
    "B_high_perf_small": "dataset_B_high_perf_small.parquet",
    "C_low_perf_big": "dataset_C_low_perf_big.parquet",
    "D_low_perf_small": "dataset_D_low_perf_small.parquet",
}
SUMMARY_FILE = os.path.join(DATA_DIR, "summary_importance.parquet")
@st.cache_data
def load_group_data(group_name):
    path = os.path.join(DATA_DIR, GROUP_FILES[group_name])
    return pd.read_parquet(path)
@st.cache_data
def get_all_unique_values():
    mda_vals, cat_vals, type_vals = set(), set(), set()
    for g, f in GROUP_FILES.items():
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            try:
                df_check = pd.read_parquet(path, columns=["mda_idx","ads_category","ads_type"])
                mda_vals.update(df_check["mda_idx"].unique())
                cat_vals.update(df_check["ads_category"].unique())
                type_vals.update(df_check["ads_type"].unique())
            except:
                continue
    return sorted(mda_vals), sorted(cat_vals), sorted(type_vals)
def find_group_for_value(value, column):
    for g, f in GROUP_FILES.items():
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df_check = pd.read_parquet(path, columns=[column])
            try:
                value_casted = df_check[column].dtype.type(value)
            except Exception:
                value_casted = value
            if value_casted in df_check[column].unique():
                return g
    return None
def colored_metric(label, value, threshold_high=None, threshold_low=None, is_percent=False):
    if is_percent:
        display_val = f"{value:.2f}%"
    else:
        display_val = f"{value:.2f}"
    color = "white"
    if threshold_high is not None and value >= threshold_high:
        color = "limegreen"
    elif threshold_low is not None and value <= threshold_low:
        color = "red"
    st.markdown(
        f"""
        <div style="padding:10px; border-radius:10px; background-color:#222; text-align:center">
            <div style="font-size:14px; color:lightgray">{label}</div>
            <div style="font-size:22px; font-weight:bold; color:{color}">{display_val}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.set_page_config(page_title="광고 추천 대시보드", layout="wide")
# =============================
# 제목 섹션
# =============================
st.markdown(
    """
    <div style="padding: 1.5rem; background-color: #F9FAFB; border-radius: 12px; margin-bottom: 1.5rem;">
        <h2 style="margin-bottom: 0.3rem; color: #111827;">Model</h2>
        <p style="margin: 0; color: #6B7280; font-size: 0.95rem;">
            매체사·광고유형 조합별 성과를 예측하고 최적 조합을 추천합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
mda_list, cat_list, type_list = get_all_unique_values()
col1, col2, col3 = st.columns(3)
mda_sel = col1.selectbox("mda_idx 선택", [""] + mda_list)
cat_sel = col2.selectbox("카테고리 선택", [""] + cat_list)
type_sel = col3.selectbox("타입 선택", [""] + type_list)
inputs = [bool(mda_sel), bool(cat_sel), bool(type_sel)]
if sum(inputs) == 0:
    st.info("mda_idx, 카테고리, 타입 중 하나만 선택하세요.")
elif sum(inputs) > 1:
    st.warning("하나만 선택해야 합니다.")
else:
    if mda_sel:
        mode, user_input, col_name = "mda_idx", mda_sel, "mda_idx"
    elif cat_sel:
        mode, user_input, col_name = "카테고리", cat_sel, "ads_category"
    else:
        mode, user_input, col_name = "타입", type_sel, "ads_type"
    group = find_group_for_value(user_input, col_name)
    if not group:
        st.error("해당 값이 포함된 그룹을 찾을 수 없습니다.")
    else:
        st.success(f"{mode}={user_input} → {group} 그룹에서 발견됨")
        # 데이터 불러오기
        df = load_group_data(group)
        df_sel = df[df[col_name] == user_input]
        if df_sel.empty:
            st.warning("선택한 값에 해당하는 데이터가 없습니다.")
        else:
            df_sel["ROAS"] = df_sel["earn_cost"] / df_sel["adv_cost"].replace(0, 1)
            df_sel["profitability"] = (df_sel["earn_cost"] - df_sel["adv_cost"]) / df_sel["adv_cost"].replace(0, 1)
            df["ROAS"] = df["earn_cost"] / df["adv_cost"].replace(0, 1)
            df["profitability"] = (df["earn_cost"] - df["adv_cost"]) / df["adv_cost"].replace(0, 1)
            st.subheader("핵심 지표 (선택 vs 그룹 평균)")
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                colored_metric("CVR", df_sel["CVR"].mean(), threshold_high=df["CVR"].mean()*1.2, threshold_low=df["CVR"].mean()*0.8, is_percent=True)
                st.caption(f"그룹 평균: {df['CVR'].mean():.2f}%")
            with k2:
                colored_metric("Margin Rate", df_sel["margin_rate"].mean(), threshold_high=df["margin_rate"].mean()*1.2, threshold_low=df["margin_rate"].mean()*0.8)
                st.caption(f"그룹 평균: {df['margin_rate'].mean():.2f}")
            with k3:
                colored_metric("ROAS", df_sel["ROAS"].mean(), threshold_high=df["ROAS"].mean()*1.2, threshold_low=df["ROAS"].mean()*0.8)
                st.caption(f"그룹 평균: {df['ROAS'].mean():.2f}")
            with k4:
                colored_metric("Profitability", df_sel["profitability"].mean(), threshold_high=df["profitability"].mean()*1.2, threshold_low=df["profitability"].mean()*0.8)
                st.caption(f"그룹 평균: {df['profitability'].mean():.2f}")
            with k5:
                m1 = df_sel["CVR"].mean() * 0.5 + df_sel["margin_rate"].mean() * 0.5
                colored_metric("균형 KPI", m1, threshold_high=(df["CVR"].mean()*0.5 + df["margin_rate"].mean()*0.5)*1.2, threshold_low=(df["CVR"].mean()*0.5 + df["margin_rate"].mean()*0.5)*0.8)
                st.caption(f"그룹 평균: {(df['CVR'].mean()*0.5 + df['margin_rate'].mean()*0.5):.2f}")
            # =============================
            # 추천 조합 (Rank만 보여주고 인덱스 숨김)
            # =============================
            if mode == "mda_idx":
                group_cols = ["ads_category", "ads_type"]
            elif mode == "카테고리":
                group_cols = ["mda_idx", "ads_type"]
            else:
                group_cols = ["mda_idx", "ads_category"]
            top_combos = (
                df_sel.groupby(group_cols)
                .size()
                .reset_index(name="count")
            )
            top_combos = (
                top_combos.sort_values("count", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            # Rank 추가
            top_combos.insert(0, "Rank", range(1, len(top_combos) + 1))
            # count 제거
            top_combos = top_combos.drop(columns=["count"])
            st.subheader(f"상위 {len(top_combos)}개 추천 조합")
            st.dataframe(top_combos, hide_index=True)  # :흰색_확인_표시: 인덱스 제거
            # =============================
            # KPI 막대그래프 비교
            # =============================
            sel_metrics = {
                "CVR": df_sel["CVR"].mean(),
                "Margin Rate": df_sel["margin_rate"].mean(),
                "ROAS": df_sel["ROAS"].mean(),
                "Profitability": df_sel["profitability"].mean(),
                "균형 KPI": m1
            }
            group_metrics = {
                "CVR": df["CVR"].mean(),
                "Margin Rate": df["margin_rate"].mean(),
                "ROAS": df["ROAS"].mean(),
                "Profitability": df["profitability"].mean(),
                "균형 KPI": (df["CVR"].mean()*0.5 + df["margin_rate"].mean()*0.5)
            }
            metrics_df = pd.DataFrame({
                "지표": list(sel_metrics.keys()),
                "선택된 값": list(sel_metrics.values()),
                "그룹 평균": list(group_metrics.values())
            })
            st.subheader("선택된 값 vs 그룹 평균 비교 (막대그래프)")
            chart = alt.Chart(metrics_df).transform_fold(
                ["선택된 값", "그룹 평균"],
                as_=["구분", "값"]
            ).mark_bar().encode(
                x=alt.X("지표:N", axis=alt.Axis(title="KPI")),
                y=alt.Y("값:Q", axis=alt.Axis(title="값")),
                color="구분:N",
                xOffset="구분:N"
            ).properties(width=600, height=400)
            st.altair_chart(chart, use_container_width=True)
