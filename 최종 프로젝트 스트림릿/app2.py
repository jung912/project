import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
import gdown

# =============================
# Page & Global Styles (UI only)
# =============================
st.set_page_config(page_title="Trend EDA", layout="wide")

ACCENT = "#7C83FF"           # 연보라 메인
GRID = "#EEF2FF"

st.markdown(
    f"""
<style>
/* 전체 레이아웃 폭/여백 (좌우 여백 보장) */
.block-container {{
  max-width: 1180px;
  margin-left: auto;
  margin-right: auto;
  padding-top: 1.0rem !important;
  padding-bottom: 1.2rem !important;
}}

/* 페이지 타이틀(히어로) */
.hero {{
  background: #F8FAFC;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  padding: 16px 18px;
  margin-bottom: 12px;
}}
.hero h1 {{
  margin: 0 0 6px 0;
  font-size: 24px;
  font-weight: 800; color: #111827;
}}
.hero p {{ margin: 0; color: #6B7280; font-size: 13px; }}

/* 섹션 타이틀 — 리본만 */
.sec-row {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
.sec-title {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 0;
  background: transparent;
  border: none;
  color: #0F172A;
  font-size: 18px;
  font-weight: 800;
  margin: 0 0 6px 0;   /* 제목 밑 한 칸 */
}}
.sec-dot {{
  width: 6px;
  height: 22px;
  border-radius: 3px;
  background: {ACCENT};
  display: inline-block;
}}

/* KPI 박스 톤 */
[data-testid="stMetric"] {{
  border: 1px solid #E5E7EB; border-radius: 12px; padding: 10px 12px; background: #FFFFFF;
}}
[data-testid="stMetricValue"] {{ font-size: 1.85rem; font-weight: 900; color: #0F172A; }}
[data-testid="stMetricLabel"] {{ font-size: 1.05rem; font-weight: 700; color: #6B7280; }}

/* 표 헤더 톤 */
thead tr th {{ background:#F3F4F6 !important; color:#374151 !important; }}

/* Plotly 배경/그리드 */
.js-plotly-plot .plotly, .stPlotlyChart {{ background: transparent !important; }}

/* 얇은 구분선 */
.sep {{ border:none; border-top:1px solid #E5E7EB; margin:10px 0 12px 0; }}

/* 버튼 기본 스타일 */
.stButton > button {{
  border-radius: 9px; 
  min-width: 112px; 
  min-height: 36px;
  padding: 8px 12px; 
  font-weight: 700; 
}}
/* primary / secondary 색상(연보라 일관) */
[data-testid="baseButton-primary"], .stButton button[kind="primary"] {{
  background:{ACCENT}; border-color:{ACCENT}; color:#FFFFFF;
}}
[data-testid="baseButton-primary"]:hover, .stButton button[kind="primary"]:hover {{
  background:#6E75FF; border-color:#6E75FF; color:#FFFFFF;
}}
[data-testid="baseButton-secondary"], .stButton button[kind="secondary"] {{
  background:#EEF0FF; border-color:#DDE1FF; color:#3730A3;
}}
[data-testid="baseButton-secondary"]:hover, .stButton button[kind="secondary"]:hover {{
  background:#E2E6FF; border-color:#D3D8FF; color:#2E2A8F;
}}

/* 순수익 카드의 '원'만 작게 */
.won-unit {{ font-size: 0.72em; color:#6B7280; margin-left:4px; }}

/* ✅ 비어있는/고아 pillwrap 완전 제거 (원형 유령 제거) */
.pillwrap, .pillwrap:empty {{
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
}}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# =============================
# 파일 로드/정규화 (1년 CSV만 사용)
# =============================

# 현황 데이터 드라이브 주소
# https://drive.google.com/file/d/17Ko2r5aa-oqcNZyBwn6Hl7_zY0DD-bKG/view?usp=sharing

@st.cache_data
def download_file():
    file_id = "17Ko2r5aa-oqcNZyBwn6Hl7_zY0DD-bKG"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "trend_eda_1year_mapping.csv"   # 괄호 제거
    gdown.download(url, output, quiet=False)
    return output

YEAR_FILE_CANDIDATES = [
    download_file()   # "trend_eda_1year_mapping.csv"
]

# 3. CSV 읽기 함수
@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(p)

# 4. 후보들 중 하나 읽기
@st.cache_data(show_spinner=False)
def _read_one(cands):
    for p in cands:
        try:
            df = _read_csv(p)
            return df, Path(p).name
        except FileNotFoundError:
            continue
    raise FileNotFoundError(str(cands))

# 5. 숫자 처리 유틸
def to_num(s): return pd.to_numeric(s, errors="coerce")

def safe_div(a, b):
    a = to_num(a); b = to_num(b)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = a / b
    return x.replace([np.inf, -np.inf], np.nan)

def clean_media_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True)

# 6. 데이터 정규화
@st.cache_data(show_spinner=False)
def normalize_year(df):
    out = pd.DataFrame()
    out["_date"] = pd.to_datetime(df["rpt_time_date"], errors="coerce")
    out["_ym"]   = out["_date"].dt.to_period("M").astype(str)
    out["_type"]     = (df["ads_type_nm"].astype(str) if "ads_type_nm" in df.columns else df["ads_type"].astype(str))
    out["_category"] = (df["ads_category_nm"].astype(str) if "ads_category_nm" in df.columns else df["ads_category"].astype(str))
    out["_media_id"] = df["mda_idx"].astype(str)
    out["_clicks"] = to_num(df["rpt_time_clk"])
    out["_conv"]   = to_num(df["rpt_time_turn"])
    out["_earn"]   = to_num(df["rpt_time_earn"])
    out["_cost"]   = to_num(df["rpt_time_cost"])
    out["_acost"]  = to_num(df.get("rpt_time_acost", np.nan))
    out["_cvr"]    = safe_div(out["_conv"], out["_clicks"])
    out["_margin"] = safe_div(out["_acost"] - out["_earn"], out["_acost"])
    out["_acos"]   = safe_div(out["_cost"], out["_earn"])
    for c in ["_type","_category","_media_id","_ym"]:
        out[c] = out[c].astype("category")
    return out

# 7. 실행
df_raw, year_used = _read_one(YEAR_FILE_CANDIDATES)
df = normalize_year(df_raw)

# =============================
# 공통 유틸
# =============================
@st.cache_data(show_spinner=False)
def grouped_ratio(df: pd.DataFrame, metric: str, group_cols):
    group_cols = list(group_cols)
    if metric == "_cvr":
        g = (df.groupby(group_cols, dropna=False)
               .agg(sum_clk=("_clicks","sum"), sum_conv=("_conv","sum"))
               .reset_index())
        g = g[g["sum_clk"].fillna(0) > 0]
        g["value"] = safe_div(g["sum_conv"], g["sum_clk"])
    else:
        g = (df.groupby(group_cols, dropna=False)
               .agg(sum_acost=("_acost","sum"), sum_earn=("_earn","sum"))
               .reset_index())
        g = g[g["sum_acost"].fillna(0) > 0]
        g["value"] = safe_div(g["sum_acost"] - g["sum_earn"], g["sum_acost"])
    return g

def ratio_of_sums_cvr(df_):
    clk, conv = df_["_clicks"].sum(), df_["_conv"].sum()
    return (conv/clk) if clk else np.nan

def ratio_of_sums_margin(df_):
    acost, earn = df_["_acost"].sum(), df_["_earn"].sum()
    return ((acost - earn)/acost) if acost else np.nan

def pct(v, d=2): return "-" if pd.isna(v) else f"{v*100:.{d}f}%"

def tune_fig(fig, h=360):
    fig.update_layout(
        template="simple_white",
        height=h,
        margin=dict(l=40, r=30, t=50, b=50),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#111827"),
    )
    fig.update_xaxes(automargin=True, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor=GRID, tickformat=".1%")
    return fig

def build_table(df_, metric, asc):
    g = grouped_ratio(df_, metric, ("_media_id",)).sort_values("value", ascending=asc).head(10).copy()
    g["_media_id"] = clean_media_id(g["_media_id"])
    if metric == "_cvr":
        g["전환율(%)"] = (g["value"]*100).round(2)
        g.rename(columns={"_media_id":"매체번호","sum_clk":"클릭수","sum_conv":"전환수"}, inplace=True)
        table = g[["매체번호","클릭수","전환수","전환율(%)"]]
        fmt   = {"클릭수":"{:,.0f}", "전환수":"{:,.0f}", "전환율(%)":"{:.2f}"}
    else:
        sums = df_.groupby("_media_id").agg(ACOST=("_acost","sum"), EARN=("_earn","sum")).reset_index()
        sums["_media_id"] = clean_media_id(sums["_media_id"])
        g = g.rename(columns={"_media_id":"매체번호"}).merge(
            sums, left_on="매체번호", right_on="_media_id", how="left"
        ).drop(columns=["_media_id"])
        g["마진율(%)"] = (g["value"]*100).round(2)
        table = g[["매체번호","ACOST","EARN","마진율(%)"]]
        fmt   = {"ACOST":"{:,.0f}", "EARN":"{:,.0f}", "마진율(%)":"{:.2f}"}
    return table, fmt

# =============================
# 히어로 타이틀
# =============================
st.markdown(
    """
<div class="hero">
  <h1>Trend EDA</h1>
  <p>전환율/마진율을 월별·연간 추이를 확인하고 상/하위 매체를 식별합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 상단: 제목 + 지표 버튼 + 타입/카테고리
# ============================================================
st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>전환율/마진율 현황</div></div>', unsafe_allow_html=True)

left_third, mid_third, right_third = st.columns([1, 1, 1])

# 지표 토글 (세션 스테이트)
if "metric" not in st.session_state:
    st.session_state.metric = "전환율"

with left_third:
    b1, b2 = st.columns(2)
    with b1:
        if st.button(
            "전환율",
            type=("primary" if st.session_state.metric == "전환율" else "secondary"),
            use_container_width=True,
            key="btn_metric_cvr",
        ):
            st.session_state.metric = "전환율"
    with b2:
        if st.button(
            "마진율",
            type=("primary" if st.session_state.metric == "마진율" else "secondary"),
            use_container_width=True,
            key="btn_metric_margin",
        ):
            st.session_state.metric = "마진율"

with mid_third:
    st.write("")

with right_third:
    all_types = sorted(pd.Series(df["_type"].unique()).astype(str).tolist())
    all_cats  = sorted(pd.Series(df["_category"].unique()).astype(str).tolist())
    fc1, fc2 = st.columns(2)
    sel_type = fc1.selectbox("타입", ["(전체)"] + all_types, index=0, key="type")
    sel_cat  = fc2.selectbox("카테고리", ["(전체)"] + all_cats, index=0, key="cat")

sel_metric = st.session_state.metric
met = "_cvr" if sel_metric == "전환율" else "_margin"

def apply_typecat(df_):
    if sel_type != "(전체)": df_ = df_[df_["_type"] == sel_type]
    if sel_cat  != "(전체)": df_ = df_[df_["_category"] == sel_cat]
    return df_

df_g = apply_typecat(df.copy())

# =============================
# KPI (타입/카테고리만 반영)
# =============================
scope = df_g
total_conv   = scope["_conv"].sum()
net_profit   = scope["_earn"].sum() - scope["_cost"].sum()
weighted_cvr = ratio_of_sums_cvr(scope)
weighted_mgn = ratio_of_sums_margin(scope)

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("총 전환수", f"{total_conv:,.0f}")
with k2: st.metric("평균 전환율", pct(weighted_cvr, 2))
with k3:
    # 숫자는 동일, '원'만 작게
    st.markdown(
        f"""
        <div style="border:1px solid #E5E7EB; border-radius:12px; padding:10px 12px; background:#FFFFFF;">
          <div style="color:#6B7280; font-size:13px; font-weight:700;">순수익(매출-비용)</div>
          <div style="font-size:1.85rem; font-weight:900; color:#0F172A;">
            {net_profit:,.0f}<span class="won-unit"> 원</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4: st.metric("평균 마진율", pct(weighted_mgn, 2))

st.markdown('<hr class="sep" />', unsafe_allow_html=True)

# =============================
# 좌(월별) / 우(1년)
# =============================
left, right = st.columns(2)

@st.cache_data(show_spinner=False)
def monthly_weighted_series(df_, metric):
    if metric == "_cvr":
        g = (df_.groupby("_ym").agg(sum_clk=("_clicks","sum"), sum_conv=("_conv","sum")).reset_index())
        g = g[g["sum_clk"].fillna(0) > 0]; g["value"] = g["sum_conv"] / g["sum_clk"]
    else:
        g = (df_.groupby("_ym").agg(sum_acost=("_acost","sum"), sum_earn=("_earn","sum")).reset_index())
        g = g[g["sum_acost"].fillna(0) > 0]; g["value"] = (g["sum_acost"] - g["sum_earn"]) / g["sum_acost"]
    g["_ym"] = g["_ym"].astype(str)
    g["_ym_ord"] = pd.to_datetime(g["_ym"], format="%Y-%m", errors="coerce")
    return g.sort_values("_ym_ord").drop(columns=["_ym_ord"])

# ---------- 좌: 월별 ----------
with left:
    tcol, fcol = st.columns([0.66, 0.34])
    with tcol:
        st.markdown('<div class="sec-title"><span class="sec-dot"></span>월별 추이</div>', unsafe_allow_html=True)
    with fcol:
        months = sorted(df["_ym"].astype(str).unique().tolist())
        default_idx = months.index(max(months)) if months else 0
        sel_month = st.selectbox("월 선택", months, index=default_idx, key="month")

    month_df = df_g[df_g["_ym"].astype(str) == sel_month]

    @st.cache_data(show_spinner=False)
    def media_top10_weighted(df_, metric):
        g = grouped_ratio(df_, metric, ("_media_id",)).sort_values("value", ascending=False).head(10).copy()
        g["_media_id"] = clean_media_id(g["_media_id"])
        order = g["_media_id"].tolist()
        g["_media_id"] = pd.Categorical(g["_media_id"], categories=order, ordered=True)
        return g[["_media_id","value"]], order

    if sel_type == "(전체)" or sel_cat == "(전체)":
        top10, order = media_top10_weighted(month_df, met)
        if top10.empty:
            st.info("표시할 데이터가 없습니다.")
        else:
            fig = px.bar(
                top10, x="_media_id", y="value",
                labels={"_media_id":"매체번호","value":f"{sel_metric} (가중)"},
                text=(top10["value"]*100).round(2).astype(str)+"%"
            )
            fig.update_layout(xaxis=dict(type="category", categoryorder="array", categoryarray=order), bargap=0.35)
            fig.update_traces(textposition="outside",
                              marker_color=ACCENT, marker_line_color=ACCENT,
                              hovertemplate="매체=%{x}<br>"+sel_metric+"=%{y:.2%}<extra></extra>")
            st.plotly_chart(tune_fig(fig, 420), use_container_width=True, key="plot_month_top10")
    else:
        g = grouped_ratio(month_df, met, ("_media_id",))
        fig = px.histogram(
            g, x="value", nbins=20,
            labels={"value": f"[{sel_type}·{sel_cat}] {sel_month} 매체별 가중 {sel_metric} 분포"}
        )
        fig.update_traces(marker_color=ACCENT, marker_line_color=ACCENT)
        v = ratio_of_sums_cvr(month_df) if met == "_cvr" else ratio_of_sums_margin(month_df)
        if pd.notna(v):
            fig.add_vline(x=float(v), line_dash="dash", line_width=2, line_color="#94A3B8")
        fig = tune_fig(fig, 420)
        fig.update_yaxes(tickformat="", title_text="count")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, key="plot_month_hist")

    month_top_df,  month_top_fmt  = build_table(month_df, met, asc=False)
    month_low_df,  month_low_fmt  = build_table(month_df, met, asc=True)

# ---------- 우: 1년 ----------
with right:
    tcol, fcol = st.columns([0.66, 0.34])
    with tcol:
        st.markdown('<div class="sec-title"><span class="sec-dot"></span>1년 월별 추이</div>', unsafe_allow_html=True)
    with fcol:
        if sel_type != "(전체)" and sel_cat != "(전체)":
            media_pool = df_g["_media_id"]
        else:
            media_pool = df["_media_id"]
        media_options = ["(전체)"] + sorted(clean_media_id(media_pool.astype(str)).unique().tolist())
        sel_media = st.selectbox("매체 선택", media_options, index=0, key="media")

    scope_y = df_g.copy()
    if sel_media != "(전체)":
        scope_y = scope_y[clean_media_id(scope_y["_media_id"]) == sel_media]

    line = monthly_weighted_series(scope_y, met)
    suffix = "" if sel_media == "(전체)" else f" — 매체 {sel_media}"
    figY = px.line(line, x="_ym", y="value", markers=True,
                   labels={"_ym":"월", "value": f"월별 가중 {sel_metric}{suffix}"})
    figY.update_traces(line_color=ACCENT, marker_color=ACCENT)
    st.plotly_chart(tune_fig(figY, 420), use_container_width=True, key="plot_year_line")

    year_top_df,  year_top_fmt  = build_table(df_g, met, asc=False)
    year_low_df,  year_low_fmt  = build_table(df_g, met, asc=True)

st.markdown('<hr class="sep" />', unsafe_allow_html=True)

# =============================
# 표: 상위 / 하위
# =============================
st.markdown('<div class="sec-title"><span class="sec-dot"></span>상위 매체 10 (월/1년)</div>', unsafe_allow_html=True)
r1c1, r1c2 = st.columns(2)
with r1c1:
    if len(month_top_df)==0: st.info("표시할 데이터가 없습니다.")
    else:
        st.dataframe(month_top_df.set_index("매체번호").style.format(month_top_fmt),
                     use_container_width=True)
with r1c2:
    if len(year_top_df)==0: st.info("표시할 데이터가 없습니다.")
    else:
        st.dataframe(year_top_df.set_index("매체번호").style.format(year_top_fmt),
                     use_container_width=True)

st.markdown('<div class="sec-title" style="margin-top:10px;"><span class="sec-dot"></span>하위 매체 10 (월/1년)</div>', unsafe_allow_html=True)
r2c1, r2c2 = st.columns(2)
with r2c1:
    if len(month_low_df)==0: st.info("표시할 데이터가 없습니다.")
    else:
        st.dataframe(month_low_df.set_index("매체번호").style.format(month_low_fmt),
                     use_container_width=True)
with r2c2:
    if len(year_low_df)==0: st.info("표시할 데이터가 없습니다.")
    else:
        st.dataframe(year_low_df.set_index("매체번호").style.format(year_low_fmt),
                     use_container_width=True)

st.markdown('<hr class="sep" />', unsafe_allow_html=True)

# =============================
# 📝 인사이트
# =============================
st.markdown('<div class="sec-title"><span class="sec-dot"></span> 인사이트</div>', unsafe_allow_html=True)

if (sel_type == "(전체)") or (sel_cat == "(전체)"):
    st.caption("타입과 카테고리를 선택하면 인사이트가 표시됩니다.")
else:
    base_year = df[(df["_type"] == sel_type) & (df["_category"] == sel_cat)].copy()
    if base_year.empty:
        st.info("선택한 조건의 1년 데이터가 없습니다.")
    else:
        here_combo = grouped_ratio(base_year, met, ["_type","_category"])
        val_combo = float(here_combo["value"].iloc[0]) if not here_combo.empty else np.nan

        A, B, C = 15.0, 50.0, 80.0  # %
        def advice(p):
            if pd.isna(p): return "데이터가 부족합니다."
            p *= 100.0
            if p <= A: return "성과가 매우 낮습니다. 예산 축소/삭제 또는 전략 재평가가 필요합니다."
            if p <= B: return "보통 수준입니다. 소재/랜딩/입찰 최적화를 통해 상위 진입을 시도하세요."
            if p <= C: return "안정적인 주력 후보군입니다. 추가 세분화/확대로 상위권을 노려보세요."
            return "최상위 구간입니다. 예산 증액 및 집중 운영을 권장합니다."

        c1, c2, c3 = st.columns([0.45, 0.22, 0.33])
        with c1: st.metric("조합", f"{sel_type} · {sel_cat}")
        with c2: st.metric("지표", sel_metric)
        with c3: st.metric("가중평균", pct(val_combo, 2))

        st.markdown("**조합 운영 제안**")
        st.markdown(f'> “{advice(val_combo)}”')

        st.markdown(f"#### 조합 내 매체 분포 — {sel_type} · {sel_cat}")
        media_grp = grouped_ratio(base_year, met, ["_media_id"]).dropna(subset=["value"]).copy()
        media_grp["_media_id"] = clean_media_id(media_grp["_media_id"])

        def bucket_abs(v):
            if pd.isna(v): return "-"
            p = v * 100.0
            if p <= A: return "0–15%"
            if p <= B: return "16–50%"
            if p <= C: return "51–80%"
            return "81–100%"

        media_grp["구간"] = media_grp["value"].apply(bucket_abs)
        zone_media = {
            z: media_grp.loc[media_grp["구간"] == z, "_media_id"].tolist()
            for z in ["0–15%","16–50%","51–80%","81–100%"]
        }
        zone_tip = {
            "0–15%":  "성과 매우 낮음 · 정리/전략 재평가",
            "16–50%": "보통 · 개선 포인트 탐색",
            "51–80%": "양호 · 주력 후보 유지",
            "81–100%":"상위 · 집중 운영/확대",
        }

        cL, cR = st.columns(2)
        with cL:
            st.markdown(f"**0–15%**")
            st.caption(", ".join(zone_media["0–15%"]) or "-")
            st.markdown(f"**16–50%**")
            st.caption(", ".join(zone_media["16–50%"]) or "-")
        with cR:
            st.markdown(f"**51–80%**")
            st.caption(", ".join(zone_media["51–80%"]) or "-")
            st.markdown(f"**81–100%**")
            st.caption(", ".join(zone_media["81–100%"]) or "-")

