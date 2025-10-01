import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import rcParams
import gdown
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Apple SD Gothic Neo'  # 꼭 실제 이름 그대로!
mpl.rcParams['axes.unicode_minus'] = False

# ==== ⛽️ 드롭인 로딩 가속 블록 v2 (임포트 바로 아래에 붙이기) ====

# ==============================
# 전역 페이지/디자인 토큰 & 스타일 (3번 디자인만 이식)
# ==============================
st.set_page_config(page_title="어뷰징 대시보드", layout="wide")

ACCENT = "#7C83FF"
GRID   = "#EEF2FF"

st.markdown(
    f"""
<style>
/* 전체 레이아웃 폭/여백 (3번 스타일) */
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
  font-weight: 800;
  color: #111827;
}}
.hero p {{ margin: 0; color: #6B7280; font-size: 13px; }}

/* 섹션 리본 제목 */
.sec-row {{ display:flex; align-items:center; gap:12px; margin: 8px 0; }}
.sec-title {{
  display: inline-flex; align-items: center; gap: 10px;
  background: transparent; border: none; color: #0F172A;
  font-size: 18px; font-weight: 800; margin: 0;
}}
.sec-dot {{
  width: 6px; height: 22px; border-radius: 3px; background: {ACCENT};
  display: inline-block;
}}

/* 얇은 구분선 */
.sep {{ border:none; border-top:1px solid #E5E7EB; margin:10px 0 12px 0; }}

/* KPI/카드 톤 (3번 스타일 감성) */
.kpi-card {{
  border:1px solid #E5E7EB; border-radius:12px; padding:10px 12px; background:#FFFFFF;
}}
.kpi-title {{
  color:#6B7280; font-size:13px; font-weight:700; margin:0 0 6px 0;
}}
.kpi-value {{
  font-size:3.5rem; font-weight:900; color:#0F172A; margin:0;
}}
.unit-inline {{ font-size: 0.72em; color:#6B7280; margin-left:4px; font-weight:700; }}
.ratio-inline {{ font-size: 1.05rem; color:#6B7280; font-weight:700; margin-left:8px; }}

/* 상단 페이지 버튼(Overview/Details) */
.btn-row > div button {{
  border-radius: 10px;
  min-width: 200px;
  min-height: 45px;           /* ← 높이 미세 조정 */
  padding: 12px 18px;
  font-weight: 800;
  white-space: nowrap !important;
  line-height: 1.2 !important;
}}
/* Primary/Secondary 색상 (연보라) */
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

/* BaseWeb Select 줄바꿈/오버플로 해제 (3번 스타일) */
[data-baseweb="select"] {{ overflow: visible !important; }}
[data-baseweb="select"] > div {{
  padding: 2px 5px !important;
  height: auto !important;
  min-height: 0 !important;
  align-items: stretch !important;
}}
[data-baseweb="select"] div[role="combobox"] {{
  white-space: normal !important;
  overflow: visible !important;
}}
[data-baseweb="select"] div[role="combobox"] > div {{
  flex-wrap: wrap !important;
  max-width: none !important;
  row-gap: 2px !important;
}}
[data-baseweb="select"] div[role="combobox"] span,
[data-baseweb="select"] div[role="combobox"] div,
[data-baseweb="select"] [aria-live] {{
  white-space: normal !important;
  text-overflow: clip !important;
  overflow: visible !important;
  line-height: 1.4 !important;
}}
[data-baseweb="select"] input {{ height: auto !important; }}

/* Tooltip (CSS-only, 전역 다중 동작) */
.tip {{
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px; height: 22px;
  border-radius: 50%;
  background: #F3F4F6;
  color:#111827;
  font-size: 12px; font-weight: 900;
  cursor: default;
  user-select: none;
}}
.tip .tipbox {{
  visibility: hidden; opacity: 0;
  position: absolute; z-index: 1000;
  bottom: 130%; left: 50%; transform: translateX(-50%);
  width: 420px;
  background:#fff; color:#374151; text-align:left;
  border:1px solid #E5E7EB; border-radius:8px; padding:10px 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.12);
  transition: opacity .18s ease-in-out;
  line-height: 1.4;
}}
.tip:hover .tipbox {{ visibility: visible; opacity: 1; }}

/* Plotly 배경/그리드 */
.js-plotly-plot .plotly, .stPlotlyChart {{ background: transparent !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# Plotly 스타일 통일 유틸 (디자인만)
def tune_fig(fig, h=360, y_percent=False):
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
    if y_percent:
        fig.update_yaxes(automargin=True, showgrid=True, gridcolor=GRID, tickformat=".1%")
    else:
        fig.update_yaxes(automargin=True, showgrid=True, gridcolor=GRID)
    return fig

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
st.markdown("""
<style>
/* 숫자만 키우기 (부모) */
.kpi-card .kpi-value{
  font-size: 25px !important;   /* ← 숫자 크기만 조정 */
  font-weight: 900;
  line-height: 1;
}

/* 단위/비율은 고정 크기로 유지 (부모 변경과 무관) */
.kpi-card .kpi-value .unit-inline,
.kpi-card .kpi-value .ratio-inline{
  font-size: 17px !important;   /* ← 원/비율은 그대로(필요시 12~16px로 조절) */
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 2번 코드 로직: 데이터 로드/계산 (그대로 유지)
# ==============================

# list_1 = pd.read_parquet('광고목록_전처리.parquet')
# part = pd.read_parquet("part.parquet")
# point = pd.read_parquet("point.parquet")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# 공통 다운로드 + 로드 함수 (Parquet용)
@st.cache_data
def load_data(file_id: str, filename: str) -> pd.DataFrame:
    file_path = Path(BASE_DIR) / filename
    if not file_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(file_path), quiet=False)
    if not file_path.exists():
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없음: {file_path}")
    print(f"✅ 불러오는 파일: {file_path}")
    return pd.read_parquet(file_path, engine="pyarrow")

# CSV용 (이미 있는 코드 재사용)
@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(p)

# 예시: 데이터 로드
list_1 = load_data("1DXrhmL95OLYJ6_EIpOnf4LmPb5pDvvO3", "광고목록_전처리.parquet")
part   = load_data("1HsR5qstEd9A04yFu1lhz570DVQ3TDN7Q", "광고참여_어뷰징.parquet")
point  = load_data("1-sTUaLKCsqT0fPTXFwbp7yxyLnVjfead", "광고적립_어뷰징.parquet")

# 광고목록 데이터 드라이브 주소
# https://drive.google.com/file/d/1DXrhmL95OLYJ6_EIpOnf4LmPb5pDvvO3/view?usp=sharing

# 광고참여 데이터 드라이브 주소
# https://drive.google.com/file/d/1HsR5qstEd9A04yFu1lhz570DVQ3TDN7Q/view?usp=sharing

# 광고적립 데이터 드라이브 주소
# https://drive.google.com/file/d/1-sTUaLKCsqT0fPTXFwbp7yxyLnVjfead/view?usp=sharing


# ----- 지표 1 -----
df_counts_1 = (
    part.groupby(['dvc_idx', 'ads_idx', 'click_date'])
      .size()
      .reset_index(name='cnt')
)
df_abuse_1 = df_counts_1[(df_counts_1['cnt'] >= 2) & (df_counts_1['dvc_idx'] != 0)]
unique_users_1 = df_abuse_1['dvc_idx'].unique().tolist()

# ----- 지표 2 -----
point['click_date'] = pd.to_datetime(point['click_date'])
point['regdate'] = pd.to_datetime(point['regdate'])
point_2 = point.sort_values(by=['dvc_idx', 'click_date'])
point_2['prev_regdate'] = point_2.groupby('dvc_idx')['regdate'].shift(1)
point_2['time_diff'] = (point_2['click_date'] - point_2['prev_regdate']).dt.total_seconds()
df_fast = point_2[point_2['time_diff'] < 3]
df_fast = df_fast[df_fast['dvc_idx'] != 0]
unique_users_2 = df_fast['dvc_idx'].unique().tolist()

# ----- 지표 3 -----
list_3 = list_1[['ads_idx','ads_type']]
point_3 = point[['ads_idx','dvc_idx','ctit','click_date']]
df_3 = pd.merge(point_3, list_3, how='left',on='ads_idx')
df_3 = df_3[df_3['dvc_idx'] != 0]
df_3['click_date'] = pd.to_datetime(df_3['click_date'])
df_3['hour'] = df_3['click_date'].dt.hour
df_3['date'] = df_3['click_date'].dt.date
df_3_1  = df_3[df_3['ads_type'] == 1].copy()
df_3_5  = df_3[df_3['ads_type'] == 5].copy()
df_3_7  = df_3[df_3['ads_type'] == 7].copy()
df_3_11 = df_3[df_3['ads_type'] == 11].copy()

# 설치형(1)
df_3_ctit_mean = df_3_1.groupby('dvc_idx').agg(ctit_mean=('ctit','mean'))
df_3_1 = pd.merge(df_3_1, df_3_ctit_mean, how='left',on='dvc_idx').drop_duplicates(subset='dvc_idx',keep='first')
df_3_1['label'] = df_3_1['ctit_mean'].apply(lambda x: '의심' if x < 10 else '정상')
type1_users_ctit = df_3_1[df_3_1['label'] == '의심']['dvc_idx'].unique().tolist()

# 페북(5)
df_3_ctit_mean = df_3_5.groupby('dvc_idx').agg(ctit_mean=('ctit','mean'))
df_3_5 = pd.merge(df_3_5, df_3_ctit_mean, how='left',on='dvc_idx').drop_duplicates(subset='dvc_idx',keep='first')
q1 = np.percentile(df_3_5['ctit_mean'], 25); q3 = np.percentile(df_3_5['ctit_mean'], 75)
iqr = q3 - q1; bound_기준 = q1 - (1.5 * iqr)
df_3_5['type5_의심'] = (df_3_5['ctit_mean'] < bound_기준)
df_3_5['label'] = '정상'; df_3_5.loc[df_3_5['type5_의심'], 'label'] = '의심'
type5_users_ctit = df_3_5[df_3_5['label'] == '의심']['dvc_idx'].unique().tolist()

# 인스타(7)
df_3_ctit_mean = df_3_7.groupby('dvc_idx').agg(ctit_mean=('ctit','mean'))
df_3_7 = pd.merge(df_3_7, df_3_ctit_mean, how='left',on='dvc_idx').drop_duplicates(subset='dvc_idx',keep='first')
q1 = np.percentile(df_3_7['ctit_mean'], 25); q3 = np.percentile(df_3_7['ctit_mean'], 75)
iqr = q3 - q1; bound_기준 = q1 - (1.5 * iqr)
df_3_7['type7_의심'] = (df_3_7['ctit_mean'] < bound_기준)
df_3_7['label'] = '정상'; df_3_7.loc[df_3_7['type7_의심'], 'label'] = '의심'
type7_users_ctit = df_3_7[df_3_7['label'] == '의심']['dvc_idx'].unique().tolist()

# 네이버(11)
df_3_ctit_mean = df_3_11.groupby('dvc_idx').agg(ctit_mean=('ctit','mean'))
df_3_11 = pd.merge(df_3_11, df_3_ctit_mean, how='left',on='dvc_idx').drop_duplicates(subset='dvc_idx',keep='first')
q1 = np.percentile(df_3_11['ctit_mean'], 25); q3 = np.percentile(df_3_11['ctit_mean'], 75)
iqr = q3 - q1; bound_기준 = q1 - (1.5 * iqr)
df_3_11['type11_의심'] = (df_3_11['ctit_mean'] < bound_기준)
df_3_11['label'] = '정상'; df_3_11.loc[df_3_11['type11_의심'], 'label'] = '의심'
type11_users_ctit = df_3_11[df_3_11['label'] == '의심']['dvc_idx'].unique().tolist()

all_users_3 = type1_users_ctit + type5_users_ctit + type7_users_ctit + type11_users_ctit
unique_users_3 = list(set(all_users_3))

# ----- 지표 4 -----
list_4 = list_1[['ads_idx','ads_rejoin_type']]
part_4 = part[['ads_idx','dvc_idx','click_date']]
df_4 = pd.merge(part_4, list_4, how='left',on='ads_idx')
df_4 = df_4[df_4['dvc_idx'] != 0]
df_4 = df_4.dropna(axis=0)
df_4['click_date'] = pd.to_datetime(df_4['click_date'])
df_4['date'] = df_4['click_date'].dt.date
df_4_dvc = (df_4[df_4['ads_rejoin_type'] != 'REJOINABLE']
            .groupby(['dvc_idx','ads_idx','date']).size()
            .reset_index(name='count').query('count >= 3'))
df_4_dvc1 = df_4_dvc['dvc_idx'].unique().tolist()
df_4_1 = (df_4[df_4['ads_rejoin_type'] == 'REJOINABLE']
          .groupby(['dvc_idx','ads_idx']).size()
          .reset_index(name='count').query('count > 1'))
df_4_1 = pd.merge(df_4_1, df_4, how='left', on=['dvc_idx','ads_idx']).sort_values(by=['dvc_idx','ads_idx','click_date'])
df_4_1['time_diff'] = df_4_1.groupby(['dvc_idx', 'ads_idx'])['click_date'].diff().dt.total_seconds()
fast_repeat = df_4_1[df_4_1['time_diff'] <= 2]['dvc_idx'].unique()
unique_users_4 = list(set(list(df_4_dvc1) + list(fast_repeat)))

# ----- 지표 5 -----
list_5 = list_1[['ads_idx','ads_type']]
part_5 = part[['ads_idx','dvc_idx','reward_price','contract_price']]
df_5 = pd.merge(part_5, list_5, how='left', on='ads_idx')
df_5 = df_5[df_5['dvc_idx'] != 0]
df_5['reward_ratio'] = np.where(df_5['contract_price'] == 0, df_5['reward_price'], df_5['reward_price'] / df_5['contract_price'])
reward_ratio_10 = df_5['reward_ratio'].quantile(0.9)
df_reward_ratio_10 = df_5[df_5['reward_ratio'] >= reward_ratio_10].round(2).reset_index(drop=True)
top10 = df_reward_ratio_10['ads_idx'].unique()
part_cnt = df_reward_ratio_10.groupby('dvc_idx')['ads_idx'].nunique().reset_index(name='count')
part_cnt['part_ratio'] = part_cnt['count'] / top10.size
part_cnt = part_cnt.sort_values(by='part_ratio', ascending=False).reset_index(drop=True)
user_10 = part_cnt['part_ratio'].quantile(0.9)
unique_users_5 = part_cnt[part_cnt['part_ratio'] >= user_10]['dvc_idx'].unique()

lists = {
    "unique_users_1": list(unique_users_1),
    "unique_users_2": list(unique_users_2),
    "unique_users_3": list(unique_users_3),
    "unique_users_4": list(unique_users_4),
    "unique_users_5": list(unique_users_5)
}
all_users = []
for name, lst in lists.items():
    for user in lst:
        all_users.append((user, name))
df_users = pd.DataFrame(all_users, columns=["dvc_idx", "source"])
df_counts = df_users.groupby("dvc_idx").size().reset_index(name="count")
df_cnt = df_counts[~df_counts['count'].isin([1, 2])]
abuse_dvc = df_cnt['dvc_idx'].unique()

# ----- point_merged -----
point_merged = point[['ads_idx','mda_idx','dvc_idx','rwd_cost','regdate']].merge(
    list_1[['ads_idx', 'ads_type','ads_category']], on='ads_idx', how='left'
)
point_merged['regdate'] = pd.to_datetime(point_merged['regdate'])
point_merged['hour'] = point_merged['regdate'].dt.hour
point_merged['combo'] = point_merged['ads_type'].astype(str) + "-" + point_merged['mda_idx'].astype(str)
point_merged['group'] = point_merged['dvc_idx'].isin(abuse_dvc).map({True:'abuse', False:'normal'})
point_merged["weekday_name"] = point_merged["regdate"].dt.day_name()

# ==============================
# 헤더(히어로)
# ==============================
st.markdown(
    """
<div class="hero">
  <h1>Abusing</h1>
  <p>어뷰징 의심 참여/리워드 규모를 집계하고, 패턴을 탐색합니다. Top10 조합, KPI·손실액 시뮬레이션, 조합별 세부표를 제공합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ==============================
# 상단 메뉴(Overview / Details) — 3번식 버튼 UI만 적용
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "Overview"

st.markdown('<div class="btn-row">', unsafe_allow_html=True)
bL, bR, _sp = st.columns([1,1,8], gap="small")
with bL:
    if st.button(
        "Overview",
        type=("primary" if st.session_state.page == "Overview" else "secondary"),
        key="btn_page_overview"
    ):
        st.session_state.page = "Overview"
        st.rerun()
with bR:
    if st.button(
        "Details",
        type=("primary" if st.session_state.page == "Details" else "secondary"),
        key="btn_page_details"
    ):
        st.session_state.page = "Details"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page

# ==============================
# Overview 페이지 (2번 로직 그대로, UI만 다듬음)
# ==============================
if page == "Overview":
    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # ---- 전체 지표 ----
    st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>전체 지표</div></div>', unsafe_allow_html=True)

    abuse_dvc_set = set(df_cnt["dvc_idx"].unique())
    merged_abuse = point_merged[point_merged["dvc_idx"].isin(abuse_dvc_set)].copy()

    total_rwd = point["rwd_cost"].sum()
    abuse_rwd = merged_abuse["rwd_cost"].sum()
    total_cnt = point['rwd_cost'].count()
    abuse_cnt = merged_abuse['rwd_cost'].count()
    abuse_cnt_ratio = abuse_cnt / total_cnt if total_cnt else 0
    abuse_count = len(merged_abuse)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>전체 어뷰징 건수(적립)</div><p class='kpi-value'>{abuse_count:,.0f}</p></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>전체 리워드 총액</div><p class='kpi-value'>{total_rwd:,.0f}<span class='unit-inline'>원</span></p></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>어뷰징 리워드 총액</div><p class='kpi-value'>{abuse_rwd:,.0f}<span class='unit-inline'>원</span></p></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>어뷰징 참여 비율</div><p class='kpi-value'>{abuse_cnt_ratio:.1%}</p></div>", unsafe_allow_html=True)

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # ---- 어뷰징 유저 수 ----
    st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>어뷰징 유저 수</div></div>', unsafe_allow_html=True)

    total_users = point["dvc_idx"].nunique()
    users_with_any_abuse = df_counts["dvc_idx"].nunique()
    final_abuse_users = df_cnt['dvc_idx'].nunique()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>전체 유저 수</div><p class='kpi-value'>{total_users:,}</p></div>", unsafe_allow_html=True)
    with c2:
        ratio = (users_with_any_abuse/total_users) if total_users else 0
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>지표 하나 이상 해당 유저 수</div><p class='kpi-value'>{users_with_any_abuse:,}<span class='ratio-inline'>{ratio:.1%}</span></p></div>", unsafe_allow_html=True)
    with c3:
        ratio2 = (final_abuse_users/total_users) if total_users else 0
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>최종 어뷰징 유저 수</div><p class='kpi-value'>{final_abuse_users:,}<span class='ratio-inline'>{ratio2:.1%}</span></p></div>", unsafe_allow_html=True)

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # 어뷰징 적립액 Top10 (여기에 '필터'를 소제목 없이 배치)
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>어뷰징 적립액 Top10</div></div>', unsafe_allow_html=True)

    # 안내문구 + ❓툴팁 (동작)
    info_l, info_r = st.columns([0.85, 0.15])
    with info_l:
        st.markdown("<div style='color:#6B7280; font-weight:800; margin:0 0 6px 2px;'>단일선택, 최대 3개</div>", unsafe_allow_html=True)
    with info_r:
        st.markdown(
            """
            <div style="display:flex; justify-content:flex-end;">
              <div class="tip">?
                <div class="tipbox">
                  <div style="font-weight:800; font-size:13px; color:#111827; margin-bottom:6px;">코드 안내</div>
                  <div style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px;">
                    <b>광고유형</b><br>
                    1: 설치형, 2: 실행형, 3: 참여형, 4: 클릭형, 5: 페북,
                    6: 트위터, 7: 인스타, 8: 노출형, 9: 퀘스트, 10: 유튜브, 11: 네이버, 12: CPS
                    <br><br>
                    <b>광고카테고리</b><br>
                    0: 선택안함, 1: 앱, 3: 구독, 4: 간편미션-퀴즈, 13: 간편미션, 2: 경험하기,
                    5: 게임(CPA), 6: 멀티보상, 8: 무료참여, 7: 금융, 10: 유료참여, 11: 쇼핑-상품, 12: 제휴몰
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 필터 컨트롤 (단일 선택 · 최대 3개)
    ads_type_opts = ["전체"] + sorted(point_merged["ads_type"].astype(str).unique().tolist(), key=lambda x: int(x))
    mda_opts      = ["전체"] + sorted(point_merged["mda_idx"].astype(int).unique().tolist())
    cat_opts      = ["전체"] + sorted(point_merged["ads_category"].astype(int).unique().tolist())
    hour_opts     = ["전체"] + list(range(24))

    f1, f2, f3, f4 = st.columns(4)
    ads_type_sel = f1.selectbox("광고유형", ads_type_opts, index=0)
    mda_sel      = f2.selectbox("매체사", mda_opts, index=0)
    cat_sel      = f3.selectbox("카테고리", cat_opts, index=0)
    hour_sel     = f4.selectbox("시간대", hour_opts, index=0)

    chosen_cnt = sum([ads_type_sel!="전체", mda_sel!="전체", cat_sel!="전체", hour_sel!="전체"])
    if chosen_cnt > 3:
        st.error("필터는 최대 3개까지만 선택 가능합니다.")
        st.stop()

    filtered_overview = point_merged.copy()
    if ads_type_sel != "전체":
        filtered_overview = filtered_overview[filtered_overview["ads_type"] == int(ads_type_sel)]
    if mda_sel != "전체":
        filtered_overview = filtered_overview[filtered_overview["mda_idx"] == int(mda_sel)]
    if cat_sel != "전체":
        filtered_overview = filtered_overview[filtered_overview["ads_category"] == int(cat_sel)]
    if hour_sel != "전체":
        filtered_overview = filtered_overview[filtered_overview["hour"] == int(hour_sel)]

    # ---- Top10 바차트 ----
    combo_df = (
        filtered_overview[filtered_overview['group'] == 'abuse']
        .groupby(["ads_type","mda_idx","ads_category","hour"], as_index=False)["rwd_cost"]
        .sum().rename(columns={"rwd_cost":"abuse_rwd_sum"})
    )
    combo_df["combo"] = (combo_df["ads_type"].astype(str)+"-"
                         +combo_df["mda_idx"].astype(str)+"-"
                         +combo_df["ads_category"].astype(str)+"-"
                         +combo_df["hour"].astype(str))
    combo_top10 = combo_df.sort_values("abuse_rwd_sum", ascending=False).head(10)

    if combo_top10.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
    else:
        fig_bar = px.bar(combo_top10, x="combo", y="abuse_rwd_sum", text_auto='.2s')
        fig_bar.update_traces(marker_color=ACCENT, marker_line_color=ACCENT)
        fig_bar.update_layout(xaxis_title="광고유형-매체사-카테고리-시간대", yaxis_title="어뷰징 적립액 합계")
        st.plotly_chart(tune_fig(fig_bar, 420, y_percent=False), use_container_width=True)

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # ---- 요일-시간대 히트맵 ----
    st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>요일-시간대별 어뷰징 적립 패턴</div></div>', unsafe_allow_html=True)

    heatmap_df = (
        filtered_overview[filtered_overview['group'] == 'abuse']
        .groupby(["weekday_name","hour"], as_index=False)["rwd_cost"]
        .sum().rename(columns={"rwd_cost":"abuse_rwd_sum"})
    )
    if heatmap_df.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
    else:
        weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heatmap_df["weekday_name"] = pd.Categorical(heatmap_df["weekday_name"], categories=weekday_order, ordered=True)
        heatmap_df = heatmap_df.sort_values(["weekday_name","hour"])
        fig_heat = px.density_heatmap(heatmap_df, x="hour", y="weekday_name", z="abuse_rwd_sum",
                                      color_continuous_scale=["#F5F4FF","#DEDBFF","#BFB9FF","#9D98FF","#7C83FF"])
        fig_heat.update_layout(xaxis_title="시간대", yaxis_title="요일")
        st.plotly_chart(tune_fig(fig_heat, 420, y_percent=False), use_container_width=True)

# ==============================
# Details 페이지 (2번 로직 그대로, UI만 다듬음)
# ==============================
else:
    # 위험도 분류 리스트
    high_list   = ['12-270', '2-817', '11-270','4-854']
    medium_list = ['3-563', '9-270', '3-58']
    low_list    = ['3-87', '3-294', '3-482']

    pm = point_merged.copy()
    pm['risk classification'] = 'Normal'
    pm.loc[pm['combo'].isin(high_list), 'risk classification'] = 'High'
    pm.loc[pm['combo'].isin(medium_list), 'risk classification'] = 'Medium'
    pm.loc[pm['combo'].isin(low_list), 'risk classification'] = 'Low'

    # 위험도 표(토글)
    sample_df = pd.DataFrame({
        "콤보 (광고유형-매체사)": [
            "12-270 (46점)", "2-817 (29점)", "11-270 (18점)", "4-854 (7점)",
            "3-563", "9-270", "3-58", "3-87", "3-294", "3-482"
        ],
        "카테고리-요일-시간대": [
            "11-토-17","2-일-22","3-금-6,17,18 / 3-토-7, 3-목-18","13-일-18",
            "4-일-0","6-월-12","7-일-18, 7-월-19","8-목-17","8-금-3","8-수-20"
        ],
        "조심구간 (러프)": [
            "11-토-오후","2-일-저녁","3-금-오전,오후,저녁 / 3-토-오전 / 3-목-저녁",
            "13-일-저녁","4-일-새벽","6-월-오후","7-일-저녁 / 7-월-저녁","8-목-오후","8-금-새벽","8-수-저녁"
        ],
        "Risk Level": [
            "High","High","High","High","Medium","Medium","Medium","Low","Low","Low"
        ]
    })

    def highlight_row(row):
        if row["Risk Level"] == "High":
            return ['background-color: #f8d7da']*len(row)
        elif row["Risk Level"] == "Medium":
            return ['background-color: #fff3cd']*len(row)
        elif row["Risk Level"] == "Low":
            return ['background-color: #d4edda']*len(row)
        else:
            return ['']*len(row)

    styled_df = sample_df.style.apply(highlight_row, axis=1)

    if "show_popup" not in st.session_state:
        st.session_state.show_popup = False

    rowL, rowR = st.columns([8.6, 1.4])
    with rowL:
        st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>위험도 분류 결과표</div></div>', unsafe_allow_html=True)
    with rowR:
        if st.button("결과표 보기", key="toggle_button"):
            st.session_state.show_popup = not st.session_state.show_popup
            st.rerun()

    if st.session_state.show_popup:
        st.dataframe(styled_df, use_container_width=True)

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # KPI
    c_headL, c_headR = st.columns([8, 2])
    with c_headL:
        st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>KPI</div></div>', unsafe_allow_html=True)
    with c_headR:
        risk_level = st.selectbox("리스크 레벨 선택", ["High", "Medium", "Low"])

    selected_risk = risk_level
    filtered = pm.loc[pm['risk classification'].eq(selected_risk)].copy()
    abuse = filtered.loc[filtered['group'].eq('abuse')]
    normal = filtered.loc[filtered['group'].eq('normal')]

    reward_sum = abuse['rwd_cost'].sum()
    reward_cnt = abuse['rwd_cost'].count()
    total_count = filtered['rwd_cost'].count()
    abuse_count = abuse['rwd_cost'].count()
    abuse_ratio = ((abuse_count / total_count) * 100).round(2) if total_count else 0
    abuse_avg_reward = abuse['rwd_cost'].mean()
    normal_count = normal['rwd_cost'].count() or 1
    loss_amount = (abuse_avg_reward * (abuse_count / normal_count)).round(2) if pd.notna(abuse_avg_reward) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>어뷰징 리워드 총액</div><p class='kpi-value'>{reward_sum:,.0f}<span class='unit-inline'>원</span></p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>어뷰징 적립 건수</div><p class='kpi-value'>{reward_cnt:,.0f}<span class='unit-inline'>건</span></p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>전체 대비 어뷰징 참여비율</div><p class='kpi-value'>{abuse_ratio:.2f}%</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>손실액 (러프)</div><p class='kpi-value'>{loss_amount:,.0f}<span class='unit-inline'>원</span></p></div>", unsafe_allow_html=True)

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # 카테고리/유형/매체사 분포 (+ 작동하는 ❓툴팁)
    ads_category_unique = filtered['ads_category'].unique()
    ads_type_unique     = filtered['ads_type'].unique()
    mda_idx_unique      = filtered['mda_idx'].unique()
    ads_category_text = " · ".join(map(str, ads_category_unique))
    ads_type_text     = " · ".join(map(str, ads_type_unique))
    mda_idx_text      = " · ".join(map(str, mda_idx_unique))

    rowL2, rowR2 = st.columns([0.9, 0.1])
    with rowL2:
        st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>카테고리 / 광고유형 / 매체사 분포</div></div>', unsafe_allow_html=True)
    with rowR2:
        st.markdown(
            """
            <div style="display:flex;justify-content:flex-end;align-items:center;">
              <div class="tip">?
                <div class="tipbox">
                  <div style="font-weight:800; font-size:13px; color:#111827; margin-bottom:6px;">코드 안내</div>
                  <div style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px;">
                    <b>광고유형</b><br>
                    1: 설치형, 2: 실행형, 3: 참여형, 4: 클릭형, 5: 페북,
                    6: 트위터, 7: 인스타, 8: 노출형, 9: 퀘스트, 10: 유튜브, 11: 네이버, 12: CPS
                    <br><br>
                    <b>광고카테고리</b><br>
                    0: 선택안함, 1: 앱, 3: 구독, 4: 간편미션-퀴즈, 13: 간편미션, 2: 경험하기,
                    5: 게임(CPA), 6: 멀티보상, 8: 무료참여, 7: 금융, 10: 유료참여, 11: 쇼핑-상품, 12: 제휴몰
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(f"<p style='text-align:center; font-size:16px;'>카테고리</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:22px; font-weight:700; color:#ef4444;'>{ads_category_text}</p>", unsafe_allow_html=True)
    with cc2:
        st.markdown(f"<p style='text-align:center; font-size:16px;'>광고유형</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:22px; font-weight:700; color:#f59e0b;'>{ads_type_text}</p>", unsafe_allow_html=True)
    with cc3:
        st.markdown(f"<p style='text-align:center; font-size:16px;'>매체사</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:22px; font-weight:700; color:#3b82f6;'>{mda_idx_text}</p>", unsafe_allow_html=True)

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # 시간대 분포: 주차 선택 + 라인/히트맵(2번 로직 그대로)
    hdL, hdR = st.columns([0.7, 0.3])
    with hdL:
        st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>시간대 분포</div></div>', unsafe_allow_html=True)
    with hdR:
        week_level = st.selectbox(
            "주차 선택",
            ["7월 4주차", "7월 5주차", "8월 1주차", "8월 2주차", "8월 3주차", "8월 4주차","8월 5주차"],
            key="week_sel"
        )

    # 주차 파생
    filtered["month"] = filtered["regdate"].dt.month
    filtered["first_day"] = filtered["regdate"].values.astype("datetime64[M]")
    first_day = pd.to_datetime(filtered["first_day"])
    first_weekday = (first_day.dt.weekday + 1) % 7
    filtered["first_weekday"] = first_weekday
    filtered["week_of_month"] = ((filtered["regdate"].dt.day + filtered["first_weekday"] - 1) // 7) + 1
    filtered["month_week"] = filtered["month"].astype(str) + "월 " +  filtered["week_of_month"].astype(str) + "주차"


    selected_week = week_level
    filtered_2 = filtered[filtered['month_week'] == selected_week]

    left_g, right_g = st.columns(2)

    with left_g:
        daily_abuse = (filtered_2.groupby(filtered_2['regdate'].dt.date)['rwd_cost']
                       .sum().reset_index())
        daily_abuse.columns = ['date', 'reward_sum']
        weekday_map = {0:"Monday", 1:"Turesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}
        daily_abuse['label'] = daily_abuse['date'].apply(lambda x: x.strftime("%m-%d") + f" ({weekday_map[x.weekday()]})")

        fig_line, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(daily_abuse['label'], daily_abuse['reward_sum'], marker='o')
        ax1.set_title(f"Abuse Payout", fontsize=12)
        ax1.set_xlabel("Day(Weekday)"); ax1.set_ylabel("Payout"); ax1.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_line)

    with right_g:
        heatmap_df = filtered_2.copy()
        heatmap_df['weekday'] = (heatmap_df['regdate'].dt.weekday + 1) % 7
        heatmap_df['hour']    = heatmap_df['regdate'].dt.hour
        heatmap_data = heatmap_df.groupby(['weekday','hour'])['rwd_cost'].sum().reset_index()
        total_sum = heatmap_data['rwd_cost'].sum() or 1
        heatmap_data['ratio'] = heatmap_data['rwd_cost'] / total_sum
        pivot = heatmap_data.pivot(index='weekday', columns='hour', values='ratio').reindex(
            index=range(7), columns=range(24), fill_value=0
        )

        # 👉 히트맵 "색깔만" 2번 코드의 라벤더 팔레트로 교체
        from matplotlib.colors import LinearSegmentedColormap
        lav_cmap = LinearSegmentedColormap.from_list(
            "lavender_boost",
            ["#F5F4FF","#DEDBFF","#BFB9FF","#9D98FF","#7C83FF"]
        )

        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.heatmap(
            pivot, cmap=lav_cmap, ax=ax2, cbar_kws={'label': 'Ratio'},
            vmin=0, vmax=max(0.0001, float(pivot.values.max()))
        )
        ax2.set_yticks(range(7))
        ax2.set_yticklabels(
            ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], 
            rotation=0
        )
        ax2.set_title(f"Weekday × Hour Abuse Payout Ratio", fontsize=12)
        ax2.set_xlabel("Hour (0–23)")
        ax2.set_ylabel("Weekday")
        st.pyplot(fig2)

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # 손실액 시뮬레이터 (2번 로직)
    st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>어뷰징 줄일 시, 손실액 차감</div></div>', unsafe_allow_html=True)

    def calculate_loss(df):
        abuse_vals  = df.loc[df['group'].eq('abuse'),  'rwd_cost']
        normal_vals = df.loc[df['group'].eq('normal'), 'rwd_cost']
        if abuse_vals.empty or normal_vals.empty:
            return 0.0
        return abuse_vals.mean() * (len(abuse_vals) / max(len(normal_vals), 1))

    top_col1, top_col2 = st.columns([2, 0.7])
    with top_col2:
        reduction_str = st.text_input("어뷰징 감소율 입력 (%)", value="0")
        try:
            reduction_pct = float(reduction_str)
        except Exception:
            reduction_pct = 0.0

    df_risk = pm.loc[pm['risk classification'].eq(selected_risk)]
    original_loss  = calculate_loss(df_risk)
    new_loss       = max(0.0, original_loss * (1 - reduction_pct / 100.0))
    reduced_amount = max(0.0, original_loss - new_loss)

    gL, gR = st.columns(2)
    with gL:
        fig_bar_narrow = px.bar(pd.DataFrame({"시나리오":[f"{int(reduction_pct)}% 감소 후"], "손실액":[new_loss]}),
                                x="시나리오", y="손실액", text="손실액")
        fig_bar_narrow.update_traces(width=0.35, texttemplate="%{text:,.0f}", textposition="outside",
                                     marker_color=ACCENT, marker_line_color=ACCENT)
        fig_bar_narrow.update_layout(yaxis_title="손실액", xaxis_title="", bargap=0.6)
        st.plotly_chart(tune_fig(fig_bar_narrow, 320), use_container_width=True)
    with gR:
        st.markdown(
            f"""
            <div style="margin-top:6px;">
              <div style="font-size:44px; font-weight:900; color:#0F172A; display:flex; align-items:baseline; gap:8px;">
                <span>{new_loss:,.0f}</span><span class="unit-inline" style="font-size:28px;">원</span>
              </div>
              <div style="font-size:28px; color:#2563eb; font-weight:900; margin-top:6px;">
                ▼ {reduced_amount:,.0f} 절약
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<hr class="sep" />', unsafe_allow_html=True)

    # 세부 테이블 (2번 로직)
    st.markdown('<div class="sec-row"><div class="sec-title"><span class="sec-dot"></span>세부 테이블</div></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.7, 0.7, 0.5])
    with col3:
        combo_map = {"High": high_list, "Medium": medium_list, "Low": low_list}
        combo_options = combo_map[selected_risk]
        selected_combo = st.selectbox("(광고유형-매체사) 조합 선택", combo_options, key="combo_select")

    filtered_3 = filtered.loc[filtered['combo'].astype(str).str.strip().eq(selected_combo)]
    if filtered_3.empty:
        st.warning("선택한 조합에 대한 데이터가 없습니다.")
    else:
        ads_type = filtered_3['ads_type'].iloc[0]
        mda_idx  = filtered_3['mda_idx'].iloc[0]
        abuse_df = filtered_3.loc[filtered_3['group'].eq('abuse')]

        result = (abuse_df
                  .assign(weekday=filtered_3['regdate'].dt.weekday)
                  .assign(weekday_name=filtered_3['regdate'].dt.weekday.map({0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}))
                  .groupby(['ads_category','weekday_name','hour'])
                  .agg(reward_sum=('rwd_cost','sum'))
                  .reset_index()
                  .sort_values('reward_sum', ascending=False))
        top_row = result.iloc[0] if not result.empty else None
        top_category = top_row['ads_category'] if top_row is not None else "-"
        top_weekday  = top_row['weekday_name'] if top_row is not None else "-"
        top_hour     = int(top_row['hour']) if top_row is not None else "-"

        total_count = len(filtered_3)
        abuse_count2 = (filtered_3['group'] == 'abuse').sum()
        normal_count2 = (filtered_3['group'] == 'normal').sum()
        normal_ratio = round((normal_count2 / total_count) * 100, 1) if total_count else 0
        abuse_ratio2  = round((abuse_count2  / total_count) * 100, 1) if total_count else 0

        normal_rwd_mean = filtered_3.loc[filtered_3['group'].eq('normal'), 'rwd_cost'].mean()
        abuse_rwd_mean  = filtered_3.loc[filtered_3['group'].eq('abuse'),  'rwd_cost'].mean()
        abuse_rwd_sum   = filtered_3.loc[filtered_3['group'].eq('abuse'),  'rwd_cost'].sum()

        abuse_rwd_mean  = 0 if pd.isna(abuse_rwd_mean)  else round(abuse_rwd_mean,  0)
        normal_rwd_mean = 0 if pd.isna(normal_rwd_mean) else round(normal_rwd_mean, 0)
        abuse_rwd_sum   = 0 if pd.isna(abuse_rwd_sum)   else int(abuse_rwd_sum)

        table_html = pd.DataFrame({
            "광고 유형":[ads_type],
            "매체사":[mda_idx],
            "카테고리":[top_category],
            "요일":[top_weekday],
            "시간대":[top_hour],
            "정상 참여 비율":[f"{normal_ratio}%"],
            "어뷰징 참여 비율":[f"{abuse_ratio2}%"],
            "정상 평균 리워드":[f"{normal_rwd_mean:,.0f}"],
            "어뷰징 평균 리워드":[f"{abuse_rwd_mean:,.0f}"],
            "어뷰징 리워드 총액":[f"{abuse_rwd_sum:,.0f}"],
        }).to_html(index=False)
        st.markdown(f"<div style='margin-top:10px;'>{table_html}</div>", unsafe_allow_html=True)
