import sys, pathlib, streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import gdown

# ─────────────────────────────────────────────────────────────────────────────
# 0) 이 파일에서만 set_page_config 1회 호출 (원본 파일들은 손대지 않음)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Integrated Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1) 원본 파일 경로 (내용 절대 수정 X)
# ─────────────────────────────────────────────────────────────────────────────
TREND_FILE   = "최종 프로젝트 스트림릿/app2.py"     # 1번(Trend)
ABUSING_FILE = "최종 프로젝트 스트림릿/abuse.py"     # 3번(Abusing)
PICKS_FILE   = "최종 프로젝트 스트림릿/streamlit_app.py"  # ← 이 파일이 실제로 있으면 그대로,
                                                         # 없으면 존재하는 파일명으로 교체

# ─────────────────────────────────────────────────────────────────────────────
# 2) 사이드바 스타일
# ─────────────────────────────────────────────────────────────────────────────
SIDEBAR_CSS = """
<style>
:root { --sbw: 180px; --lav:#7C83FF; --lav2:#EEF0FF; --ink:#0F172A; }

[data-testid="stSidebar"]{ min-width:var(--sbw)!important; width:var(--sbw)!important; }
[data-testid="stSidebar"] .block-container{ padding:10px 8px !important; }

.sb-brand{ font-weight:900; font-size:14px; color:var(--ink);
           letter-spacing:.4px; margin:4px 0 10px 6px; }
.sb-menu{ display:flex; flex-direction:column; gap:8px; }

/* 카드형 버튼 (버튼만, 아이콘/링크 없음) */
[data-testid="stSidebar"] .stButton > button{
  width:100% !important;
  border-radius:12px !important;
  padding:10px 12px !important;
  border:1px solid #E5E7EB !important;
  background:#FFFFFF !important;
  color:#111827 !important;
  font-weight:800 !important;
  text-align:left !important;
}

/* 선택/비선택 톤 */
[data-testid="stSidebar"] .stButton > button[kind="primary"]{
  background:var(--lav) !important; border-color:var(--lav) !important; color:#FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]{
  background:#F6F7FF !important; color:#1f2937 !important;
}
</style>
"""
st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3) 현재 탭(nav) 상태
# ─────────────────────────────────────────────────────────────────────────────
if "__nav" not in st.session_state:
    st.session_state["__nav"] = "Trend"
nav = st.session_state["__nav"]

def _safe_rerun():
    """버전별 rerun 호환"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 4) 사이드바 메뉴 렌더(버튼만, 카드형)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">MENU</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-menu">', unsafe_allow_html=True)

    def card_btn(label: str):
        active = (st.session_state["__nav"] == label)
        clicked = st.button(
            label,
            key=f"__nav_{label}",
            type=("primary" if active else "secondary"),
            use_container_width=True,
        )
        if clicked and not active:
            st.session_state["__nav"] = label
            _safe_rerun()

    # 순서: Trend / Abusing / Picks(=추천 자리)
    card_btn("Trend")
    card_btn("Abusing")
    card_btn("Picks")   # 추천 시스템 자리

    st.markdown('</div>', unsafe_allow_html=True)

# 최신 상태 다시 읽기
nav = st.session_state["__nav"]

# ─────────────────────────────────────────────────────────────────────────────
# 5) 원본 스크립트를 '그대로' 실행 (set_page_config만 임시 무력화)
# ─────────────────────────────────────────────────────────────────────────────
def run_original_streamlit_script(file_path: str):
    p = pathlib.Path(file_path)
    if not p.exists():
        st.error(f"파일을 찾을 수 없습니다: {p.resolve()}")
        return

    try:
        src = p.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return

    # 중복 set_page_config 방지
    _orig_spc = st.set_page_config
    st.set_page_config = (lambda *a, **k: None)
    try:
        ns = {}
        exec(src, ns, ns)   # ← 원본 그대로 실행 (수정/삭제/추가 없음)
    except Exception:
        st.error("원본 코드 실행 중 오류가 발생했습니다.")
        st.exception(sys.exc_info())
    finally:
        st.set_page_config = _orig_spc

# ─────────────────────────────────────────────────────────────────────────────
# 6) 라우팅
# ─────────────────────────────────────────────────────────────────────────────
if nav == "Trend":
    run_original_streamlit_script(TREND_FILE)
elif nav == "Abusing":
    run_original_streamlit_script(ABUSING_FILE)
else:
    # st.title("Picks")
    # st.info("추천 시스템은 여기로 붙일 예정입니다. (연결만 준비)")
    run_original_streamlit_script(PICKS_FILE)
