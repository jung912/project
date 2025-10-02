# -*- coding: utf-8 -*-
from pathlib import Path
import sys, traceback
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

# 레포 루트(= /mount/src/project) 계산: 이 파일은 '최종 프로젝트 스트림릿/app.py'에 있으므로 parents[1]
ROOT = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# 1) 원본 파일 경로 (레포 루트 기준 상대경로 문자열)
# ─────────────────────────────────────────────────────────────────────────────
TREND_FILE   = "최종 프로젝트 스트림릿/app2.py"          # 1번(Trend)
ABUSING_FILE = "최종 프로젝트 스트림릿/abuse.py"          # 3번(Abusing)
PICKS_FILE   = "최종 프로젝트 스트림릿/streamlit_app.py"  # 실제 없으면 자동 숨김

def _abs(rel: str) -> Path:
    return (ROOT / rel).resolve()

def _exists(rel: str) -> bool:
    try:
        return _abs(rel).exists()
    except Exception:
        return False

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
[data-testid="stSidebar"] .stButton > button{
  width:100% !important; border-radius:12px !important;
  padding:10px 12px !important; border:1px solid #E5E7EB !important;
  background:#FFFFFF !important; color:#111827 !important;
  font-weight:800 !important; text-align:left !important;
}
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
AVAILABLE = [("Trend", TREND_FILE), ("Abusing", ABUSING_FILE)]
if _exists(PICKS_FILE):
    AVAILABLE.append(("Picks", PICKS_FILE))  # 파일이 있을 때만 노출

if "__nav" not in st.session_state:
    st.session_state["__nav"] = AVAILABLE[0][0]  # 첫 메뉴로 초기화

def _safe_rerun():
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

    for label, _rel in AVAILABLE:
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

    st.markdown('</div>', unsafe_allow_html=True)

nav = st.session_state["__nav"]

# ─────────────────────────────────────────────────────────────────────────────
# 5) 원본 스크립트를 '그대로' 실행하되, 실행 환경을 실제 파일처럼 맞춤
#    - __file__/__name__/sys.path 주입
#    - 중복 set_page_config 무력화
#    - 예외는 화면+로그에 스택트레이스로 노출
# ─────────────────────────────────────────────────────────────────────────────
def run_original_streamlit_script(file_rel: str):
    p = _abs(file_rel)
    if not p.exists():
        st.error(f"파일을 찾을 수 없습니다: {p}")
        return

    try:
        src = p.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return

    # 중복 set_page_config 방지
    _orig_spc = st.set_page_config
    st.set_page_config = (lambda *a, **k: None)

    # 실행 네임스페이스 구성
    ns = {
        "__name__": "__streamlit_runfile__",
        "__file__": str(p),
        "__package__": None,
    }

    # 해당 스크립트 디렉터리를 임시로 sys.path 최우선에 추가(상대 import 대비)
    added = False
    script_dir = str(p.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        added = True

    # 헬스 체크(로그 & 화면)
    print(f"HEALTH: executing {p}")
    st.write(f"🧭 실행 파일: `{file_rel}`")

    try:
        exec(compile(src, str(p), "exec"), ns, ns)
    except Exception:
        st.error("원본 코드 실행 중 오류가 발생했습니다. (아래 스택트레이스 참고)")
        st.code(traceback.format_exc())
        # 로그에도 남기기
        print(traceback.format_exc())
    finally:
        st.set_page_config = _orig_spc
        if added:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# 6) 라우팅
# ─────────────────────────────────────────────────────────────────────────────
LABEL_TO_FILE = {label: rel for label, rel in AVAILABLE}
target_rel = LABEL_TO_FILE.get(nav)

if target_rel is None:
    st.error("선택된 페이지가 존재하지 않습니다. 사이드바에서 다시 선택해 주세요.")
else:
    run_original_streamlit_script(target_rel)
