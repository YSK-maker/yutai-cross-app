import re
import sqlite3
import time
import json
import os
from io import BytesIO
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import altair as alt

st.sidebar.write("BUILD:", int(time.time()))

# =========================
# 設定
# =========================
REQUIRED_COLS = ["約定日", "取引種類", "銘柄名（ファンド名）", "銘柄コード", "数量", "摘要"]
OPTIONAL_PRICE_COLS = [
    "約定単価", "約定単価(円)", "約定単価（円）", "単価", "単価(円)", "単価（円）",
    "約定価格", "約定価格(円)", "約定価格（円）",
]

JPX_MASTER_CSV_PATH = "tse_incentive_stooq_20260104.csv"  # code,name,price,rights_raw,incentive_text
MEMO_DB = "memo.db"

UI_TEXT_PATH = "ui_texts.json"
DEFAULT_UI_TEXTS = {
    "page_title": "優待クロス",
    "tabs": ["優待管理", "ダッシュボード"],
    "uploader_label": "SMBC日興 取引履歴CSV（TorirekiXXXX.csv）",
    "uploader_info": "左のサイドバーからCSVをアップロードしてください。",
    "newsell_pattern_none": "（絞り込みなし）",
    "newsell_pattern_opt_1": "本日前後1カ月（前月同日〜翌月同日）",
    "newsell_pattern_opt_2": "本日後1カ月（本日〜翌月同日）",
    "newsell_pattern_opt_3": "本日前後2週間（±14日）",
    "memo_header": "メモ",
    "memo_help": "このセルを編集すると自動保存されます。",
}

PAGE_SIZE = 50
ALL_MONTHS = set(range(1, 13))

# =========================
# UIテキスト
# =========================
@st.cache_data(show_spinner=False)
def load_ui_texts(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

_ui_loaded = load_ui_texts(UI_TEXT_PATH)

def t(key: str, default: str | None = None) -> str:
    if default is None:
        default = DEFAULT_UI_TEXTS.get(key, "")
    return str(_ui_loaded.get(key, default))

def tlist(key: str, default: list[str]) -> list[str]:
    v = _ui_loaded.get(key, default)
    return v if isinstance(v, list) else default

# =========================
# CSVパース
# =========================
def parse_yyMMdd(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in ["", "---", "None", "nan"]:
        return None

    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return date(y, mo, d)
        except Exception:
            return None

    m2 = re.search(r"(\d{2})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m2:
        try:
            y = int(m2.group(1)) + 2000
            mo, d = int(m2.group(2)), int(m2.group(3))
            return date(y, mo, d)
        except Exception:
            return None

    return None

def parse_qty(x):
    if pd.isna(x):
        return 0
    s = str(x).replace(",", "").strip()
    if s in ["", "---", "None", "nan"]:
        return 0
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 0

def parse_price_like(x):
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().replace(",", "")
    if s == "" or s in ["---", "None", "nan"]:
        return None
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None

def credit_kind_from_summary(s):
    s = "" if pd.isna(s) else str(s).strip()
    if s.startswith("一般"):
        return "一般信用"
    if s.startswith("制度"):
        return "制度信用"
    return "不明"

def trade_kind(trade_type: str, summary: str) -> str:
    tt = "" if pd.isna(trade_type) else str(trade_type)
    if "現物" in tt:
        return "現物"
    ck = credit_kind_from_summary(summary)
    return ck if ck in ("制度信用", "一般信用") else "不明"

def month_num(d):
    if d is None or pd.isna(d):
        return None
    return int(d.month)

def md_num(d: date | None) -> int | None:
    if d is None or pd.isna(d):
        return None
    return d.month * 100 + d.day

def md_in_range(x_md: int | None, start_md: int, end_md: int) -> bool:
    if x_md is None:
        return False
    if start_md <= end_md:
        return start_md <= x_md <= end_md
    return (x_md >= start_md) or (x_md <= end_md)

def _last_day_of_month(y: int, m: int) -> int:
    if m == 12:
        return (date(y + 1, 1, 1) - timedelta(days=1)).day
    return (date(y, m + 1, 1) - timedelta(days=1)).day

def add_months_same_day(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last = _last_day_of_month(y, m)
    return date(y, m, min(d.day, last))

def get_newsell_md_window(pattern_key: str) -> tuple[int, int, date, date]:
    today = date.today()
    if pattern_key == "pm1":
        d1 = add_months_same_day(today, -1)
        d2 = add_months_same_day(today, +1)
    elif pattern_key == "p1":
        d1 = today
        d2 = add_months_same_day(today, +1)
    elif pattern_key == "pw2":
        d1 = today - timedelta(days=14)
        d2 = today + timedelta(days=14)
    else:
        d1 = add_months_same_day(today, -1)
        d2 = add_months_same_day(today, +1)
    return md_num(d1), md_num(d2), d1, d2

def chunk_lines(lines, chunk_size=7):
    if not lines:
        return "（なし）"
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
    return "\n\n".join(["\n".join(ch) for ch in chunks])

# =========================
# 表示整形
# =========================
def _fmt_price(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return "-"
    s = str(v)
    return s.rstrip("0").rstrip(".") if "." in s else s

def _fmt_text(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
        return "-"
    s = str(v).strip()
    return s if s else "-"

def is_incentive_rights_text(rights_text: str) -> bool:
    r = _fmt_text(rights_text)
    return r not in ["-", ""]

def extract_rights_months(rights_txt: str) -> list[int]:
    """
    権利確定日の文字から月を抽出。
    例: '3月末 / 9月末' -> [3,9]
    抽出できない形式は [] を返す（全選択時は除外しないよう、後段でフィルタをスキップする）
    """
    txt = _fmt_text(rights_txt)
    if txt in ("-", ""):
        return []
    months = re.findall(r"(\d{1,2})月", txt)
    mm = sorted({int(m) for m in months if m.isdigit() and 1 <= int(m) <= 12})
    return mm

# =========================
# SQLite（メモ）
# =========================
def _memo_conn():
    conn = sqlite3.connect(MEMO_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memo (
            code TEXT PRIMARY KEY,
            memo TEXT,
            updated_at INTEGER
        )
        """
    )
    return conn

def memo_get_all() -> dict:
    try:
        conn = _memo_conn()
        cur = conn.cursor()
        cur.execute("SELECT code, memo FROM memo")
        rows = cur.fetchall()
        conn.close()
        return {str(c).zfill(4): (m or "") for c, m in rows}
    except Exception:
        return {}

def memo_put(code: str, memo: str) -> bool:
    try:
        conn = _memo_conn()
        conn.execute(
            """
            INSERT INTO memo(code, memo, updated_at)
            VALUES(?,?,?)
            ON CONFLICT(code) DO UPDATE SET
                memo=excluded.memo,
                updated_at=excluded.updated_at
            """,
            (code, memo, int(time.time())),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

# =========================
# CSV読み込み
# =========================
@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    def _usecols(c):
        return (c in REQUIRED_COLS) or (c in OPTIONAL_PRICE_COLS)
    return pd.read_csv(BytesIO(file_bytes), encoding="cp932", usecols=_usecols)

def pick_unit_price_column(df: pd.DataFrame) -> str | None:
    for c in OPTIONAL_PRICE_COLS:
        if c in df.columns:
            return c
    return None

# =========================
# JPXマスタ
# =========================
@st.cache_data(show_spinner=False)
def load_jpx_master_csv(path: str) -> pd.DataFrame:
    if (not path) or (not os.path.exists(path)):
        return pd.DataFrame(columns=["code", "name", "price", "rights_raw", "incentive_text"])
    _ = os.path.getmtime(path)
    df = pd.read_csv(
        path,
        dtype={"code": "string", "name": "string"},
        usecols=["code", "name", "price", "rights_raw", "incentive_text"],
    )
    df["code"] = df["code"].astype(str).str.zfill(4)
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return df

# =========================
# SessionState
# =========================
def init_state():
    ss = st.session_state
    ss.setdefault("flt_q", "")
    ss.setdefault("flt_month_mode", "rights")   # rights / cover
    ss.setdefault("flt_cover_mms", set(range(1, 13)))
    ss.setdefault("flt_rights_mms", set(range(1, 13)))
    ss.setdefault("flt_ns_pattern", t("newsell_pattern_none"))
    ss.setdefault("flt_kind_teido", True)
    ss.setdefault("flt_kind_ippan", True)
    ss.setdefault("flt_kind_genbutsu", True)
    ss.setdefault("flt_inprogress", False)
    ss.setdefault("flt_incentive_only", True)  # デフォルトON
    ss.setdefault("page", 1)

    ss.setdefault("memo_prev", {})
    ss.setdefault("_memo_loaded", False)
    ss.setdefault("_init_rights_checks", False)
    ss.setdefault("_init_cover_checks", False)
    ss.setdefault("ui_mobile", True)  # ←追加（スマホ向け表示）

def reset_page():
    st.session_state.page = 1

def clear_filters_to_default():
    ss = st.session_state
    ss.flt_q = ""
    ss.flt_month_mode = "rights"
    ss.flt_cover_mms = set(range(1, 13))
    ss.flt_rights_mms = set(range(1, 13))
    ss.flt_ns_pattern = t("newsell_pattern_none")
    ss.flt_kind_teido = True
    ss.flt_kind_ippan = True
    ss.flt_kind_genbutsu = True
    ss.flt_inprogress = False
    ss.flt_incentive_only = True
    ss.page = 1
    for i in range(1, 13):
        ss[f"rights_mm_{i}"] = True
        ss[f"cover_mm_{i}"] = True
    ss._init_rights_checks = True
    ss._init_cover_checks = True

# =========================
# 取引の行テキスト（ページ分だけ生成して高速化）
# =========================
def build_trade_text_map(df: pd.DataFrame, codes_set: set[str], label_kind: bool) -> dict[str, str]:
    """
    df: columns [code, trade_date, kind, qty] がある前提
    codes_set: このページの銘柄だけ
    """
    if df.empty or not codes_set:
        return {c: "（なし）" for c in codes_set}

    sub = df[df["code"].isin(codes_set)].copy()
    if sub.empty:
        return {c: "（なし）" for c in codes_set}

    sub = sub.sort_values(["code", "trade_date"], ascending=[True, False])

    out: dict[str, str] = {}
    for code, g in sub.groupby("code", sort=False):
        lines = []
        for _, rr in g.iterrows():
            tag = ""
            if label_kind:
                tag = "（制度）" if rr["kind"] == "制度信用" else ("（現物）" if rr["kind"] == "現物" else "")
            dt = rr["trade_date"].strftime("%Y-%m-%d") if rr["trade_date"] else ""
            lines.append(f"- {dt} {int(rr['qty'])}株{tag}")
        out[code] = chunk_lines(lines, chunk_size=7) if lines else "（なし）"

    # codes_set に存在するが df に無いものの補完
    for c in codes_set:
        out.setdefault(c, "（なし）")
    return out

# =========================
# UI
# =========================
st.set_page_config(page_title=t("page_title"), layout="wide")

# =========================
# PWA（manifest読み込み / Service Worker登録）
# 重要：/static/... に統一（/app/static は使わない）
# =========================
st.markdown(
    """
<link rel="manifest" href="/static/manifest.json">
<meta name="theme-color" content="#ffffff">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<script>
  // ぐるぐる回避：/static/ を明示して登録。scopeもルートに固定。
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
      navigator.serviceWorker.register('/static/service-worker.js', { scope: '/' });
    });
  }
</script>
    """,
    unsafe_allow_html=True,
)

init_state()
ss = st.session_state

st.markdown(
    """
<style>
div[data-testid="stTabs"] button p { font-size: 1.5em !important; font-weight: 700 !important; }
.small-ui, .small-ui * { font-size: 13px !important; }
.small-ui .stButton button { white-space: nowrap !important; }

.small-ui .stButton button {
  padding: 0.01rem 0.18rem !important;
  font-size: 10.5px !important;
  min-height: 19px !important;
  line-height: 1.0 !important;
  border-radius: 6px !important;
  border-width: 1px !important;
}
.small-ui div[data-testid="stCheckbox"] { margin-top: -10px !important; margin-bottom: -10px !important; }

div[data-testid="stDataEditor"] td {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  line-height: 1.2 !important;
}
div[data-testid="stDataEditor"] table th:nth-child(7),
div[data-testid="stDataEditor"] table td:nth-child(7),
div[data-testid="stDataEditor"] table th:nth-child(8),
div[data-testid="stDataEditor"] table td:nth-child(8) {
  max-width: 150px !important;
  width: 150px !important;
}
div[data-testid="stDataEditor"] table th:last-child,
div[data-testid="stDataEditor"] table td:last-child {
  max-width: 160px !important;
  width: 160px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# サイドバー：取引履歴CSV
uploaded = st.sidebar.file_uploader(t("uploader_label"), type=["csv"])
if not uploaded:
    st.sidebar.info(t("uploader_info"))
    st.stop()

df_raw = read_csv_cached(uploaded.getvalue())
missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing:
    st.error(f"CSVに必要列がありません: {missing}")
    st.stop()

unit_price_col = pick_unit_price_column(df_raw)

tabs_labels = tlist("tabs", DEFAULT_UI_TEXTS["tabs"])
if (not isinstance(tabs_labels, list)) or (len(tabs_labels) != 2):
    tabs_labels = DEFAULT_UI_TEXTS["tabs"]
tabs = st.tabs(tabs_labels)

# 取引履歴整形
df2_all = pd.DataFrame()
df2_all["code"] = df_raw["銘柄コード"].astype(str).str.strip()
df2_all["name"] = df_raw["銘柄名（ファンド名）"].astype(str).fillna("")
df2_all["trade_type"] = df_raw["取引種類"].astype(str)
df2_all["trade_date"] = df_raw["約定日"].apply(parse_yyMMdd)
df2_all["qty"] = df_raw["数量"].apply(parse_qty)
df2_all["summary"] = df_raw["摘要"].astype(str)
df2_all["kind"] = [trade_kind(tt, sm) for tt, sm in zip(df2_all["trade_type"].tolist(), df2_all["summary"].tolist())]
df2_all["unit_price"] = df_raw[unit_price_col].apply(parse_price_like) if unit_price_col else None
df2_all = df2_all[df2_all["code"].str.fullmatch(r"\d{4}")].copy()
df2_all["code"] = df2_all["code"].astype(str).str.zfill(4)

# JPXマスタ
jpx_master_df = load_jpx_master_csv(JPX_MASTER_CSV_PATH)
if len(jpx_master_df) == 0:
    st.error(f"JPX銘柄マスタCSVが見つからないか空です: {JPX_MASTER_CSV_PATH}")
    st.stop()

# メモ読み込み（初回のみ）
if not ss._memo_loaded:
    ss.memo_prev = memo_get_all()
    ss._memo_loaded = True

# 取引中判定（新規売り > (現渡+返済)）
open_cnt = (
    df2_all[df2_all["trade_type"].str.contains("新規売り", na=False)]
    .groupby("code").size().to_dict()
)
close_cnt = (
    df2_all[df2_all["trade_type"].str.contains("現渡|返済買い|返済売り", na=False)]
    .groupby("code").size().to_dict()
)
inprogress_codes = {c for c, oc in open_cnt.items() if oc > int(close_cnt.get(c, 0))}

# =========================
# 優待管理
# =========================
with tabs[0]:
    st.markdown("### 優待管理")
    ss.ui_mobile = st.toggle("スマホ向け表示（カード）", value=bool(ss.ui_mobile))
    st.markdown('<div class="small-ui">', unsafe_allow_html=True)

    # 上段：リセット / 検索 / 右側フィルタ
    c0, c1, c2 = st.columns([1.1, 3.0, 5.9], vertical_alignment="center")

    with c0:
        if st.button("リセット", type="primary", width="stretch"):
            clear_filters_to_default()
            st.rerun()

    with c1:
        ss.flt_q = st.text_input("検索", ss.flt_q, key="w_flt_q", on_change=reset_page)

    with c2:
        rA, rB = st.columns([2.3, 2.7], vertical_alignment="center")
        with rA:
            st.caption("区分 / 状態")
            k1, k2, k3 = st.columns([1.0, 1.0, 1.0], gap="small", vertical_alignment="center")
            with k1:
                ss.flt_kind_teido = st.checkbox("制度", value=bool(ss.flt_kind_teido), key="w_k_teido", on_change=reset_page)
            with k2:
                ss.flt_kind_ippan = st.checkbox("一般", value=bool(ss.flt_kind_ippan), key="w_k_ippan", on_change=reset_page)
            with k3:
                ss.flt_kind_genbutsu = st.checkbox("現物", value=bool(ss.flt_kind_genbutsu), key="w_k_genbutsu", on_change=reset_page)

            ss.flt_inprogress = st.checkbox("取引中のみ", value=bool(ss.flt_inprogress), key="w_inprog", on_change=reset_page)

            # 優待銘柄のみ（デフォルトON）／OFFで非優待も含む全銘柄表示
            ss.flt_incentive_only = st.checkbox("優待銘柄のみ", value=bool(ss.flt_incentive_only), key="w_incentive_only", on_change=reset_page)

        with rB:
            st.caption("信用新規売り（約定日）")
            PATTERNS = [
                ("none", t("newsell_pattern_none")),
                ("pm1", t("newsell_pattern_opt_1")),
                ("p1",  t("newsell_pattern_opt_2")),
                ("pw2", t("newsell_pattern_opt_3")),
            ]
            disp_list = [v for _, v in PATTERNS]
            key_by_disp = {v: k for k, v in PATTERNS}
            cur_disp = ss.flt_ns_pattern if ss.flt_ns_pattern in disp_list else t("newsell_pattern_none")
            ss.flt_ns_pattern = st.selectbox(
                " ",
                disp_list,
                index=disp_list.index(cur_disp),
                key="w_ns",
                on_change=reset_page,
                label_visibility="collapsed",
            )

    # 月フィルタ行（左にモード）
    left = st.columns([1.8, 8.2], vertical_alignment="center")
    with left[0]:
        mode = st.radio(
            "月フィルタ",
            options=["権利確定月", "現渡実行月"],
            horizontal=False,
            index=0 if ss.flt_month_mode == "rights" else 1,
            key="w_month_mode",
            on_change=reset_page,
        )
        ss.flt_month_mode = "rights" if mode == "権利確定月" else "cover"

    with left[1]:
        def set_month_checks(prefix: str, values: bool):
            for i in range(1, 13):
                ss[f"{prefix}_mm_{i}"] = values

        def get_month_set(prefix: str) -> set[int]:
            s = set()
            for i in range(1, 13):
                if ss.get(f"{prefix}_mm_{i}", False):
                    s.add(i)
            return s

        def sync_month_set(prefix: str):
            s = get_month_set(prefix)
            if prefix == "rights":
                ss.flt_rights_mms = s
            else:
                ss.flt_cover_mms = s

        if ss.flt_month_mode == "rights":
            if not ss._init_rights_checks:
                for i in range(1, 13):
                    ss[f"rights_mm_{i}"] = (i in ss.flt_rights_mms)
                ss._init_rights_checks = True

            st.caption("権利確定月")
            widths = [0.9, 0.9] + [0.42] * 12
            cols = st.columns(widths, gap="small", vertical_alignment="center")

            with cols[0]:
                if st.button("全選択", key="btn_rights_all", width="stretch"):
                    set_month_checks("rights", True)
                    sync_month_set("rights")
                    reset_page()
                    st.rerun()
            with cols[1]:
                if st.button("全解除", key="btn_rights_none", width="stretch"):
                    set_month_checks("rights", False)
                    sync_month_set("rights")
                    reset_page()
                    st.rerun()

            for i in range(1, 13):
                with cols[1 + i]:
                    st.checkbox(str(i), key=f"rights_mm_{i}", on_change=reset_page)
            sync_month_set("rights")

        else:
            if not ss._init_cover_checks:
                for i in range(1, 13):
                    ss[f"cover_mm_{i}"] = (i in ss.flt_cover_mms)
                ss._init_cover_checks = True

            st.caption("現渡実行月（MM）")
            widths = [0.9, 0.9] + [0.42] * 12
            cols = st.columns(widths, gap="small", vertical_alignment="center")

            with cols[0]:
                if st.button("全選択", key="btn_cover_all", width="stretch"):
                    set_month_checks("cover", True)
                    sync_month_set("cover")
                    reset_page()
                    st.rerun()
            with cols[1]:
                if st.button("全解除", key="btn_cover_none", width="stretch"):
                    set_month_checks("cover", False)
                    sync_month_set("cover")
                    reset_page()
                    st.rerun()

            for i in range(1, 13):
                with cols[1 + i]:
                    st.checkbox(str(i), key=f"cover_mm_{i}", on_change=reset_page)
            sync_month_set("cover")

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # フィルタ適用（高速 & 全選択バグ修正）
    # =========================
    selected_kinds = []
    if ss.flt_kind_teido:
        selected_kinds.append("制度信用")
    if ss.flt_kind_ippan:
        selected_kinds.append("一般信用")
    if ss.flt_kind_genbutsu:
        selected_kinds.append("現物")

    if not selected_kinds:
        st.warning("区分が未選択のため、表示できる銘柄がありません。")
        st.stop()

    df_trade_kind = df2_all[df2_all["kind"].isin(selected_kinds)].copy()

    master = jpx_master_df[["code", "name", "price", "rights_raw", "incentive_text"]].copy()
    master["code"] = master["code"].astype(str).str.zfill(4)
    master["rights_txt"] = master["rights_raw"].apply(_fmt_text)
    master["is_incentive"] = master["rights_txt"].apply(is_incentive_rights_text)

    # デフォルト：非優待は非表示。OFFで全銘柄（非優待含む）
    if ss.flt_incentive_only:
        master = master[master["is_incentive"]].copy()

    # 取引中のみ
    if ss.flt_inprogress:
        master = master[master["code"].isin(inprogress_codes)].copy()

    # 現渡実行月（cover）
    if ss.flt_month_mode == "cover":
        if len(ss.flt_cover_mms) == 0:
            st.warning("現渡実行月が未選択のため、表示できる銘柄がありません。")
            st.stop()

        cov = df_trade_kind[df_trade_kind["trade_type"] == "現渡"].dropna(subset=["trade_date"]).copy()
        cov["cover_mm"] = cov["trade_date"].apply(month_num)
        cover_codes = set(cov[cov["cover_mm"].isin(ss.flt_cover_mms)]["code"].unique())

        master = master[master["code"].isin(cover_codes)].copy()

    # 信用新規売り期間
    PATTERNS = [
        ("none", t("newsell_pattern_none")),
        ("pm1", t("newsell_pattern_opt_1")),
        ("p1",  t("newsell_pattern_opt_2")),
        ("pw2", t("newsell_pattern_opt_3")),
    ]
    key_by_disp = {v: k for k, v in PATTERNS}
    ns_pattern_key = key_by_disp.get(ss.flt_ns_pattern, "none")
    if ns_pattern_key != "none":
        start_md, end_md, d1, d2 = get_newsell_md_window(ns_pattern_key)
        ns = df_trade_kind[df_trade_kind["trade_type"].str.contains("新規売り", na=False)].dropna(subset=["trade_date"]).copy()
        ns["md"] = ns["trade_date"].apply(md_num)
        ns = ns[ns["md"].apply(lambda x: md_in_range(x, start_md, end_md))]
        ns_codes = set(ns["code"].unique())
        master = master[master["code"].isin(ns_codes)].copy()
        st.caption(f"信用新規売り(mmdd): {start_md:04d}〜{end_md:04d}（参考 {d1.strftime('%m/%d')}〜{d2.strftime('%m/%d')}）")

    # 権利確定月（rights）
    if ss.flt_month_mode == "rights":
        if len(ss.flt_rights_mms) == 0:
            st.warning("権利確定月が未選択のため、表示できる銘柄がありません。")
            st.stop()

        # ★全選択（1〜12全部）なら「月で落とさない」＝権利月フィルタをスキップ
        if ss.flt_rights_mms != ALL_MONTHS:
            master["rights_months"] = master["rights_txt"].apply(extract_rights_months)
            master = master[master["rights_months"].apply(lambda mm: any(m in ss.flt_rights_mms for m in mm))].copy()

    # 検索（最後）
    q = (ss.flt_q or "").strip()
    if q:
        master = master[
            master["code"].astype(str).str.contains(q, na=False, case=False)
            | master["name"].astype(str).str.contains(q, na=False, case=False)
        ].copy()

    master = master.drop_duplicates(subset=["code"]).sort_values(["code"]).reset_index(drop=True)

    if len(master) == 0:
        st.warning("条件に一致する銘柄がありません。")
        st.stop()

    # =========================
    # 集計は「全件」→ 表示は「50件」なので、
    # 文字列生成（現渡/新規売りの明細）はページ対象の50銘柄だけに限定して高速化
    # =========================
    # 現渡回数（ソートに使うので全対象のカウントだけは一括で）
    cov_all = df_trade_kind[df_trade_kind["trade_type"] == "現渡"].dropna(subset=["trade_date"]).copy()
    cov_cnt = cov_all.groupby("code").size().rename("現渡回数").reset_index()

    # 表ソート用
    base = master[["code", "name", "price", "rights_raw", "incentive_text"]].copy()
    base = base.merge(cov_cnt, on="code", how="left")
    base["現渡回数"] = base["現渡回数"].fillna(0).astype(int)

    # 表示順（現渡回数 desc, code asc）
    base = base.sort_values(["現渡回数", "code"], ascending=[False, True]).reset_index(drop=True)

    # ページング
    total = len(base)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    ss.page = max(1, min(int(ss.page), int(total_pages)))

    start = (ss.page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page = base.iloc[start:end].copy()

    page_codes = set(page["code"].tolist())

    # ページ分だけ明細作成（ここが効く）
    cov_agg = cov_all.groupby(["code", "trade_date", "kind"], as_index=False)["qty"].sum()
    cover_map = build_trade_text_map(cov_agg, page_codes, label_kind=True)

    ns_all = df_trade_kind[df_trade_kind["trade_type"].str.contains("新規売り", na=False)].dropna(subset=["trade_date"]).copy()
    ns_agg = ns_all.groupby(["code", "trade_date", "kind"], as_index=False)["qty"].sum()
    newsell_map = build_trade_text_map(ns_agg, page_codes, label_kind=True)
    for c in page_codes:
        if newsell_map.get(c, "（なし）") == "（なし）":
            newsell_map[c] = "（新規売りなし）"

    # 画面に出すdf
    page["権利確定日"] = page["rights_raw"].apply(_fmt_text)
    page["株価"] = page["price"].apply(_fmt_price)
    page["現渡（日付・株数）"] = page["code"].map(lambda c: cover_map.get(c, "（現渡なし）"))
    page["信用新規売り（日付・株数）"] = page["code"].map(lambda c: newsell_map.get(c, "（新規売りなし）"))
    page["株主優待の内容"] = page["incentive_text"].apply(_fmt_text)
    page["メモ"] = page["code"].map(lambda c: ss.memo_prev.get(c, ""))

    display_df = page.rename(columns={"code": "コード", "name": "銘柄名"})[
        ["コード", "銘柄名", "権利確定日", "株価", "現渡回数", "現渡（日付・株数）", "信用新規売り（日付・株数）", "株主優待の内容", "メモ"]
    ]

# =========================
# 表示：PCは表、スマホはカード
# =========================
if ss.ui_mobile:
    st.caption("スマホ向け：カード表示（メモは各カード内で編集→保存）")

    for _, r in page.iterrows():
        code = str(r["code"]).zfill(4)
        name = str(r["name"])
        rights = _fmt_text(r["rights_raw"])
        price = _fmt_price(r["price"])
        cov_n = int(r.get("現渡回数", 0))
        memo_now = ss.memo_prev.get(code, "")

        with st.container(border=True):
            # 1段目：タイトル
            st.markdown(f"### {name}（{code}）")
            # 2段目：要点（行数を絞る）
            cA, cB, cC = st.columns([1.2, 1.2, 1.2])
            cA.metric("株価", price)
            cB.metric("権利確定", rights)
            cC.metric("現渡回数", cov_n)

            # 詳細（折りたたみ）
            with st.expander("詳細（現渡 / 新規売り / 優待内容）"):
                st.markdown("**現渡（日付・株数）**")
                st.write(cover_map.get(code, "（現渡なし）"))
                st.markdown("**信用新規売り（日付・株数）**")
                st.write(newsell_map.get(code, "（新規売りなし）"))
                st.markdown("**株主優待の内容**")
                st.write(_fmt_text(r["incentive_text"]))

            # メモ編集（スマホでの編集を最適化）
            new_memo = st.text_area(
                "メモ",
                value=memo_now,
                key=f"memo_{code}_{ss.page}",  # ページングしてもキー衝突しない
                height=90,
                placeholder="メモを書いて保存",
            )

            # 保存ボタン（スマホは“自動保存”より明示がUX良い）
            btn_col1, btn_col2 = st.columns([1.0, 4.0])
            with btn_col1:
                if st.button("保存", key=f"save_{code}_{ss.page}", type="primary"):
                    ok = memo_put(code, new_memo)
                    if ok:
                        ss.memo_prev[code] = new_memo
                        st.toast("保存しました")
                    else:
                        st.error("保存に失敗しました")

else:
    st.caption("PC向け：表（メモはセル編集→自動保存）")

    # 画面に出すdf（あなたの既存の display_df を使用）
    display_df = page.rename(columns={"code": "コード", "name": "銘柄名"})[
        ["コード", "銘柄名", "権利確定日", "株価", "現渡回数", "現渡（日付・株数）", "信用新規売り（日付・株数）", "株主優待の内容", "メモ"]
    ]

    edited = st.data_editor(
        display_df,
        width="stretch",
        height=760,
        num_rows="fixed",
        column_config={
            "メモ": st.column_config.TextColumn(t("memo_header"), help=t("memo_help")),
        },
        disabled=[
            "コード",
            "銘柄名",
            "権利確定日",
            "株価",
            "現渡回数",
            "現渡（日付・株数）",
            "信用新規売り（日付・株数）",
            "株主優待の内容",
        ],
    )

    # 自動保存（このページだけ）
    for _, row in edited.iterrows():
        code = str(row.get("コード", "")).strip()
        if not re.fullmatch(r"\d{4}", code):
            continue
        memo_now = "" if pd.isna(row.get("メモ")) else str(row.get("メモ"))
        memo_prev = ss.memo_prev.get(code, "")
        if memo_prev != memo_now:
            ok = memo_put(code, memo_now)
            if ok:
                ss.memo_prev[code] = memo_now


    # 前へ/次へ（表の最下段）
    b1, b2, b3 = st.columns([1.2, 1.2, 7.6], vertical_alignment="center")
    with b1:
        if st.button("◀ 前へ", disabled=(ss.page <= 1), width="stretch"):
            ss.page -= 1
            st.rerun()
    with b2:
        if st.button("次へ ▶", disabled=(ss.page >= total_pages), width="stretch"):
            ss.page += 1
            st.rerun()
    with b3:
        st.markdown(f"**{ss.page} / {total_pages} ページ**（{PAGE_SIZE}件/ページ / 全{total}件）")

# =========================
# ダッシュボード（元のまま）
# =========================
with tabs[1]:
    st.markdown("### ダッシュボード")

    cov = df2_all[df2_all["trade_type"] == "現渡"].dropna(subset=["trade_date"]).copy()
    if len(cov) == 0:
        st.info("現渡データが見つかりませんでした。")
        st.stop()

    cov["year"] = cov["trade_date"].apply(lambda d: int(d.year))
    cov["month"] = cov["trade_date"].apply(lambda d: int(d.month))

    years = sorted(cov["year"].unique())
    blues = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
    if len(years) <= len(blues):
        color_range = blues[-len(years):]
    else:
        color_range = (blues * (len(years) // len(blues) + 1))[-len(years):]
    year_domain = [str(y) for y in years]

    def blue_year_scale():
        return alt.Scale(domain=year_domain, range=color_range)

    def cfg18(c):
        return (
            c.configure_axis(labelFontSize=18, titleFontSize=18, labelAngle=0)
             .configure_legend(labelFontSize=18, titleFontSize=18)
             .configure_view(strokeWidth=0)
        )

    cov_year_cnt = cov.groupby("year").size().reset_index(name="count")
    cov_year_qty = cov.groupby("year")["qty"].sum().reset_index(name="qty")

    if unit_price_col and cov["unit_price"].notna().any():
        cov_amt_base = cov.copy()
        cov_amt_base["amount"] = cov_amt_base["qty"] * cov_amt_base["unit_price"]
        amount_note = f"（単価列『{unit_price_col}』×数量）"
    else:
        tmp = jpx_master_df[["code", "price"]].copy()
        tmp["code"] = tmp["code"].astype(str).str.zfill(4)
        tmp["px"] = tmp["price"].apply(parse_price_like)
        px_map = dict(zip(tmp["code"].tolist(), tmp["px"].fillna(0).tolist()))

        cov_amt_base = cov.copy()
        cov_amt_base["px"] = cov_amt_base["code"].astype(str).str.zfill(4).map(px_map).fillna(0)
        cov_amt_base["amount"] = cov_amt_base["qty"] * cov_amt_base["px"]
        amount_note = "（単価列なし：JPXマスタ株価×数量の概算）"

    cov_year_amt = cov_amt_base.groupby("year")["amount"].sum().reset_index(name="amount")

    cov_year_cnt["year"] = cov_year_cnt["year"].astype(str)
    cov_year_qty["year"] = cov_year_qty["year"].astype(str)
    cov_year_amt["year"] = cov_year_amt["year"].astype(str)

    top = st.columns([1.2, 3.2, 3.2, 3.2], vertical_alignment="center")
    with top[0]:
        st.markdown(f"<div style='font-size:44px; font-weight:800; color:#1d4ed8;'>{int(cov_year_cnt['count'].sum())}</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; color:#1e3a8a;'>現渡回数 合計</div>", unsafe_allow_html=True)

    with top[1]:
        ch0a = alt.Chart(cov_year_cnt).mark_bar().encode(
            x=alt.X("year:N", title="年", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="現渡回数"),
            color=alt.Color("year:N", scale=blue_year_scale(), legend=None),
            tooltip=["year:N", alt.Tooltip("count:Q", title="現渡回数")],
        ).properties(height=260, title="0a. 年ごとの現渡回数")
        st.altair_chart(cfg18(ch0a), width="stretch")

    with top[2]:
        ch0b = alt.Chart(cov_year_qty).mark_bar().encode(
            x=alt.X("year:N", title="年", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("qty:Q", title="現渡株数"),
            color=alt.Color("year:N", scale=blue_year_scale(), legend=None),
            tooltip=["year:N", alt.Tooltip("qty:Q", title="現渡株数", format=",d")],
        ).properties(height=260, title="0b. 年ごとの現渡株数")
        st.altair_chart(cfg18(ch0b), width="stretch")

    with top[3]:
        ch0c = alt.Chart(cov_year_amt).mark_bar().encode(
            x=alt.X("year:N", title="年", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("amount:Q", title="現渡金額"),
            color=alt.Color("year:N", scale=blue_year_scale(), legend=None),
            tooltip=["year:N", alt.Tooltip("amount:Q", title="現渡金額", format=",.0f")],
        ).properties(height=260, title=f"0c. 年ごとの現渡金額 {amount_note}")
        st.altair_chart(cfg18(ch0c), width="stretch")

    cov_cnt = cov.groupby(["year", "month"], as_index=False).size().rename(columns={"size": "count"})
    full = pd.MultiIndex.from_product([years, list(range(1, 13))], names=["year", "month"]).to_frame(index=False)
    cov_cnt = full.merge(cov_cnt, on=["year", "month"], how="left").fillna({"count": 0})
    cov_cnt["year"] = cov_cnt["year"].astype(str)
    cov_cnt["month"] = cov_cnt["month"].astype(int)

    ch1 = alt.Chart(cov_cnt).mark_bar().encode(
        x=alt.X("month:O", title="月（1〜12）", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count:Q", title="現渡回数"),
        color=alt.Color("year:N", title="年", scale=blue_year_scale()),
        xOffset="year:N",
        tooltip=["year:N", "month:O", alt.Tooltip("count:Q", title="現渡回数")],
    ).properties(height=340, title="1. yyyy-mmごとの現渡回数（年別）")
    st.altair_chart(cfg18(ch1), width="stretch")

    cov_amt = cov_amt_base.groupby(["year", "month"], as_index=False)["amount"].sum()
    cov_amt = full.merge(cov_amt, on=["year", "month"], how="left").fillna({"amount": 0})
    cov_amt["year"] = cov_amt["year"].astype(str)
    cov_amt["month"] = cov_amt["month"].astype(int)

    ch2 = alt.Chart(cov_amt).mark_bar().encode(
        x=alt.X("month:O", title="月（1〜12）", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("amount:Q", title="現渡金額"),
        color=alt.Color("year:N", title="年", scale=blue_year_scale()),
        xOffset="year:N",
        tooltip=["year:N", "month:O", alt.Tooltip("amount:Q", title="現渡金額", format=",.0f")],
    ).properties(height=340, title=f"2. yyyy-mmごとの現渡金額（年別）{amount_note}")
    st.altair_chart(cfg18(ch2), width="stretch")


