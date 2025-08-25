# app_schoolzone.py
# -*- coding: utf-8 -*-
"""
대구 구 단위 위험도 시뮬레이터 (Shiny 단일앱 / shiny==0.10.2)
- 슬라이더: 범위 로직 그대로, 표기는 정수(step=1)
- 지도 크게(MAP_HEIGHT), 범례(초록=안전, 노랑=보통, 빨강=위험) 추가
- Folium 지도를 iframe(srcdoc)으로 렌더 → 빈 화면 이슈 방지
- '초기화' 버튼 추가 → 현재 선택한 구의 기본값(데이터 평균)으로 슬라이더/결과 복원
"""

import os, sys, re, unicodedata, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from shapely.ops import unary_union
from shiny import App, ui, render, reactive
from html import escape  # iframe srcdoc용

warnings.filterwarnings("ignore", category=UserWarning)

# ============ 전역 옵션 ============
MAP_HEIGHT = 560  # 지도 세로 크기(px). 필요 시 600~700 조절

# ============ 경로 / 환경 ============
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
CSV_PATH  = os.path.join(DATA_DIR, "dataset.csv")
SHP_PATH  = os.path.join(DATA_DIR, "N3A_G0100000.shp")

# (Windows Conda) GDAL/PROJ 경고 완화
gdal_dir = os.path.join(sys.prefix, "Library", "share", "gdal")
proj_dir = os.path.join(sys.prefix, "Library", "share", "proj")
if os.path.isdir(gdal_dir):
    os.environ.setdefault("GDAL_DATA", gdal_dir)
if os.path.isdir(proj_dir):
    os.environ.setdefault("PROJ_LIB", proj_dir)

# ============ 파라미터/가중치 ============
DIST_LABELS = ['북구','남구','동구','서구','중구','수성구','달서구','달성군']
LABEL2CODE = {n:i for i,n in enumerate(DIST_LABELS)}
CODE2LABEL = {i:n for n,i in LABEL2CODE.items()}

LOCKED_COLS  = ['사고건수','어린이비율']
EDIT_COLS    = ['평균주행속도','불법주정차위반건수','보호구역도로폭',
                '시설물 CCTV 수','시설물 도로표지판 수','시설물 과속방지턱 수','신호등 300m']
RISK_COLS    = LOCKED_COLS + ['평균주행속도','불법주정차위반건수','보호구역도로폭']
PROTECT_COLS = ['시설물 CCTV 수','시설물 도로표지판 수','시설물 과속방지턱 수','신호등 300m']
ALL_COLS     = RISK_COLS + PROTECT_COLS

WEIGHTS = {
    '사고건수': 0.30, '평균주행속도': 0.13, '불법주정차위반건수': 0.10, '어린이비율': 0.07, '보호구역도로폭': 0.05,
    '시설물 CCTV 수': 0.10, '시설물 도로표지판 수': 0.05, '시설물 과속방지턱 수': 0.10, '신호등 300m': 0.10
}

# ============ 유틸 ============
def safe_read_csv(path, encs=("utf-8","cp949","euc-kr")):
    last=None
    for enc in encs:
        try: return pd.read_csv(path, encoding=enc)
        except Exception as e: last=e
    raise last

def normalize_text(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00A0"," ").replace("\u3000"," ")
    return re.sub(r"\s+"," ", s).strip()

def extract_gugun_from_address(s):
    s = normalize_text(str(s))
    pat = re.compile(r"(달서\s*구|달성\s*군|수성\s*구|북\s*구|남\s*구|동\s*구|서\s*구|중\s*구)\b")
    m = pat.search(s);  return (re.sub(r"\s+","",m.group(1)) if m else None)

def agg_mean_by_code(df, col):
    return (df.groupby("구군코드")[col].mean().reindex(range(8)))

def minmax_arr(arr: np.ndarray):
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax==vmin:
        return np.zeros_like(arr, dtype=float), vmin, vmax
    return (arr - vmin) / (vmax - vmin), vmin, vmax

def normalize_weights(w):
    s = sum(w.values())
    return {k:(v/s if s>0 else 1/len(w)) for k,v in w.items()}

def color_for_risk(r):  # 0~100 → hex (초록→노랑→빨강)
    r = max(0,min(100,float(r)))
    if r <= 50:
        t = r/50.0
        R,G,B = int(34 + t*(255-34)), int(197 + t*(205-197)), 94
    else:
        t = (r-50)/50.0
        R,G,B = 255, int(205*(1-t)), int(94*(1-t))
    return f"#{R:02x}{G:02x}{B:02x}"

def _as_iframe(html_str: str, height_px: int) -> str:
    # Folium 전체 HTML을 iframe(srcdoc)으로 안전하게 삽입 (중첩 HTML 충돌 방지)
    return f'<iframe srcdoc="{escape(html_str, quote=True)}" style="width:100%;height:{height_px}px;border:0;"></iframe>'

# ============ 데이터 ============
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV 없음: {CSV_PATH}")

df = safe_read_csv(CSV_PATH)

if '어린이비율' not in df.columns and {'전체인구','어린이인구'}.issubset(df.columns):
    df['전체인구']   = pd.to_numeric(df['전체인구'].astype(str).str.replace(',','',regex=False), errors='coerce')
    df['어린이인구'] = pd.to_numeric(df['어린이인구'].astype(str).str.replace(',','',regex=False), errors='coerce')
    df['어린이비율'] = (df['어린이인구']/df['전체인구']*100).round(2)

if "구군코드" not in df.columns:
    if "주소" in df.columns:
        gg = df["주소"].map(extract_gugun_from_address)
        df["구군코드"] = gg.map(LABEL2CODE)
    elif "구" in df.columns:
        df["구군코드"] = df["구"].astype(str).map(LABEL2CODE)
    else:
        raise RuntimeError("지역 구분을 위해 '구군코드' 또는 '주소'/'구' 컬럼이 필요합니다.")

missing = [c for c in ALL_COLS if c not in df.columns]
if missing:
    raise RuntimeError(f"필수 지표 누락: {missing}")

means_raw = {c: agg_mean_by_code(df, c).values.astype(float) for c in ALL_COLS}

# 정규화 기준(초기 데이터 기준, 고정)
norm_baseline = {}
minmax_table  = {}
for c in ALL_COLS:
    n01, vmin, vmax = minmax_arr(means_raw[c])
    if c in PROTECT_COLS:
        n01 = 1 - n01
    norm_baseline[c] = n01
    minmax_table[c]  = (vmin, vmax)

# 위험도 계산(Soft 5~95)
WN = normalize_weights(WEIGHTS)
def risk_from_norm(norm_dict: dict[str, np.ndarray]) -> np.ndarray:
    sc = np.zeros(8, dtype=float)
    for c, arr in norm_dict.items():
        sc += WN.get(c, 0.0) * np.asarray(arr, dtype=float)
    lo, hi = float(np.min(sc)), float(np.max(sc))
    r01 = (sc - lo) / (hi - lo + 1e-9)
    r01 = 0.05 + 0.90 * np.clip(r01, 0, 1)  # 0.05~0.95
    return r01 * 100.0

risk_now_all = risk_from_norm(norm_baseline)

# ============ 지도(SHP) ============
def _centroid_latlon(gdf):
    if hasattr(gdf.geometry, "union_all"):  u = gdf.geometry.union_all()
    else:                                   u = unary_union(gdf.geometry.values)
    c = u.centroid;  return float(c.y), float(c.x)

def load_daegu_shp(shp_path: str):
    if not os.path.exists(shp_path): return None
    g = gpd.read_file(shp_path)
    code_cols = [c for c in ["SIG_CD","ADM_CD","BJD_CD","BJCD"] if c in g.columns]
    if code_cols:
        ccol = code_cols[0]
        g = g[g[ccol].astype(str).str.startswith("27")].copy()
    elif "CTP_KOR_NM" in g.columns:
        g = g[g["CTP_KOR_NM"].astype(str).eq("대구광역시")].copy()
    else:
        name_cols = [c for c in ["NAME","SIG_KOR_NM","FULL_NM"] if c in g.columns]
        if name_cols:
            ncol = name_cols[0]
            g = g[g[ncol].astype(str).str.match(r"^대구광역시(\s|$)")].copy()
    if "SIG_KOR_NM" in g.columns:
        s = g["SIG_KOR_NM"].astype(str); g["_GU"] = s.str.replace(r"^대구광역시\s*","",regex=True)
    elif "NAME" in g.columns:
        s = g["NAME"].astype(str);       g["_GU"] = s.str.replace(r"^대구광역시\s*","",regex=True)
    else:
        g["_GU"] = g.index.astype(str)
    g["_GU"] = g["_GU"].str.split().str[-1]
    g = g[g["_GU"].isin(DIST_LABELS)][["_GU","geometry"]].drop_duplicates(subset=["_GU"])
    if g.empty: return None
    g = (g.set_crs(5179, allow_override=True).to_crs(4326) if g.crs is None else g.to_crs(4326))
    return g.rename(columns={"_GU":"GU"}).reset_index(drop=True)

dg_gdf = load_daegu_shp(SHP_PATH)

# 범례(legend) HTML
def _risk_legend_html():
    return """
    <div style="
        position: fixed;
        bottom: 20px; right: 20px;
        z-index: 9999;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,.15);
        padding: 8px 10px;
        font-size: 12px; line-height: 1.2;">
        <div style="font-weight: 600; margin-bottom: 6px;">
            위험도 색상 가이드
        </div>
        <div style="width: 220px; height: 10px;
            background: linear-gradient(to right, #22c55e, #ffcd5e 50%, #ff0000);
            border-radius: 6px;">
        </div>
        <div style="display:flex; justify-content:space-between; margin-top: 4px;">
            <span>안전</span><span>보통</span><span>위험</span>
        </div>
    </div>
    """

def make_folium_map(dg_gdf, selected_gu, risk_val, title="위험도"):
    lat, lon = _centroid_latlon(dg_gdf)
    m = folium.Map(
        location=[lat,lon], zoom_start=11, tiles="CartoDB positron",
        control_scale=True, width="100%", height=MAP_HEIGHT
    )
    for _, row in dg_gdf.iterrows():
        gu = row["GU"]
        if gu == selected_gu:
            col = color_for_risk(risk_val)
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=col: {"color":"#000","weight":3,"fillColor":c,"fillOpacity":0.7},
                tooltip=f"{gu} · 위험도 {risk_val:.1f}",
            ).add_to(m)
        else:
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {"color":"#666","weight":1,"fillColor":"#e5e7eb","fillOpacity":0.15},
                tooltip=gu,
            ).add_to(m)

    # 제목 + 범례
    m.get_root().html.add_child(folium.Element(
        f"<div style='padding:6px;font-size:12px'><b>{title}</b></div>"
    ))
    m.get_root().html.add_child(folium.Element(_risk_legend_html()))

    # Folium 전체 HTML을 iframe(srcdoc)으로 감싸 반환 → 빈 화면 이슈 방지
    html_full = m.get_root().render()
    return _as_iframe(html_full, MAP_HEIGHT)

# ============ UI (슬라이더 콤팩트 + 4열 배치) ============
def slider_limits(col: str, cur: float):
    """
    범위 산식은 기존 그대로 사용하되, UI에서 정수로 보이도록:
    - lo: 내림하여 정수
    - hi: 올림하여 정수
    - step: 1
    """
    vmin, vmax = minmax_table[col]
    lo_f = 0.0 if vmin >= 0 else (vmin * 0.8)  # 기존 범위 산식 유지
    hi_f = (max(cur, vmax) * 1.5 if vmax > 0 else
            (cur * 1.5 if cur > 0 else 1.0))   # 기존 범위 산식 유지

    lo = int(np.floor(lo_f))
    hi = int(np.ceil(hi_f))
    if hi <= lo:
        hi = lo + 1
    step = 1
    return lo, hi, step

def slider_block(id_, label, min_, max_, step_, value_):
    return ui.card(
        ui.input_slider(id_, label, min=min_, max=max_, value=value_, step=step_),
        class_="compact"
    )

INIT_GU = LABEL2CODE["중구"]
# 초기 표시값도 정수로 세팅
cur_vals_init = {c: int(round(float(means_raw[c][INIT_GU]))) for c in EDIT_COLS}
limits = {c: slider_limits(c, cur_vals_init[c]) for c in EDIT_COLS}

app_ui = ui.page_fillable(
    ui.panel_title("대구 위험도 시뮬레이터 "),
    # CSS로 슬라이더/카드 여백 줄이기
    ui.head_content(ui.tags.style("""
    .card.compact .card-header{padding:6px 10px; font-size:0.95rem;}
    .card.compact .card-body{padding:8px;}
    .form-range{height:0.35rem;}
    .form-range::-webkit-slider-thumb{width:0.9rem;height:0.9rem;}
    .form-range::-moz-range-thumb{width:0.9rem;height:0.9rem;}
    """)),
    # 상단: 4열(1/4)로 촘촘하게
    ui.layout_column_wrap(
        1/4,
        slider_block("sl_speed", "평균주행속도(↑ 위험)", *limits['평균주행속도'], cur_vals_init['평균주행속도']),
        slider_block("sl_illeg", "불법주정차위반건수(↑ 위험)", *limits['불법주정차위반건수'], cur_vals_init['불법주정차위반건수']),
        slider_block("sl_width", "보호구역도로폭(↑ 위험)", *limits['보호구역도로폭'], cur_vals_init['보호구역도로폭']),
        slider_block("sl_cctv",  "시설물 CCTV 수(↑ 안전)", *limits['시설물 CCTV 수'], cur_vals_init['시설물 CCTV 수']),
        slider_block("sl_sign",  "시설물 도로표지판 수(↑ 안전)", *limits['시설물 도로표지판 수'], cur_vals_init['시설물 도로표지판 수']),
        slider_block("sl_bump",  "시설물 과속방지턱 수(↑ 안전)", *limits['시설물 과속방지턱 수'], cur_vals_init['시설물 과속방지턱 수']),
        slider_block("sl_sig",   "신호등 300m(↑ 안전)", *limits['신호등 300m'], cur_vals_init['신호등 300m']),
        ui.card(
            ui.input_select("gu", "구/군 선택", {LABEL2CODE[n]: n for n in DIST_LABELS}, selected=INIT_GU),
            ui.div(
                ui.input_action_button("apply", "적용", class_="btn-primary"),
                ui.input_action_button("reset", "초기화", class_="btn-secondary ms-2"),
            ),
            class_="compact"
        ),
        gap="8px"
    ),
    # 하단: 지도(세로 큰 높이)
    ui.layout_columns(
        ui.card(ui.card_header("현재 위험도"), ui.output_ui("map_before"), class_="compact"),
        ui.card(ui.card_header("적용 후 위험도"), ui.output_ui("map_after"), class_="compact"),
        col_widths=(6,6),
    ),
)

# ============ Server ============
def server(input, output, session):

    # 구 변경 시 슬라이더를 '정수'로 동기화
    @reactive.Effect
    def _sync_sliders_to_gu():
        gu_idx = int(input.gu())
        for col, sid in [
            ('평균주행속도','sl_speed'),
            ('불법주정차위반건수','sl_illeg'),
            ('보호구역도로폭','sl_width'),
            ('시설물 CCTV 수','sl_cctv'),
            ('시설물 도로표지판 수','sl_sign'),
            ('시설물 과속방지턱 수','sl_bump'),
            ('신호등 300m','sl_sig'),
        ]:
            cur_int = int(round(float(means_raw[col][gu_idx])))
            try:
                ui.update_slider(session, sid, value=cur_int)
            except Exception:
                session.send_input_message(sid, {"value": cur_int})

    # ▶ 초기화: 현재 선택된 구의 기본값으로 복원 + 적용결과 제거
    after_scores = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.reset)
    def _reset_sliders():
        gu_idx = int(input.gu())
        for col, sid in [
            ('평균주행속도','sl_speed'),
            ('불법주정차위반건수','sl_illeg'),
            ('보호구역도로폭','sl_width'),
            ('시설물 CCTV 수','sl_cctv'),
            ('시설물 도로표지판 수','sl_sign'),
            ('시설물 과속방지턱 수','sl_bump'),
            ('신호등 300m','sl_sig'),
        ]:
            cur_int = int(round(float(means_raw[col][gu_idx])))
            try:
                ui.update_slider(session, sid, value=cur_int)
            except Exception:
                session.send_input_message(sid, {"value": cur_int})
        # 적용 결과 초기화
        after_scores.set(None)

    @output
    @render.ui
    def map_before():
        if dg_gdf is None:
            return ui.HTML("<div style='padding:12px'>SHP 로딩 실패: data/N3A_G0100000.shp 확인</div>")
        gu_idx = int(input.gu())
        gu_name = CODE2LABEL[gu_idx]
        r_now = float(risk_now_all[gu_idx])
        return ui.HTML(make_folium_map(dg_gdf, gu_name, r_now, title="현재 위험도"))

    @reactive.Effect
    @reactive.event(input.apply)
    def _recalc_after():
        gu_idx = int(input.gu())
        norm_after = {k: v.copy() for k,v in norm_baseline.items()}

        new_vals = {
            '평균주행속도': float(input.sl_speed()),
            '불법주정차위반건수': float(input.sl_illeg()),
            '보호구역도로폭': float(input.sl_width()),
            '시설물 CCTV 수': float(input.sl_cctv()),
            '시설물 도로표지판 수': float(input.sl_sign()),
            '시설물 과속방지턱 수': float(input.sl_bump()),
            '신호등 300m': float(input.sl_sig()),
        }
        for col, new_val in new_vals.items():
            vmin, vmax = minmax_table[col]  # 정규화 기준은 '초기' 그대로
            if vmax == vmin: n01 = 0.0
            else:
                n01 = (new_val - vmin) / (vmax - vmin)
                n01 = max(0.0, min(1.0, n01))
            if col in PROTECT_COLS: n01 = 1.0 - n01
            norm_after[col][gu_idx] = n01

        after_scores.set(risk_from_norm(norm_after))

    @output
    @render.ui
    def map_after():
        if dg_gdf is None:
            return ui.HTML("<div style='padding:12px'>SHP 로딩 실패: data/N3A_G0100000.shp 확인</div>")
        gu_idx = int(input.gu())
        gu_name = CODE2LABEL[gu_idx]
        arr = after_scores.get()
        r_val = float(arr[gu_idx]) if arr is not None else float(risk_now_all[gu_idx])
        title = "적용 후 위험도" if arr is not None else "적용 결과(대기중)"
        return ui.HTML(make_folium_map(dg_gdf, gu_name, r_val, title=title))

app = App(app_ui, server)
