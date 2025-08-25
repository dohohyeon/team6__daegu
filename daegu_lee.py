# -*- coding: utf-8 -*-
"""
대구 위험도 점수 포인트 지도 (대구만! 시 외곽 굵은선 + 구 경계 얇은선)
- 입력: data/dataset23.csv, data/N3A_G0100000.shp (+ .shx/.dbf/.prj)
- 출력: daegu_risk_points.html (+ data/dataset23_scored.csv)
- 팔레트: 초록(안전) → 노랑 → 주황 → 빨강(위험) + 색 바(colormap) 범례
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import folium
from branca.element import MacroElement, Template
from branca.colormap import StepColormap

# ================== 경로 ==================
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV_POINTS = BASE / "data" / "dataset23.csv"
SHP_ADMIN  = BASE / "data" / "N3A_G0100000.shp"
OUT_HTML   = BASE / "daegu_risk_points.html"
OUT_CSV    = BASE / "data" / "dataset23_scored.csv"  # 점수 저장

# ================== 옵션 ==================
ZOOM_START = 12

# ColorBrewer RdYlGn_r 5단계 (빨강이 최위험)
COLOR = {
    "매우 낮음": "#1a9850",  # 진초록 (가장 안전)
    "낮음"    : "#91cf60",  # 연초록
    "보통"    : "#fee08b",  # 노랑
    "높음"    : "#fc8d59",  # 주황
    "매우 높음": "#d73027",  # 빨강 (가장 위험)
}

LAT_CANDS = ["위도","lat","LAT","Latitude","latitude","Y","y","위치Y","Y좌표"]
LON_CANDS = ["경도","lon","LON","Longitude","longitude","X","x","위치X","X좌표"]

# ====== (요청 반영) 가중치 ======
# * 안전 요소는 내부에서 자동으로 음수 방향(위험 감소)로 처리됨
WEIGHTS = {
    "사고건수": 0.30,
    "평균주행속도": 0.13,
    "불법주정차위반건수": 0.10,
    "어린이비율": 0.07,
    "보호구역도로폭": 0.05,           # 넓을수록 안전 → 값은 +지만 내부에서 위험 감소 처리
    "시설물 CCTV 수": 0.10,          # 많을수록 안전
    "시설물 도로표지판 수": 0.05,     # 많을수록 안전
    "시설물 과속방지턱 수": 0.10,     # 많을수록 안전
    "신호등 300m": 0.10               # 많을수록 안전
}

# 컬럼 매칭 패턴 (WEIGHTS의 키와 동일한 키로 유지)
KEYWORDS = {
    "사고건수": ["사고건수","사고","accident","accidents","총사고건수"],
    "평균주행속도": ["평균주행속도","주행속도","속도","speed","avg_speed"],
    "불법주정차위반건수": ["불법주정차위반건수","불법주정차","불법","위반건수","illegal"],
    "어린이비율": ["어린이비율","어린이","아동","children","kids","child_ratio"],
    "보호구역도로폭": ["보호구역도로폭","도로폭","폭","width","road_width"],
    "시설물 CCTV 수": ["시설물 CCTV 수","CCTV","cctv"],
    "시설물 도로표지판 수": ["시설물 도로표지판 수","도로표지판수","표지판","sign"],
    "시설물 과속방지턱 수": ["시설물 과속방지턱 수","과속방지턱","방지턱","speed hump","bump","hump"],
    "신호등 300m": ["신호등 300m","신호등_300m","신호등","signal","traffic_light","traffic light"]
}

# 위험을 ‘낮추는’ 안전 요소
SAFE_KEYS = {"보호구역도로폭","시설물 CCTV 수","시설물 도로표지판 수","시설물 과속방지턱 수","신호등 300m"}

# 대구만 남기는 안전 필터
DAEGU_NAMES = {"중구","동구","서구","남구","북구","수성구","달서구","달성군"}
DAEGU_BBOX = (128.25, 35.65, 128.85, 36.05)  # (minx, miny, maxx, maxy)

# ================== 헬퍼 ==================
def safe_read_csv(path, encs=("utf-8","cp949","euc-kr")):
    last=None
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last=e
    raise last

def find_first(cols, cands):
    for c in cands:
        if c in cols:
            return c
    lower={c.lower():c for c in cols}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in cols:
        if "lat" in c.lower() or "위도" in c.lower():
            return c
    for c in cols:
        if "lon" in c.lower() or "경도" in c.lower():
            return c
    return None

def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(s.median())
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

def match_column(cols, patterns):
    lc = [c.lower() for c in cols]
    for p in patterns:
        if p.lower() in lc:
            return cols[lc.index(p.lower())]
    for p in patterns:
        for c in cols:
            if p.lower() in c.lower():
                return c
    return None

def is_korea_bounds(lat, lon):
    return (33 <= lat) & (lat <= 39) & (124 <= lon) & (lon <= 132)

def in_bbox_center(geom, bbox):
    minx, miny, maxx, maxy = bbox
    c = geom.centroid
    return (minx <= c.x <= maxx) and (miny <= c.y <= maxy)

def last_token(s: str) -> str:
    s=str(s)
    return s.split()[-1] if " " in s else s

# ================== 메인 ==================
def main():
    # ---------- 1) 포인트 + 위험도 ----------
    if not CSV_POINTS.exists():
        raise FileNotFoundError(f"CSV가 없습니다: {CSV_POINTS}")
    df = safe_read_csv(CSV_POINTS)

    lat_col = find_first(df.columns.tolist(), LAT_CANDS)
    lon_col = find_first(df.columns.tolist(), LON_CANDS)
    if lat_col is None or lon_col is None:
        raise RuntimeError(f"위경도 컬럼 탐지 실패. CSV 컬럼: {list(df.columns)}")

    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["_lat","_lon"]).copy()

    # 위경도 뒤바뀜 자동 교정
    ok = is_korea_bounds(df["_lat"], df["_lon"])
    if ok.mean() < 0.5:
        df["_lat"], df["_lon"] = df["_lon"], df["_lat"]
        ok = is_korea_bounds(df["_lat"], df["_lon"])
    df = df[ok].copy().reset_index(drop=True)

    # 위험도 계산 (요청 가중치 반영 + 안전요소 음수 적용)
    used = {}
    for k, pats in KEYWORDS.items():
        mc = match_column(df.columns.tolist(), pats)
        if mc is not None:
            used[k]=mc

    parts=[]
    for k, w in WEIGHTS.items():
        if k not in used:
            continue
        v = normalize(df[used[k]])

        # 도로폭은 넓을수록 안전 → 값이 클수록 위험↓
        if k == "보호구역도로폭":
            v = 1 - v

        # 안전 요소는 위험을 낮추도록 부호 반전
        sign = -1 if k in SAFE_KEYS else 1
        parts.append(sign * w * v)

    if parts:
        risk_raw = sum(parts)
        # 최소/최대 보정 후 0~100 스케일
        risk = normalize(risk_raw) * 100
    else:
        num = df.drop(columns=[lat_col, lon_col, "_lat","_lon"], errors="ignore").select_dtypes(include="number")
        if num.shape[1]==0:
            raise RuntimeError("위험도 산식을 만들 수 있는 수치형 컬럼이 없습니다.")
        risk = normalize(num.sum(axis=1)) * 100

    df["위험도점수(0-100)"] = risk
    q20,q40,q60,q80 = df["위험도점수(0-100)"].quantile([0.2,0.4,0.6,0.8]).values
    def label_grade(v):
        if v<q20: return "매우 낮음"
        if v<q40: return "낮음"
        if v<q60: return "보통"
        if v<q80: return "높음"
        return "매우 높음"
    df["위험도등급"] = df["위험도점수(0-100)"].apply(label_grade)

    # 저장(원본+점수)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------- 2) 행정경계 (대구만, 안전 필터) ----------
    if not SHP_ADMIN.exists():
        raise FileNotFoundError(f"SHP가 없습니다: {SHP_ADMIN}")

    adm_raw = gpd.read_file(SHP_ADMIN)
    adm = adm_raw.to_crs(4326) if adm_raw.crs else adm_raw.set_crs(5179, allow_override=True).to_crs(4326)

    # 컬럼 후보
    COL_SIGCD  = [c for c in ["SIG_CD","SIGCD","ADM_CD","ADM_CD2","SIGCODE"] if c in adm.columns]
    COL_SIDCD  = [c for c in ["CTPRVN_CD","SIDO_CD","SIDOCD","SIDO"] if c in adm.columns]
    COL_NAME   = [c for c in ["SIG_KOR_NM","시군구명","SIG_KOR_NM2","NAME","명칭","SIG_ENG_NM"] if c in adm.columns]
    COL_SIDO_NM= [c for c in ["CTP_KOR_NM","시도명","SIDO_NM","SIDO_NAME"] if c in adm.columns]
    name_col   = COL_NAME[0] if COL_NAME else None

    # 1차: 코드로 대구(27) 슬라이스
    if COL_SIGCD:
        adm_dg = adm[adm[COL_SIGCD[0]].astype(str).str.startswith("27")].copy()
    elif COL_SIDCD:
        adm_dg = adm[adm[COL_SIDCD[0]].astype(str).str.startswith("27")].copy()
    else:
        adm_dg = adm.copy()

    # 2차: 시도명 "대구" 로 한 번 더 조이기
    if COL_SIDO_NM:
        tmp = adm_dg[adm_dg[COL_SIDO_NM[0]].astype(str).str.contains("대구", na=False)]
        if not tmp.empty:
            adm_dg = tmp

    # 3차: 이름 화이트리스트 + bbox 교차
    if name_col:
        adm_dg = adm_dg[adm_dg[name_col].astype(str).apply(lambda s: last_token(s) in DAEGU_NAMES)]
        adm_dg = adm_dg[[in_bbox_center(g, DAEGU_BBOX) for g in adm_dg.geometry]]
    else:
        adm_dg = adm_dg[[in_bbox_center(g, DAEGU_BBOX) for g in adm_dg.geometry]]

    if adm_dg.empty:
        raise RuntimeError("대구 경계 필터 결과가 비었습니다. SHP 컬럼/값을 확인하세요.")

    # 군위 제외(혹시라도)
    if name_col and "군위군" in adm_dg[name_col].astype(str).values:
        adm_dg = adm_dg[adm_dg[name_col] != "군위군"]

    # dissolve
    group_key = name_col if name_col else (COL_SIGCD[0] if COL_SIGCD else adm_dg.columns[0])
    adm_gu = adm_dg.dissolve(by=group_key, as_index=False)

   

    # 구 경계(얇은 검정선)
    gu_layer = folium.GeoJson(
        adm_gu[[group_key,"geometry"]], name="구 경계(라인)",
        style_function=lambda x: {"color":"#000000","weight":3,"fillOpacity":0.0}
    )
    try:
        folium.GeoJsonTooltip(fields=[group_key]).add_to(gu_layer)
    except Exception:
        pass

    # ---------- 3) 지도 ----------
    center = [df["_lat"].median(), df["_lon"].median()] if len(df) else [35.871,128.601]
    m = folium.Map(location=center, zoom_start=ZOOM_START, tiles="openstreetmap", control_scale=True)

   
    gu_layer.add_to(m)

    # 포인트(등급별 레이어)
    layers = {g: folium.FeatureGroup(name=f"포인트 - {g}", show=True)
              for g in ["매우 높음","높음","보통","낮음","매우 낮음"]}
    for g in layers.values():
        g.add_to(m)

    for _, r in df.iterrows():
        grade = r["위험도등급"]
        folium.CircleMarker(
            [r["_lat"], r["_lon"]],
            radius=6, color="#333", weight=1, fill=True,
            fill_color=COLOR.get(grade,"#888"), fill_opacity=0.9,
            tooltip=f"{grade} ({r['위험도점수(0-100)']:.1f})"
        ).add_to(layers[grade])

    # --- 색 바(colormap) 범례 (위험도 0~100) ---
    colormap = StepColormap(
        colors=[COLOR["매우 낮음"], COLOR["낮음"], COLOR["보통"], COLOR["높음"], COLOR["매우 높음"]],
        index=[0, 20, 40, 60, 80, 100],
        vmin=0, vmax=100,
        caption="위험도 (낮음 → 높음: 초록 → 노랑 → 주황 → 빨강)"
    )
    colormap.add_to(m)

    # --- 텍스트 범례(등급별 색/수치 범위 + 경계 설명) ---
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 9999;
                background: rgba(255,255,255,.96); padding: 12px 14px;
                border: 1px solid #666; border-radius: 10px; width: 360px; font-size: 13px;">
      <div style="font-weight:800; margin-bottom:6px">범례</div>
      <div style="line-height:1.5">
        <div><span style="display:inline-block;width:14px;height:14px;background:{COLOR['매우 높음']};border:1px solid #333;margin-right:8px"></span>
             매우 높음 <span style="color:#666">({q80:.0f}–100)</span></div>
        <div><span style="display:inline-block;width:14px;height:14px;background:{COLOR['높음']};border:1px solid #333;margin-right:8px"></span>
             높음 <span style="color:#666">({q60:.0f}–{q80:.0f})</span></div>
        <div><span style="display:inline-block;width:14px;height:14px;background:{COLOR['보통']};border:1px solid #333;margin-right:8px"></span>
             보통 <span style="color:#666">({q40:.0f}–{q60:.0f})</span></div>
        <div><span style="display:inline-block;width:14px;height:14px;background:{COLOR['낮음']};border:1px solid #333;margin-right:8px"></span>
             낮음 <span style="color:#666">({q20:.0f}–{q40:.0f})</span></div>
        <div><span style="display:inline-block;width:14px;height:14px;background:{COLOR['매우 낮음']};border:1px solid #333;margin-right:8px"></span>
             매우 낮음 <span style="color:#666">(0–{q20:.0f})</span></div>
       </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- 스케일바 + 레이어컨트롤 ---
    scale = MacroElement()
    scale._template = Template("""
    {% macro script(this, kwargs) %}
    L.control.scale({position: 'bottomleft', metric: true, imperial: false}).addTo({{this._parent.get_name()}});
    {% endmacro %}
    """)
    m.get_root().add_child(scale)

    folium.LayerControl(collapsed=False).add_to(m)

    # 시 외곽에 맞춰 뷰 조정
    try:
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    except Exception:
        pass

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print("[완료] HTML:", OUT_HTML)
    print("[완료] CSV :", OUT_CSV)

if __name__ == "__main__":
    main()