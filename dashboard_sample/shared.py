# shared.py
import os, re, unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import folium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
SHP_PATH = os.path.join(BASE_DIR, "data", "N3A_G0100000.shp")

DIST_LABELS = ["북구", "남구", "동구", "서구", "중구", "수성구", "달서구", "달성군"]
LABEL2CODE = {n: i for i, n in enumerate(DIST_LABELS)}
CODE2LABEL = {i: n for n, i in LABEL2CODE.items()}

WEIGHTS = {
    "사고건수": 0.30, "평균주행속도": 0.13, "불법주정차위반건수": 0.10,
    "어린이비율": 0.07, "보호구역도로폭": 0.05,
    "시설물 CCTV 수": 0.10, "시설물 도로표지판 수": 0.05,
    "시설물 과속방지턱 수": 0.10, "신호등 300m": 0.10,
}
PROTECT_COLS = ["시설물 CCTV 수","시설물 도로표지판 수","시설물 과속방지턱 수","신호등 300m"]

# ---------- 데이터 불러오기 ----------
def load_data():
    df = pd.read_csv(CSV_PATH)
    # 어린이비율 계산
    if "어린이비율" not in df.columns and {"전체인구","어린이인구"}.issubset(df.columns):
        df["전체인구"] = pd.to_numeric(df["전체인구"].astype(str).str.replace(",","",regex=False),errors="coerce")
        df["어린이인구"] = pd.to_numeric(df["어린이인구"].astype(str).str.replace(",","",regex=False),errors="coerce")
        df["어린이비율"] = (df["어린이인구"]/df["전체인구"]*100).round(2)
    return df

def load_shp():
    g = gpd.read_file(SHP_PATH)
    g = g.to_crs(4326) if g.crs else g.set_crs(5179, allow_override=True).to_crs(4326)
    g["GU"] = g["NAME"].astype(str).str.strip()
    dg = g[g["GU"].isin(DIST_LABELS)][["GU","geometry"]].reset_index(drop=True)
    return dg

# ---------- 위험도 계산 ----------
def normalize_weights(w):
    s = sum(w.values())
    return {k:(v/s if s>0 else 1/len(w)) for k,v in w.items()}

WN = normalize_weights(WEIGHTS)

def risk_from_norm(norm_dict):
    sc = np.zeros(8,dtype=float)
    for c,arr in norm_dict.items():
        sc += WN.get(c,0.0)*np.asarray(arr,dtype=float)
    lo,hi = float(np.min(sc)), float(np.max(sc))
    r01 = (sc-lo)/(hi-lo+1e-9)
    r01 = 0.05+0.90*np.clip(r01,0,1)
    return r01*100.0

# ---------- 지도 ----------
def _centroid_latlon(gdf):
    u = unary_union(gdf.geometry.values)
    c = u.centroid
    return float(c.y), float(c.x)

def color_for_risk(r):
    r = max(0,min(100,float(r)))
    if r<=50:
        t=r/50.0
        R,G,B=int(34+t*(255-34)),int(197+t*(205-197)),94
    else:
        t=(r-50)/50.0
        R,G,B=255,int(205*(1-t)),int(94*(1-t))
    return f"#{R:02x}{G:02x}{B:02x}"

def make_folium_map(gdf,selected_gu,risk_val,title="위험도"):
    lat,lon=_centroid_latlon(gdf)
    m=folium.Map(location=[lat,lon],zoom_start=11,tiles="CartoDB positron")
    for _,row in gdf.iterrows():
        gu=row["GU"]
        if gu==selected_gu:
            col=color_for_risk(risk_val)
            folium.GeoJson(row.geometry.__geo_interface__,
                           style_function=lambda x,c=col:{"color":"#000","weight":3,"fillColor":c,"fillOpacity":0.7},
                           tooltip=f"{gu} · 위험도 {risk_val:.1f}").add_to(m)
        else:
            folium.GeoJson(row.geometry.__geo_interface__,
                           style_function=lambda x:{"color":"#666","weight":1,"fillColor":"#e5e7eb","fillOpacity":0.15},
                           tooltip=gu).add_to(m)
    return m.get_root().render()
