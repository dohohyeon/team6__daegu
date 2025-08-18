import requests
import pandas as pd
import geopandas as gpd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from rapidfuzz import fuzz


df1 = pd.read_csv("data/2020.csv")
df2 = pd.read_csv("data/2022.csv")
df2['사고건수'] = df1['어린이보행자 사고건수 (2020년 ~ 2021년)'] + df2['어린이보행자 사고건수 (2022년 ~ 2024년)']

del df2['어린이보행자 사고건수 (2022년 ~ 2024년)']
df2

df3 = pd.read_csv("data/대구어린이보호구역.csv",encoding='cp949')

df2.head()
df3.head()

# df2 df3 조인

# 전처리 함수
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).replace(" ", "").strip().lower()

# 전처리
df2['시설명_clean'] = df2['시설명'].apply(clean_text)
df2['주소_clean'] = df2['주소'].apply(clean_text)
df3['대상시설명_clean'] = df3['대상시설명'].apply(clean_text)
df3['소재지도로명주소_clean'] = df3['소재지도로명주소'].apply(clean_text)

matches = []

for i, (name2, addr2) in enumerate(zip(df2['시설명_clean'], df2['주소_clean'])):
    best_score = 0
    best_idx = None
    
    for j, (name3, addr3) in enumerate(zip(df3['대상시설명_clean'], df3['소재지도로명주소_clean'])):
        score_name = fuzz.token_sort_ratio(name2, name3)
        score_addr = fuzz.token_sort_ratio(addr2, addr3)
        
        # 이름 또는 주소 둘 중 하나라도 95 이상이면 매칭 후보
        if score_name >= 95 or score_addr >= 95:
            combined_score = max(score_name, score_addr)  # 높은 쪽 점수 사용
            if combined_score > best_score:
                best_score = combined_score
                best_idx = j
    
    if best_idx is not None:
        matches.append((best_score, best_idx))
    else:
        matches.append((None, None))

matches_df = pd.DataFrame(matches, columns=['유사도', 'df3_idx'])

df2_matched = pd.concat([df2.reset_index(drop=True), matches_df], axis=1)

result = df2_matched.merge(
    df3.reset_index().rename(columns={'index': 'df3_idx'}),
    on='df3_idx',
    how='left'
)

print(result.head())
result.to_csv("결합3.csv", index=False, encoding='utf-8-sig')

# 이름,주소 유사도 기준으로 조인 후, 매칭 안 된 데이터는 수작업으로 진행


df = pd.read_csv("data/dataset.csv", encoding='cp949')
df.columns

# 위도, 경도 정보로 읍면동 변수 생성

# 예시 데이터프레임

# 카카오 REST API 키 (본인 발급 키로 교체)
KAKAO_API_KEY = "1b7f9f50feea28d092529d889942308d"

def get_address(lat, lon):
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }
    params = {
        "x": lon,  # 경도
        "y": lat   # 위도
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            try:
                # 행정동 이름 가져오기
                return data['documents'][0]['address']['region_3depth_name']
            except (IndexError, KeyError):
                return None
        else:
            print(f"API 오류 {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"예외 발생: {e}")
        return None

# 데이터프레임에 적용
df['읍면동'] = [get_address(lat, lon) for lat, lon in zip(df['위도'], df['경도'])]
time.sleep(0.2)  # API 호출 제한 고려

print(df)
len(df['읍면동'].unique())

df.columns
df.to_csv("dataset1.csv", index=False, encoding='utf-8-sig')

# 동이름 변경된 거는 수작업으로 변경함

dataset2 = pd.read_csv("data/dataset2.csv", encoding='cp949')
speed = pd.read_csv("data/speed.csv", encoding='cp949')

# 기존 데이터셋에 주행속도 데이터 조인

# 예시: dataset2와 speed 모두 '읍면동' 컬럼을 가지고 있다고 가정
result = pd.merge(
    dataset2,
    speed,
    on='읍면동',        # 조인 키
    how='left'         # left outer join
)

result[result['평균주행속도'].isnull()]
result.to_csv("dataset4.csv", index=False, encoding='utf-8-sig')



# 신호등 갯수 넣는 코드?




# 결합 끝 분석 시작
df = pd.read_csv("data/dataset9.csv", encoding='utf-8')
df.info()










# 숫자형 컬럼만 선택
numeric_cols = df.select_dtypes(include=['float64', 'int64'])

# 상관계수 계산 (스피어만)
corr_matrix = numeric_cols.corr(method='spearman')
print(corr_matrix)


plt.rc('font', family='Malgun Gothic')  # Windows: 맑은 고딕
plt.rc('axes', unicode_minus=False)     # 음수 기호 깨짐 방지

# ---- (2) 상관계수 계산 ----
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_cols.corr(method='spearman')

# ---- (3) 히트맵 시각화 ----
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,       # 상관계수 숫자 표시
            fmt=".2f",        # 소수점 2자리
            cmap='coolwarm',  # 색상
            cbar=True,
            square=True)

plt.title('변수 간 상관계수 히트맵', fontsize=14)
plt.show()


df.hist(figsize=(12, 10), bins=20, grid=False)
plt.tight_layout()
plt.show()

plt.scatter(df['보호구역도로폭'],df['불법주정차위반건수'])

(df['보호구역도로폭'].isnull() & df['불법주정차위반건수'].isnull()).sum()



df = df.dropna(subset=['보호구역도로폭'])



# 2) 불법주정차위반건수 결측을 0으로 채움
df['불법주정차위반건수'] = df['불법주정차위반건수'].fillna(0)

# 확인
df.info()



col_accidents   = '사고건수'
col_speed       = '평균주행속도'
col_illegal     = '불법주정차위반건수'
col_cctv        = '시설물 CCTV 수'
col_sign        = '시설물 도로표지판 수'
col_bump        = '시설물 과속방지턱 수'
col_width       = '보호구역도로폭'
col_signals     = '신호등_150m_개수'

# ================================
# 2) 결측치 처리(간단: 중앙값 대체)
# ================================
num_cols = [col_accidents, col_speed, col_illegal, col_cctv, col_sign, col_bump, col_width, col_signals]
for c in num_cols:
    if c not in df.columns:
        raise KeyError(f"컬럼 '{c}' 를 찾을 수 없습니다. 실제 컬럼명을 확인해 주세요.")
    med = df[c].median()
    df[c] = df[c].fillna(med)

# ================================
# 3) Min-Max 정규화(0~1)
# ================================
def minmax(series):
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx - mn == 0:
        return pd.Series(0.0, index=s.index)  # 모두 같은 값이면 0으로
    return (s - mn) / (mx - mn)

dfn = pd.DataFrame(index=df.index)
dfn['사고건수*']           = minmax(df[col_accidents])
dfn['평균주행속도*']       = minmax(df[col_speed])
dfn['불법주정차위반건수*'] = minmax(df[col_illegal])
dfn['CCTV수*']            = minmax(df[col_cctv])
dfn['도로표지판수*']       = minmax(df[col_sign])
dfn['과속방지턱수*']       = minmax(df[col_bump])
dfn['도로폭*']            = minmax(df[col_width])
dfn['신호등수*']          = minmax(df[col_signals])   # ⇦ 추가

# ================================
# 4) 가중치 설정 (합=1)
#    기존 식에 '신호등수*' 0.05 추가, 일부 가중치 소폭 조정
# ================================
w = {
    '사고건수*'           : 0.36,  # 위험↑
    '평균주행속도*'       : 0.18,  # 위험↑
    '불법주정차위반건수*' : 0.14,  # 위험↑
    'CCTV수*'            : 0.10,  # 안전↑ → 1-값
    '도로표지판수*'       : 0.07,  # 안전↑ → 1-값
    '과속방지턱수*'       : 0.05,  # 안전↑ → 1-값
    '도로폭*'            : 0.05,  # 안전↑(해석에 따라 조정 가능)
    '신호등수*'          : 0.05   # 안전↑ → 1-값
}
assert abs(sum(w.values()) - 1.0) < 1e-8, "가중치 합이 1이 되도록 설정하세요."

# 값이 클수록 '위험이 낮다'고 보는 안전 변수 목록
safe_vars = {'CCTV수*', '도로표지판수*', '과속방지턱수*', '도로폭*', '신호등수*'}

# ================================
# 5) 위험도 Raw Score 계산
# ================================
raw = 0
for k, wk in w.items():
    if k in safe_vars:
        raw += wk * (1 - dfn[k])
    else:
        raw += wk * dfn[k]

df['위험도_raw'] = raw

# ================================
# 6) 0~100 점수로 변환
# ================================
rmin, rmax = df['위험도_raw'].min(), df['위험도_raw'].max()
if rmax - rmin == 0:
    df['위험도점수(0~100)'] = 50.0
else:
    df['위험도점수(0~100)'] = (df['위험도_raw'] - rmin) / (rmax - rmin) * 100

# ================================
# 7) 결과 저장/미리보기
# ================================



df_out = df.sort_values('위험도점수(0~100)', ascending=False)
df_out.to_csv("dataset7_risk_scored_with_signals.csv", index=False)

df_out
plt.scatter(df['위험도점수(0~100)'],df['사고건수'])

print("저장 완료: dataset7_risk_scored_with_signals.csv")
print(df_out[['시설명', '주소', '사고건수', '평균주행속도', '불법주정차위반건수', '신호등_150m_개수', '위험도점수(0~100)']].head(20))

plt.hist(df['신호등_150m_개수'])

sum(df['신호등_150m_개수'] >= 200)



'''
# pip install osmnx geopandas shapely pyproj
import os
import re
import time
import math
import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

# --------------------
# 0) OSMnx 설정
# --------------------
ox.settings.use_cache = True
ox.settings.timeout = 180

# --------------------
# 1) 데이터 로드 & 컬럼 정규화
# --------------------
df = pd.read_csv("data/dataset9.csv", encoding='utf-8')
df.info()

# 위경도 후보 사전
lat_candidates = ['lat','latitude','위도','y','Y','Y좌표','tm_y','TM_Y','utm_y']
lon_candidates = ['lon','lng','longitude','경도','x','X','X좌표','tm_x','TM_X','utm_x']

def find_col(cands):
    for c in df.columns:
        lc = c.strip().lower()
        if lc in cands:
            return c
    # 부분일치(한글/영문 혼합 대비)
    for c in df.columns:
        lc = c.strip().lower()
        if any(k in lc for k in cands):
            return c
    return None

lat_col = find_col(lat_candidates)
lon_col = find_col(lon_candidates)

if lat_col is None or lon_col is None:
    raise ValueError("위경도 컬럼을 찾지 못했습니다. CSV에서 위/경도 컬럼명을 확인해 주세요.")

# --------------------
# 2) 좌표계 판별 & 변환 (필요시)
# --------------------
# 간단한 휴리스틱: 경위도인지(범위), 투영좌표인지(큰 값)
sample_lat = df[lat_col].dropna().astype(float).head(20)
sample_lon = df[lon_col].dropna().astype(float).head(20)

def is_wgs84_like(lat_s, lon_s):
    if lat_s.empty or lon_s.empty:
        return True
    lat_ok = lat_s.between(20, 60).mean() > 0.8
    lon_ok = lon_s.between(100, 150).mean() > 0.8
    return lat_ok and lon_ok

wgs84_like = is_wgs84_like(sample_lat, sample_lon)

gdf = gpd.GeoDataFrame(df.copy(), geometry=None)

if wgs84_like:
    # 이미 WGS84 경위도로 가정
    gdf['lat_wgs84'] = df[lat_col].astype(float)
    gdf['lon_wgs84'] = df[lon_col].astype(float)
else:
    # 투영좌표(예: EPSG:5179)로 가정하고 변환 시도
    # ⚠️ 정확한 EPSG는 데이터 출처에 따라 다를 수 있습니다. 가장 흔한 국내 좌표계 5179로 우선 변환 시도.
    proj_crs = CRS.from_epsg(5179)
    gdf = gpd.GeoDataFrame(df.copy(),
                           geometry=gpd.points_from_xy(df[lon_col].astype(float),
                                                       df[lat_col].astype(float)),
                           crs=proj_crs)
    gdf = gdf.to_crs(epsg=4326)
    gdf['lat_wgs84'] = gdf.geometry.y
    gdf['lon_wgs84'] = gdf.geometry.x

# --------------------
# 3) 타겟 행 필터링 (도로폭 결측)
# --------------------
target_col = '보호구역도로폭'  # 필요시 변경
if target_col not in gdf.columns:
    raise ValueError(f"'{target_col}' 컬럼이 없습니다. 실제 도로폭 컬럼명을 코드에 반영해 주세요.")

targets = gdf[gdf[target_col].isna()].copy()
if targets.empty:
    print("결측 도로폭이 없습니다. 종료.")
else:
    print(f"결측 행 수: {len(targets)}")

# --------------------
# 4) width 파싱 유틸
# --------------------
def parse_width(val):
    """
    OSM width 예: '7', '7 m', '3.5;3.5', '7.0; 3.5', '7-8'
    - ';'로 구분되면 합계 또는 평균을 선택할 수 있는데, 폭은 보통 전체 폭이므로 합계가 맞지 않을 수도 있음.
      여기서는 최댓값을 대표값으로 사용.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # 단위/문자 제거, 분리
    parts = re.split(r'[;,\s/]+', s.replace('m','').replace('M',''))
    nums = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 범위 '7-8' 처리 -> 평균
        if '-' in p:
            try:
                a, b = p.split('-', 1)
                nums.append((float(a)+float(b))/2.0)
            except:
                pass
        else:
            try:
                nums.append(float(p))
            except:
                pass
    if not nums:
        return None
    # 보수적으로 최대값 사용
    return max(nums)

def estimate_from_lanes(row):
    """
    lanes 기반 폭 추정: (lanes_forward + lanes_backward + lanes) * 3.25m
    """
    def to_float(x):
        try:
            return float(str(x).strip())
        except:
            return np.nan

    lanes = row.get('lanes', np.nan)
    lf = row.get('lanes:forward', np.nan)
    lb = row.get('lanes:backward', np.nan)

    lanes = to_float(lanes)
    lf = to_float(lf)
    lb = to_float(lb)

    # 우선 순위: forward/backward 합 -> lanes
    total = 0.0
    cnt = 0
    if not np.isnan(lf):
        total += lf
        cnt += 1
    if not np.isnan(lb):
        total += lb
        cnt += 1
    if cnt == 0 and not np.isnan(lanes):
        total = lanes

    if total <= 0:
        return None
    return total * 3.25  # 차선 폭 평균값(한국 도시부 기준 3.0~3.5m 사이)

# --------------------
# 5) OSM 조회 함수 (반경 확대/타입 완화/최근접 엣지 시도)
# --------------------
def fetch_width_from_osm(lat, lon):
    # 반경과 네트워크 타입을 바꿔가며 시도
    dists = [150, 300, 500]
    net_types = ['drive_service', 'all_private', 'all']

    last_err = None

    for dist in dists:
        for nt in net_types:
            try:
                G = ox.graph_from_point((lat, lon), dist=dist, network_type=nt, simplify=True)
                if len(G.edges) == 0:
                    continue

                # 1) 최근접 엣지 우선
                try:
                    # nearest_edges 인자는 (X=lon, Y=lat)
                    nearest = ox.distance.nearest_edges(G, X=[lon], Y=[lat])
                    # osmnx>=2: nearest_edges returns array of tuples (u,v,key)
                    if hasattr(nearest, "__iter__"):
                        uvk = nearest[0]
                    else:
                        uvk = nearest
                    data = G.get_edge_data(*uvk)
                    # multi-edge dict 처리(k별로)
                    if isinstance(data, dict):
                        # 첫 edge 데이터
                        attrs = list(data.values())[0]
                    else:
                        attrs = data

                    w = parse_width(attrs.get('width'))
                    if w:
                        return w, 'OSM_width_nearest'

                    # lanes 기반 추정
                    w2 = estimate_from_lanes(attrs)
                    if w2:
                        return w2, 'OSM_lanes_est_nearest'
                except Exception as e:
                    last_err = e

                # 2) 주변 엣지 전체에서 폭 후보 수집
                edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
                if len(edges_gdf) == 0:
                    continue

                # width 직접 파싱
                widths = edges_gdf['width'].dropna().map(parse_width).dropna()
                if len(widths) > 0:
                    # 중앙값 사용
                    return float(np.median(widths)), 'OSM_width_med_edges'

                # lanes 기반 추정
                # lanes 관련 컬럼만 추출해서 추정
                ests = []
                for _, r in edges_gdf.iterrows():
                    w3 = estimate_from_lanes(r)
                    if w3:
                        ests.append(w3)
                if len(ests) > 0:
                    return float(np.median(ests)), 'OSM_lanes_med_edges'

            except Exception as e:
                last_err = e
                # Overpass 제한/일시 오류일 수 있으니 짧게 대기 후 재시도
                time.sleep(1)

    # 실패
    return None, f'fail({repr(last_err)})'

# --------------------
# 6) 그룹 중앙값(백업) 계산 (행정동/도로유형 등 그룹키를 자유롭게 바꾸세요)
# --------------------
# 예시: 같은 행정동 기준 중앙값
group_key = '행정동' if '행정동' in gdf.columns else None
group_median = None
if group_key:
    group_median = gdf[~gdf[target_col].isna()].groupby(group_key)[target_col].median()

# --------------------
# 7) 채우기 실행
# --------------------
filled_values = []
sources = []

for idx, row in targets.iterrows():
    lat = float(row['lat_wgs84'])
    lon = float(row['lon_wgs84'])

    # 경위도 유효성
    if not (20 <= lat <= 60 and 100 <= lon <= 150):
        filled_values.append(np.nan)
        sources.append('invalid_coord')
        continue

    w, src = fetch_width_from_osm(lat, lon)

    if w is None and group_key and pd.notna(row.get(group_key, np.nan)):
        # 그룹 중앙값 백업
        w = group_median.get(row[group_key], np.nan) if group_median is not None else np.nan
        src = 'group_median_backup' if pd.notna(w) else src

    filled_values.append(w)
    sources.append(src)

targets[target_col] = filled_values
targets['도로폭_출처'] = sources

# 결과 병합/저장
gdf.update(targets)
gdf.drop(columns=['geometry'], errors='ignore', inplace=True)
gdf.to_csv("dataset7_filled_osm.csv", index=False)
print("완료: dataset7_filled_osm.csv 저장")
'''