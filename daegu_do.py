import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from rapidfuzz import fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from folium import Map, CircleMarker, LayerControl, FeatureGroup
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd, re, unicodedata

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



# 어린이 인구수 조인

df1 = pd.read_csv("data/dataset10.csv", encoding='utf-8')
df2 = pd.read_csv("data/dataset2.csv", encoding='cp949')



df1.info()
df2.info()

df_merged = pd.merge(
    df1, df2[['위도','경도','읍면동']], 
    on=['위도','경도'], 
    how='left'
)

df_merged.to_csv("dataset11.csv", index=False, encoding="utf-8-sig")


df1 = pd.read_csv("data/dataset11.csv", encoding='utf-8-sig')
df2 = pd.read_csv("data/어린이인구.csv", encoding='utf-8-sig')

df_merged = pd.merge(
    df1, df2, 
    on='읍면동', 
    how='left'
)

df_merged.to_csv("dataset12.csv", index=False, encoding="utf-8-sig")
df_merged.to_csv("dataset13.csv", index=False, encoding="cp949")

df2.to_csv("어린이인구1.csv", index=False, encoding="utf-8-sig")

df = pd.read_csv("data/dataset.csv", encoding='utf-8-sig')
df.info()
df.to_csv("dataset1.csv", index=False, encoding="utf-8-sig")
df.to_csv("dataset2.csv", index=False, encoding="cp949")


# 결합 끝 분석 시작
df = pd.read_csv("data/dataset1.csv", encoding='utf-8-sig')
df.info()

col_accidents   = '사고건수'
col_speed       = '평균주행속도'
col_illegal     = '불법주정차위반건수'
col_cctv        = '시설물 CCTV 수'
col_sign        = '시설물 도로표지판 수'
col_bump        = '시설물 과속방지턱 수'
col_width       = '보호구역도로폭'
col_signals     = '신호등_150m_개수'


# 분석 시작
# EDA

df = pd.read_csv("data/dataset1.csv", encoding='utf-8')
df.to_csv("dataset1.csv", index=False, encoding='utf-8-sig')

df.info()

df['전체인구'] = df['전체인구'].str.replace(',', '', regex=False).astype(int)
df['어린이인구'] = df['어린이인구'].str.replace(',', '', regex=False).astype(int)
df['어린이비율'] = (df['어린이인구'] / df['전체인구'])*100
df['어린이비율'] = df['어린이비율'].round(2)
df['어린이비율'].describe()
df.info()

del df['위험도점수']
del df['RawScore']
del df['신호등_150m_개수']

df.info()

risk_vars = ['사고건수', '평균주행속도', '불법주정차위반건수', '어린이비율', '보호구역도로폭']
protect_vars = ['시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수', '신호등 300m']

# 2) Min-Max 스케일링 (0~1)
all_vars = risk_vars + protect_vars
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(df[all_vars]), columns=all_vars, index=df.index)

# 3) 보호 요인은 방향 뒤집기 (값↑ → 위험↓ → 반대로 계산해야 함)
for c in protect_vars:
    scaled[c] = 1 - scaled[c]

# 4) 가중치 설정 (사용자가 지정한 값)
weights = {
    '사고건수': 0.3,
    '평균주행속도': 0.13,
    '불법주정차위반건수': 0.10,
    '어린이비율': 0.07,
    '보호구역도로폭': 0.05,
    '시설물 CCTV 수': 0.10,
    '시설물 도로표지판 수': 0.05,
    '시설물 과속방지턱 수': 0.10,
    '신호등 300m': 0.10
}

# 5) 위험도 점수 계산 (가중합 → 0~100 점수화)
df['위험도점수'] = sum(scaled[col] * w for col, w in weights.items())
df['위험도점수'] = df['위험도점수'] / sum(weights.values()) * 100
scaler = MinMaxScaler(feature_range=(0,100))
df['위험도점수_norm'] = scaler.fit_transform(df[['위험도점수']])
print(df['위험도점수_norm'].describe())

del df['위험도점수_norm']

df.to_csv("dataset.csv", index=False, encoding='utf-8-sig')

df['위험도점수'].describe()

np.sum(df['위험도점수'] >= 46)

df['위험도점수'].hist()
# 6) 결과 확인
print(df[['시설명','사고건수','위험도점수']].head())




plt.scatter(df['위험도점수_노사고'], df['사고건수'])
plt.scatter(df['위험도점수'], df['사고건수'])



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

df.info()


df = pd.read_csv("data/dataset.csv", encoding='utf-8')
pat = re.compile(r'(달서\s*구|달성\s*군|수성\s*구|북\s*구|남\s*구|동\s*구|서\s*구|중\s*구)\b')

def normalize(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00A0"," ").replace("\u3000"," ")
    return re.sub(r"\s+", " ", s).strip()

pat = re.compile(r'(달서\s*구|달성\s*군|수성\s*구|북\s*구|남\s*구|동\s*구|서\s*구|중\s*구)\b')

df["구군"] = (
    df["주소"].astype(str).map(normalize)
      .str.extract(pat, expand=False)
      .str.replace(r"\s+", "", regex=True)
      .astype(str).str.strip()
)
df.to_csv("결합2.csv", index=False, encoding='utf-8-sig')
# 군집분석
df.info()

feature_cols = ['사고건수', 
    '평균주행속도', '불법주정차위반건수', '어린이비율',
    '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수',
    '보호구역도로폭', '신호등 300m', '위도', '경도' # 필요 없으면 빼도 됩니다.
]
X = df[feature_cols].copy()

'''
# (선택) 카운트형의 왜도 완화: log1p
log_candidates = ['불법주정차위반건수', '시설물 CCTV 수', '시설물 도로표지판 수',
                  '시설물 과속방지턱 수', '신호등_150m_개수', '구역지정수']
for c in feature_cols:
    if c in log_candidates:
        X[c] = np.log1p(X[c].clip(lower=0))
'''
# 3) 표준화
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

'''
# 4) k 후보에 대해 실루엣 계수 계산
k_range = range(2, 9)  # k=2~8 시험
sil_scores = {}
sse = {}  # 엘보우 참고용

for k in k_range:
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    sil_scores[k] = sil
    sse[k] = km.inertia_

print("Silhouette scores by k:", sil_scores)
print("SSE (inertia) by k:", sse)

# 5) 실루엣 최고 k 선택
best_k = max(sil_scores, key=sil_scores.get)
print("Selected k (by silhouette):", best_k)
'''

# 6) 최종 KMeans 학습 및 라벨 부여
kmeans = KMeans(n_clusters=5, n_init=50, random_state=42)
df['cluster'] = kmeans.fit_predict(Xs)
df.info()

group_mean = df.groupby('cluster')[['사고건수',
    '평균주행속도', '불법주정차위반건수', '어린이비율',
    '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수',
    '보호구역도로폭', '신호등 300m',
    '위험도점수'
]].mean().round(2)
print(group_mean)

group_stats = df.groupby("cluster")[[
    "사고건수", "평균주행속도", "불법주정차위반건수", "어린이비율",
    "시설물 CCTV 수", "시설물 도로표지판 수", "시설물 과속방지턱 수",
    "보호구역도로폭", "신호등 300m", "위험도점수"
]].describe().round(2)

print(group_stats)

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# ✅ 원하는 변수 리스트 직접 지정
variables = [
    "사고건수",
    "평균주행속도",
    "불법주정차위반건수",
    "어린이비율",
    "시설물 CCTV 수",
    "시설물 도로표지판 수",
    "시설물 과속방지턱 수",
    "보호구역도로폭",
    "신호등 300m",
    "위험도점수"
]

# ✅ 변수별 boxplot
for col in variables:
    if col not in df.columns:
        print(f"⚠️ {col} 컬럼이 데이터에 없습니다.")
        continue
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="cluster", y=col, data=df, palette="Set2")
    plt.title(f"Cluster별 {col} 분포", fontsize=14)
    plt.xlabel("Cluster")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()



center = [df['위도'].mean(), df['경도'].mean()]
m = Map(location=center, zoom_start=12, control_scale=True, tiles="CartoDB positron")


gu_geo = gpd.read_file("data/BND/BND_SIGUNGU_PG.shp")

# 2) 좌표계를 WGS84(경위도)로 변환 (Folium은 EPSG:4326 필요)
if gu_geo.crs is None or gu_geo.crs.to_string().upper() not in ("EPSG:4326", "WGS84"):
    gu_geo = gu_geo.to_crs(epsg=4326)

# 3) 대구시 구·군만 필터링
DAEGU_GU_LIST = ['수성구','달서구','달성군','동구','북구','중구','서구','남구']
gu_daegu = gu_geo[gu_geo['SIGUNGU_NM'].isin(DAEGU_GU_LIST)].copy()

# 4) (선택) 대구시 외곽 경계(하나의 폴리곤) 생성
daegu_boundary = gu_daegu.dissolve(by=lambda x: 1)  # 모든 구/군을 하나로 합치기
daegu_boundary['name'] = '대구광역시'

# 5) 군·구 경계선 레이어 (검은 점선)
folium.GeoJson(
    gu_daegu,
    name="대구 군·구 경계",
    style_function=lambda feat: {
        "color": "black",
        "weight": 2,
        "dashArray": "5,5",
        "fillOpacity": 0
    },
    tooltip=folium.GeoJsonTooltip(fields=["SIGUNGU_NM"], aliases=["군·구"])
).add_to(m)

# 6) 대구 전체 외곽선(굵은 실선)
folium.GeoJson(
    daegu_boundary,
    name="대구 외곽",
    style_function=lambda feat: {
        "color": "#222222",
        "weight": 3,
        "fillOpacity": 0
    },
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["지역"])
).add_to(m)

for idx, row in gu_daegu.iterrows():
    centroid = row['geometry'].centroid
    folium.map.Marker(
        [centroid.y, centroid.x],
        icon=folium.DivIcon(
            html=f"""
                <div style="font-size:14px; font-weight:bold; color:#333;">
                    {row['SIGUNGU_NM']}
                </div>
            """
        )
    ).add_to(m)

# 7) 지도 범위를 대구 전체로 맞춤 (시설 포인트보다 경계에 맞추고 싶을 때)
minx, miny, maxx, maxy = gu_daegu.total_bounds
m.fit_bounds([[miny, minx], [maxy, maxx]])

# 3) 군집 라벨 수집 및 색상 매핑
clusters = sorted(df['cluster'].unique())
palette = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999"
] * 5  # 군집이 많아도 반복 사용

color_map = {c: palette[i] for i, c in enumerate(clusters)}

# 4) (선택) 위험도 점수에 따라 마커 반지름을 가볍게 스케일링
def radius_by_risk(row, base=4, add=6):
    if '위험도점수(0~100)' in row:
        r = base + add * (row['위험도점수(0~100)'] / 100.0)  # 4~10px
        return float(r)
    return 6.0

# 5) 군집별 FeatureGroup 레이어 생성 + 마커 추가
for c in clusters:
    fg = FeatureGroup(name=f"Cluster {c}", show=True)
    sub = df[df['cluster'] == c]
    for _, r in sub.iterrows():
        lat, lon = float(r['위도']), float(r['경도'])
        color = color_map[c]
        # 팝업/툴팁 내용 구성
        fac = r.get('시설명', '')
        addr = r.get('주소', '')
        risk = r.get('위험도점수(0~100)', None)
        risk_txt = f"{risk:.1f}" if pd.notna(risk) else "N/A"

        popup_html = folium.Html(f"""
        <div style="font-size:13px;">
          <b>시설명:</b> {fac}<br>
          <b>주소:</b> {addr}<br>
          <b>Cluster:</b> {c}<br>
          <b>위험도점수:</b> {risk_txt}
        </div>
        """, script=True)
        popup = folium.Popup(popup_html, max_width=300)

        CircleMarker(
            location=[lat, lon],
            radius=radius_by_risk(r),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            popup=popup,
            tooltip=f"{fac} | Cluster {c}"
        ).add_to(fg)

    fg.add_to(m)

# 6) 범례(간단 텍스트) 추가
legend_html = """
<div style="
     position: fixed; bottom: 20px; left: 20px; z-index: 9999;
     background-color: white; padding: 10px 12px; border: 1px solid #ddd;
     box-shadow: 0 1px 4px rgba(0,0,0,0.2); border-radius: 6px; font-size:13px;">
  <b>Clusters</b><br>
  {}
</div>
""".format("<br>".join(
    [f'<span style="display:inline-block;width:10px;height:10px;background:{color_map[c]};margin-right:6px;"></span> Cluster {c}'
     for c in clusters]
))
m.get_root().html.add_child(folium.Element(legend_html))

df.columns

# 7) 레이어 컨트롤 버튼
LayerControl(collapsed=False).add_to(m)


m.save("군집.html")
print("저장 완료: clusters_map11.html  (브라우저로 열면 지도 확인 가능)")


# 회귀분석

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("data/dataset.csv", encoding='utf-8')
df.info()
df1 = df.groupby('읍면동')[['구역지정수', '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수',
       '보호구역도로폭', '사고건수', '평균주행속도', '불법주정차위반건수','신호등 300m','어린이인구','전체인구']].mean()
del df1['어린이비율']
df1['어린이비율'] = df1['어린이인구'] / df1['전체인구']
df1.info()

df1.describe()

del df1['어린이인구']
del df1['전체인구']
df1.describe()


X = df1.drop(columns=["사고건수"])   # 독립변수
y = df1["사고건수"]                # 종속변수

# 학습/검증 데이터 분리 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 1. 스케일링 (표준화)
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# 2. 선형회귀 모델 학습
# =============================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 예측
y_pred = model.predict(X_test_scaled)

# =============================
# 3. 성능 평가
# =============================
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("📊 선형회귀 분석 결과")
print("R² (결정계수):", r2)
print("MSE (평균제곱오차):", mse)

# =============================
# 4. 회귀계수 확인 (스케일링된 기준)
# =============================
coef_df = pd.DataFrame({
    "변수": X.columns,
    "회귀계수": model.coef_
}).sort_values(by="회귀계수", ascending=False)

print("\n회귀계수 (표준화된 데이터 기준):")
print(coef_df)

print("\n절편(intercept):", model.intercept_)
import statsmodels.api as sm

# 종속변수와 독립변수 설정
X = df1.drop(columns=["사고건수"])
y = df1["사고건수"]

# 상수항(intercept) 추가
X = sm.add_constant(X)

# OLS 회귀 모델 적합
model = sm.OLS(y, X).fit()

# 결과 요약 출력
print(model.summary())

import statsmodels.api as sm
import statsmodels.formula.api as smf

# 데이터프레임 df에서 종속변수(y)와 독립변수(X) 설정
# 여기서는 전체 열 중 "사고건수"를 종속변수로 하고 나머지를 독립변수로 사용
# (필요없는 변수는 drop하세요)

formula = "사고건수 ~ 구역지정수 + Q('시설물 CCTV 수') + Q('시설물 도로표지판 수') + Q('시설물 과속방지턱 수') + 보호구역도로폭 + 평균주행속도 + 불법주정차위반건수 + Q('신호등 300m') + 어린이비율"

# 포아송 회귀 적합
poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()

# 결과 요약
print(poisson_model.summary())


import geopandas as gpd
g = gpd.read_file("./data/N3A_G0100000.shp")
print(g.columns)
print(g.head())

# -*- coding: utf-8 -*-
import os
import geopandas as gpd
import folium
from shapely.ops import unary_union

# -----------------------
# 경로 설정
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHP_PATH = os.path.join(BASE_DIR, "data", "N3A_G0100000.shp")

# -----------------------
# Shapefile 읽기
# -----------------------
g = gpd.read_file(SHP_PATH)
g = g.to_crs(4326) if g.crs else g.set_crs(5179, allow_override=True).to_crs(4326)

# ✅ GU 컬럼 생성 (이름)
if "NAME" in g.columns:
    g["GU"] = g["NAME"].astype(str).str.strip()
else:
    g["GU"] = g.iloc[:,0].astype(str)

# ✅ 대구 8개 구/군만 추출
DIST_LABELS = ["북구", "남구", "동구", "서구", "중구", "수성구", "달서구", "달성군"]
dg = g[g["GU"].isin(DIST_LABELS)][["GU","geometry"]].reset_index(drop=True)

# -----------------------
# 중심 좌표 계산
# -----------------------
def _centroid_latlon(gdf):
    ua = getattr(gdf.geometry, "union_all", None)
    if callable(ua):
        u = gdf.geometry.union_all()
    else:
        u = unary_union(gdf.geometry.values)
    c = u.centroid
    return float(c.y), float(c.x)

lat, lon = _centroid_latlon(dg)

# -----------------------
# folium 지도 생성
# -----------------------
m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")

for _, row in dg.iterrows():
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x: {"color": "blue", "weight": 2, "fillOpacity": 0.1},
        tooltip=row["GU"]
    ).add_to(m)

# -----------------------
# HTML 저장
# -----------------------
SAVE_PATH = os.path.join(BASE_DIR, "test_map.html")
m.save(SAVE_PATH)

print(f"✅ 지도 저장 완료: {SAVE_PATH}")
print(f"✅ 파일 크기: {os.path.getsize(SAVE_PATH)/1024/1024:.2f} MB")
