import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import re
import math
import textwrap


df=pd.read_csv('./data/dataset.csv')
df.info()
df.columns
df


# 텍스트 추가 및 주석 처리
plt.rc('font', family='Malgun Gothic')  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

'''

# 주소 정리
def normalize(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00A0"," ").replace("\u3000"," ")
    return re.sub(r"\s+", " ", s).strip()


# 긴 토큰 우선 + 사이공백 허용 + 단어경계
pat = re.compile(r'(달서\s*구|달성\s*군|수성\s*구|북\s*구|남\s*구|동\s*구|서\s*구|중\s*구)\b')

df['구군'] = (
    df['주소'].astype(str).map(normalize)
      .str.extract(pat, expand=False)
      .str.replace(r'\s+', '', regex=True)   # '달서 구' -> '달서구'
)

df['구군'] = df['구군'].astype(str).str.strip()
'''

# 0~7 숫자 코드 매핑 (북구=0, …, 달성군=7)
label_to_code = {
    '북구': 0, '남구': 1, '동구': 2, '서구': 3,
    '중구': 4, '수성구': 5, '달서구': 6, '달성군': 7
}
code_to_label = {v: k for k, v in label_to_code.items()}

# 코드 컬럼 생성
df['구군코드'] = df['구군'].map(label_to_code)

df.columns


'''
1. 
대구 구군별 인프라 요소 평균

- 인프라 (+)
'시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수', '보호구역도로폭', '신호등 300m'

-인프라 (-)
'평균주행속도', '불법주정차위반건수', '어린이비율', '사고건수'

'''

# 구군 코드 → 라벨 (0~7)
labels = ['북구','남구','동구','서구','중구','수성구','달서구','달성군']
code_to_label = {i: name for i, name in enumerate(labels)}

def agg_mean(col):
    return (
        df.groupby('구군코드')[col]
          .mean()
          .reindex(range(8))
    )


# 인프라 (+)

metrics_good = [
    (agg_mean('시설물 CCTV 수'),    '구군(코드)별 평균 시설물 CCTV 수',   '평균 CCTV 수'),
    (agg_mean('시설물 도로표지판 수'), '구군(코드)별 평균 시설물 도로표지판 수', '평균 도로표지판 수'),
    (agg_mean('시설물 과속방지턱 수'), '구군(코드)별 평균 시설물 과속방지턱 수', '평균 과속방지턱 수'),
    (agg_mean('보호구역도로폭'),          '구군(코드)별 평균 보호구역도로폭','평균 보호구역도로폭'),
    (agg_mean('신호등 300m'),      '구군(코드)별 300m 이내 평균 신호등 개수',         '300m 이내 평균 신호등 개수')
]

n = len(metrics_good)
ncols = 2
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4*nrows))
axes = np.array(axes).ravel()  # 1차원 배열로 펼치기

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.text(p.get_x() + p.get_width()/2, h, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=9)

for i, (series, title, ylabel) in enumerate(metrics_good):
    ax = axes[i]
    x = series.index
    y = series.values
    bars = ax.bar(x, y, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('구군코드 (0=북구, …, 7=달성군)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{k}\n({code_to_label[k]})" for k in range(8)])
    annotate_bars(ax)

# 남는 축(개수 홀수일 때) 숨기기
for j in range(i+1, len(axes)):
    axes[j].axis('off')

fig.suptitle('대구 구군별 인프라(+) 지표 요약 (평균)', fontsize=16, y=0.995)
plt.tight_layout()
plt.show()

# 인프라 (-)
metrics_bad = [
    (agg_mean('평균주행속도'),        '구군(코드)별 평균 주행속도',       '평균 주행속도'),
    (agg_mean('불법주정차위반건수'),    '구군(코드)별 평균 불법 주정차 위반건수',   '평균 불법 주정차 위반건수'),
    (agg_mean('사고건수'), '구군(코드)별 평균 사고건수', '평균 사고건수')    
]

n = len(metrics_bad)
ncols = 2
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4*nrows))
axes = np.array(axes).ravel()  # 1차원 배열로 펼치기

for i, (series, title, ylabel) in enumerate(metrics_bad):
    ax = axes[i]
    x = series.index
    y = series.values
    bars = ax.bar(x, y, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('구군코드 (0=북구, …, 7=달성군)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{k}\n({code_to_label[k]})" for k in range(8)])
    annotate_bars(ax)

# 남는 축(개수 홀수일 때) 숨기기
for j in range(i+1, len(axes)):
    axes[j].axis('off')

fig.suptitle('대구 구군별 인프라(-) 지표 요약 (평균)', fontsize=16, y=0.995)
plt.tight_layout()
plt.show()

##################################
'''
2. 구군별 사분면 비교
'''

# =========================
# 1. 인프라 종합점수 만들기
# =========================

# 1) 지표 구성
pos_cols = ['시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수', '보호구역도로폭', '신호등 300m']  # 긍정 지표
neg_cols = ['평균주행속도', '불법주정차위반건수', '사고건수']  # 부정 지표 (값 ↑ = 나쁨)
infra_cols = pos_cols + neg_cols

# 2) 지역별 평균 테이블
infra_means = pd.DataFrame({col: agg_mean(col) for col in infra_cols})

# 3) Min-Max 정규화 (열 단위, 상수열 안전 처리)
def minmax(s):
    vmin, vmax = s.min(), s.max()
    if vmax == vmin:  # 모든 값이 동일하면 0으로 처리
        return pd.Series(0.0, index=s.index)
    return (s - vmin) / (vmax - vmin)

infra_norm01 = infra_means.apply(minmax, axis=0)

# 4) 방향 통일: 부정 지표는 1 - 정규화 (값 ↑ = 좋음이 되도록 뒤집기)
for c in neg_cols:
    infra_norm01[c] = 1 - infra_norm01[c]

# 5) 인프라 종합점수 (모든 지표 평균)
infra_means['인프라점수'] = infra_norm01.mean(axis=1)

# 6) 사고 평균(시각화 y축용, 원본 스케일)
acc_mean = agg_mean('사고건수')



# =========================
# 2-1. 대구 평균을 기준으로 사분면 분류
# =========================
infra_total = infra_means['인프라점수'].mean()   # 전체 인프라 점수 평균
acc_total   = acc_mean.mean()                   # 전체 사고건수 평균

category = []
for idx in infra_means.index:
    infra_high = infra_means.loc[idx, '인프라점수'] >= infra_total
    acc_high   = acc_mean.loc[idx] > acc_total
    if infra_high and acc_high:
        category.append("인프라↑ & 사고↑")
    elif (not infra_high) and (not acc_high):
        category.append("인프라↓ & 사고↓")
    elif infra_high and (not acc_high):
        category.append("인프라↑ & 사고↓")
    else:
        category.append("인프라↓ & 사고↑")

infra_means['사고건수'] = acc_mean
infra_means['분류'] = category
infra_means['구군명'] = infra_means.index.map(code_to_label)

# =========================
# 3-1. 시각화 (사분면 산점도)
# =========================
plt.figure(figsize=(8,6))

colors = {
    "인프라↑ & 사고↑": 'red',
    "인프라↓ & 사고↓": 'green',
    "인프라↑ & 사고↓": 'blue',
    "인프라↓ & 사고↑": 'orange'
}

for cat, color in colors.items():
    subset = infra_means[infra_means['분류'] == cat]
    plt.scatter(subset['인프라점수'], subset['사고건수'],
                color=color, s=100, label=cat)
    for _, row in subset.iterrows():
        plt.text(row['인프라점수']+0.01, row['사고건수'],
                 row['구군명'], fontsize=9)

# (B) 기준선: 대구 전체 평균
plt.axvline(infra_total, color='gray', linestyle='--', label='대구 인프라 평균')
plt.axhline(acc_total, color='gray', linestyle='--', label='대구 사고 평균')

plt.xlabel("인프라 종합점수 (Min-Max, 방향 통일)")
plt.ylabel("평균 사고건수(2020~2024)")
plt.title("인프라 수준 vs 사고건수 사분면 분석 (대구 전체 기준)")
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.show()



'''
3. 레이더 차트로 특정 구군 비교

'''
# 긍정적 지표
good_cols = [ '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수', '보호구역도로폭', '신호등 300m']

# 부정적 지표
bad_cols  = ['평균주행속도', '불법주정차위반건수'
, '사고건수']  

all_cols  = good_cols + bad_cols

# 구군별 평균 테이블(all_means) 
all_means = pd.DataFrame({col: agg_mean(col) for col in all_cols})
all_means['구군명'] = all_means.index.map(code_to_label)

# 대구 전체 평균 행 추가
total_row = all_means[all_cols].mean().to_frame().T
total_row['구군명'] = '대구 전체'

# index는 기존 것과 안 겹치게 None 처리 or 새 숫자 부여
total_row.index = [len(all_means)]  

# 합치기
all_means = pd.concat([all_means, total_row])


# ------------------------------------
# 1) 인프라 점수 표
# ------------------------------------
rules = {
    # ---- 긍정 지표 (값↑ = 좋음) ----
    '시설물 CCTV 수':        ([0, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, np.inf],
                              [10,20,30,40,50,60,70,80,90,100]),
    '시설물 도로표지판 수':  ([0, 10, 14, 18, 22, 26, 28, 30, 32, 34, np.inf],
                              [10,20,30,40,50,60,70,80,90,100]),
    '시설물 과속방지턱 수': ([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf],
                              [10,20,30,40,50,60,70,80,90,100]),
    '보호구역도로폭':        ([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, np.inf],
                              [10,20,30,40,50,60,70,80,90,100]),
    '신호등 300m':          ([0, 6, 7, 8, 9, 10, 11, 12, 13, 14, np.inf],
                              [10,20,30,40,50,60,70,80,90,100]),

    # ---- 부정 지표 (값↑ = 나쁨) → 라벨 역방향 ----
    '평균주행속도':          ([0, 30, 32, 34, 36, 38, 40, 42, 44, 46, np.inf],
                              [100,90,80,70,60,50,40,30,20,10]),
    '불법주정차위반건수':    ([0, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf],
                              [100,90,80,70,60,50,40,30,20,10]),
    '사고건수':              ([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, np.inf],
                              [100,90,80,70,60,50,40,30,20,10]),
}


# ------------------------------------
# 2) 점수화 함수
# ------------------------------------
def apply_absolute_scoring(df, rules):
    df_scored = pd.DataFrame(index=df.index)
    for col, (bins, labels) in rules.items():
        if col not in df.columns:
            raise KeyError(f"[오류] 데이터에 '{col}' 컬럼이 없습니다.")
        df_scored[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True).astype(int)
    return df_scored

score_table = apply_absolute_scoring(all_means[all_cols], rules)

# ------------------------------------
# 3) 레이더용 라벨 변경
# ------------------------------------
label_map = {
    '평균주행속도': '안전 주행 수준',
    '불법주정차위반건수': '불법 주정차 안전 수준',
    '사고건수': '사고 안전 수준',
}
radar_scores = score_table.rename(columns=label_map).copy()

# 구군명 붙이기
radar_scores['구군명'] = all_means['구군명']


# ------------------------------------
# 4) 레이더 차트 준비
#    (원래 이름 리스트 → 라벨맵으로 변환)
# ------------------------------------
radar_features_raw = good_cols + bad_cols
radar_features = [label_map.get(c, c) for c in radar_features_raw]

def wrap_labels(labels, width=6):
    """라벨을 공백 기준으로 줄바꿈해서 겹침 완화"""
    return ['\n'.join(textwrap.wrap(lbl, width=width)) for lbl in labels]

def make_angles(n_features):
    ang = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
    ang += ang[:1]
    return ang

def style_radar(ax, feature_names, angles, label_fs=12, pad_px=16):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # 라벨 줄바꿈 + 폰트 크기
    wrapped = wrap_labels(feature_names, width=6)
    ax.set_thetagrids(np.degrees(angles[:-1]), wrapped, fontsize=label_fs)
    # 라벨 패딩(그래프와 간격)
    ax.tick_params(axis='x', pad=pad_px)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 105)  # 0~100이지만 105로 약간 헤드룸
    ax.grid(True, linestyle='--', alpha=0.5)

def make_radar(ax, values, angles, label=None, alpha=0.18, linewidth=2):
    vals = values.tolist() + values.tolist()[:1]
    ax.plot(angles, vals, linewidth=linewidth, label=label, marker='o', markersize=4)
    ax.fill(angles, vals, alpha=alpha)
    


# ------------------------------------
# 5) 특정 구군 비교 
# ------------------------------------
compare_names = ['달서구', '중구']  
df_plot = radar_scores.set_index('구군명')

missing = [n for n in compare_names if n not in df_plot.index]
if missing:
    print(f"[경고] 데이터에 없는 구군: {missing} → 제외하고 그립니다.")
compare_use = [n for n in compare_names if n in df_plot.index]
if not compare_use:
    raise ValueError("선택한 구군이 데이터에 없습니다.")

angles = make_angles(len(radar_features))
sel = df_plot.loc[compare_use, radar_features]

# ------------------------------------
# 6) 그리기 (제목 겹침 방지: pad/y 조정 + tight_layout)
# ------------------------------------
plt.figure(figsize=(8.5, 8.5))
ax = plt.subplot(111, polar=True)
style_radar(ax, radar_features, angles)

for name, row in sel.iterrows():
    make_radar(ax, row, angles, label=name)

ax.set_title(
    '구군별 인프라·안전 지표 (절대기준 점수, 0~100)',
    y=1.12,            # 제목 위로 올려 겹침 방지
    fontsize=17,
    fontweight='bold',
    pad=14             # 폰트/레이블과 간격
)

plt.legend(loc='upper right', bbox_to_anchor=(1.22, 1.10), frameon=False)
plt.tight_layout()
plt.show()

# 0~7 숫자 코드 매핑 (북구=0, …, 달성군=7)
label_to_code = {
    '북구': 0, '남구': 1, '동구': 2, '서구': 3,
    '중구': 4, '수성구': 5, '달서구': 6, '달성군': 7
}
code_to_label = {v: k for k, v in label_to_code.items()}

# 코드 컬럼 생성
df['구군코드'] = df['구군'].map(label_to_code)

df.columns