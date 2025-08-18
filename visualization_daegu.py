import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import re


df=pd.read_csv('./daegu/dataset7.csv', encoding='cp949')

df.info()
df.columns

# 텍스트 추가 및 주석 처리
plt.rc('font', family='Malgun Gothic')  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

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

# 0~7 숫자 코드 매핑 (북구=0, …, 달성군=7)
label_to_code = {
    '북구': 0, '남구': 1, '동구': 2, '서구': 3,
    '중구': 4, '수성구': 5, '달서구': 6, '달성군': 7
}
code_to_label = {v: k for k, v in label_to_code.items()}

# 코드 컬럼 생성
df['구군코드'] = df['구군'].map(label_to_code)


'''
1. 구군별 평균 구역 지정 수
'''

area = (
    df.groupby('구군코드')['구역지정수']
      .mean()
      .reindex(range(8))   # 0~7 순서 고정
)

# 막대 그래프 (x축은 코드, 라벨은 구군명으로 표시)
plt.figure(figsize=(8,5))
bars = plt.bar(area.index, area.values, edgecolor='black')
plt.title('구군(코드)별 평균 구역지정수')
plt.xlabel('구군코드 (0=북구, …, 7=달성군)')
plt.ylabel('평균 구역지정수')
plt.xticks(ticks=range(8), labels=[f"{i}\n({code_to_label[i]})" for i in range(8)])
plt.tight_layout()

# 막대 위에 값 표시
for b in bars:
    h = b.get_height()
    if pd.notna(h):
        plt.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
plt.show()


'''
2. 구군별 평균 시설물 cctv 개수
'''
df.columns

cctv = (
    df.groupby('구군코드')['시설물 CCTV 수']
      .mean()
      .reindex(range(8))   # 0~7 순서 고정
)

# 막대 그래프 (x축은 코드, 라벨은 구군명으로 표시)
plt.figure(figsize=(8,5))
bars = plt.bar(cctv.index, cctv.values, edgecolor='black')
plt.title('구군(코드)별 평균 시설물 CCTV 수')
plt.xlabel('구군코드 (0=북구, …, 7=달성군)')
plt.ylabel('평균 시설물 CCTV 수')
plt.xticks(ticks=range(8), labels=[f"{i}\n({code_to_label[i]})" for i in range(8)])
plt.tight_layout()

# 막대 위에 값 표시
for b in bars:
    h = b.get_height()
    if pd.notna(h):
        plt.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
plt.show()


'''
3. 구군별 평균 시설물 도로표지판 수
'''
df.columns

sign = (
    df.groupby('구군코드')['시설물 도로표지판 수']
      .mean()
      .reindex(range(8))   # 0~7 순서 고정
)

# 막대 그래프 (x축은 코드, 라벨은 구군명으로 표시)
plt.figure(figsize=(8,5))
bars = plt.bar(sign.index, sign.values, edgecolor='black')
plt.title('구군(코드)별 평균 시설물 도로표지판 수')
plt.xlabel('구군코드 (0=북구, …, 7=달성군)')
plt.ylabel('평균 시설물 도로표지판 수')
plt.xticks(ticks=range(8), labels=[f"{i}\n({code_to_label[i]})" for i in range(8)])
plt.tight_layout()

# 막대 위에 값 표시
for b in bars:
    h = b.get_height()
    if pd.notna(h):
        plt.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
plt.show()


'''
4. 구군별 평균 시설물 과속방지턱 수
'''
df.columns
speedbump = (
    df.groupby('구군코드')['시설물 과속방지턱 수']
      .mean()
      .reindex(range(8))   # 0~7 순서 고정
)

# 막대 그래프 (x축은 코드, 라벨은 구군명으로 표시)
plt.figure(figsize=(8,5))
bars = plt.bar(speedbump.index, speedbump.values, edgecolor='black')
plt.title('구군(코드)별 평균 시설물 과속방지턱 수')
plt.xlabel('구군코드 (0=북구, …, 7=달성군)')
plt.ylabel('평균 시설물 과속방지턱 수')
plt.xticks(ticks=range(8), labels=[f"{i}\n({code_to_label[i]})" for i in range(8)])
plt.tight_layout()

# 막대 위에 값 표시
for b in bars:
    h = b.get_height()
    if pd.notna(h):
        plt.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
plt.show()


'''
5. 구군별 평균 사고 건수 (2020-2024)
'''
df.columns
accident = (
    df.groupby('구군코드')['사고건수']
      .mean()
      .reindex(range(8))   # 0~7 순서 고정
)

# 막대 그래프 (x축은 코드, 라벨은 구군명으로 표시)
plt.figure(figsize=(8,5))
bars = plt.bar(accident.index, accident.values, edgecolor='black')
plt.title('구군(코드)별 평균 사고건수 (2020~2024)')
plt.xlabel('구군코드 (0=북구, …, 7=달성군)')
plt.ylabel('평균 사고건수')
plt.xticks(ticks=range(8), labels=[f"{i}\n({code_to_label[i]})" for i in range(8)])
plt.tight_layout()

# 막대 위에 값 표시
for b in bars:
    h = b.get_height()
    if pd.notna(h):
        plt.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
plt.show()


'''
6. 구군별 평균 주행속도
'''

df.columns
speed = (
    df.groupby('구군코드')['평균주행속도']
      .mean()
      .reindex(range(8))   # 0~7 순서 고정
)

# 막대 그래프 (x축은 코드, 라벨은 구군명으로 표시)
plt.figure(figsize=(8,5))
bars = plt.bar(speed.index, speed.values, edgecolor='black')
plt.title('구군(코드)별 평균 주행속도')
plt.xlabel('구군코드 (0=북구, …, 7=달성군)')
plt.ylabel('평균 주행속도')
plt.xticks(ticks=range(8), labels=[f"{i}\n({code_to_label[i]})" for i in range(8)])
plt.tight_layout()

# 막대 위에 값 표시
for b in bars:
    h = b.get_height()
    if pd.notna(h):
        plt.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
plt.show()


'''
7. 구군별 평균 불법주정차위반건수(2020-2024) 
'''

df.columns
parking = (
    df.groupby('구군코드')['불법주정차위반건수']
      .mean()
      .reindex(range(8))   # 0~7 순서 고정
)

# 막대 그래프 (x축은 코드, 라벨은 구군명으로 표시)
plt.figure(figsize=(8,5))
bars = plt.bar(parking.index, parking.values, edgecolor='black')
plt.title('구군(코드)별 평균 불법주정차위반건수(2020-2024)')
plt.xlabel('구군코드 (0=북구, …, 7=달성군)')
plt.ylabel('평균 불법주정차위반건수(2020-2024)')
plt.xticks(ticks=range(8), labels=[f"{i}\n({code_to_label[i]})" for i in range(8)])
plt.tight_layout()

# 막대 위에 값 표시
for b in bars:
    h = b.get_height()
    if pd.notna(h):
        plt.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
plt.show()


############## 전체 합친 그래프

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (선택) 윈도우 한글 폰트
# plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 구군 코드 → 라벨 (0~7)
labels = ['북구','남구','동구','서구','중구','수성구','달서구','달성군']
code_to_label = {i: name for i, name in enumerate(labels)}

def agg_mean(col):
    return (
        df.groupby('구군코드')[col]
          .mean()
          .reindex(range(8))
    )

# 각 그래프에 들어갈 시리즈와 제목/축라벨 정의
metrics = [
    (agg_mean('구역지정수'),        '구군(코드)별 평균 구역지정수',       '평균 구역지정수'),
    (agg_mean('시설물 CCTV 수'),    '구군(코드)별 평균 시설물 CCTV 수',   '평균 CCTV 수'),
    (agg_mean('시설물 도로표지판 수'), '구군(코드)별 평균 시설물 도로표지판 수', '평균 도로표지판 수'),
    (agg_mean('시설물 과속방지턱 수'), '구군(코드)별 평균 시설물 과속방지턱 수', '평균 과속방지턱 수'),
    (agg_mean('사고건수'),          '구군(코드)별 평균 사고건수 (2020~2024)','평균 사고건수'),
    (agg_mean('평균주행속도'),      '구군(코드)별 평균 주행속도',         '평균 주행속도'),
    (agg_mean('불법주정차위반건수'), '구군(코드)별 평균 불법주정차위반건수 (2020~2024)','평균 위반건수')
]

# 서브플롯 배치 (2열 고정)
n = len(metrics)
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

for i, (series, title, ylabel) in enumerate(metrics):
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

fig.suptitle('대구 구군별 지표 요약 (평균)', fontsize=16, y=0.995)
plt.tight_layout()
plt.show()

################################

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = 'Malgun Gothic'  # 필요 시 한글폰트
plt.rcParams['axes.unicode_minus'] = False

labels = ['북구','남구','동구','서구','중구','수성구','달서구','달성군']
code_to_label = {i: name for i, name in enumerate(labels)}

def agg_mean(col):
    return (
        df.groupby('구군코드')[col]
          .mean()
          .reindex(range(8))
    )

# --------------------------
# 1) 시설 종합점수 만들기
# --------------------------
infra_cols = ['구역지정수', '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수']
infra_means = pd.DataFrame({col: agg_mean(col) for col in infra_cols})

# z-score 표준화 후 평균 → 인프라점수
infra_z = (infra_means - infra_means.mean()) / infra_means.std(ddof=0)
infra_means['인프라점수'] = infra_z.mean(axis=1)

# 사고 평균
acc_mean = agg_mean('사고건수')

# --------------------------
# 2) 중앙값 기준 사분면 분류
# --------------------------
infra_med = infra_means['인프라점수'].median()
acc_med = acc_mean.median()

category = []
for idx in infra_means.index:
    infra_high = infra_means.loc[idx, '인프라점수'] >= infra_med
    acc_high = acc_mean.loc[idx] >= acc_med
    if infra_high and acc_high:
        category.append("시설↑ & 사고↑")
    elif not infra_high and not acc_high:
        category.append("시설↓ & 사고↓")
    elif infra_high and not acc_high:
        category.append("시설↑ & 사고↓")
    else:
        category.append("시설↓ & 사고↑")

infra_means['사고건수'] = acc_mean
infra_means['분류'] = category
infra_means['구군명'] = infra_means.index.map(code_to_label)

print("\n[시설↑ & 사고↑] 구군")
print(infra_means[infra_means['분류'] == "시설↑ & 사고↑"][['구군명', '인프라점수', '사고건수']])

print("\n[시설↓ & 사고↓] 구군")
print(infra_means[infra_means['분류'] == "시설↓ & 사고↓"][['구군명', '인프라점수', '사고건수']])

# --------------------------
# 3) 시각화 (산점도 + 사분면)
# --------------------------
plt.figure(figsize=(8,6))
colors = {
    "시설↑ & 사고↑": 'red',
    "시설↓ & 사고↓": 'green',
    "시설↑ & 사고↓": 'blue',
    "시설↓ & 사고↑": 'orange'
}


for category, color in colors.items():
    subset = infra_means[infra_means['분류'] == category]
    plt.scatter(subset['인프라점수'], subset['사고건수'],
                color=color, s=100, label=category)  # <-- label 추가
    for idx, row in subset.iterrows():
        plt.text(row['인프라점수']+0.02, row['사고건수'],
                 row['구군명'], fontsize=9)

plt.axvline(infra_med, color='gray', linestyle='--')
plt.axhline(acc_med, color='gray', linestyle='--')
plt.xlabel("인프라 종합점수(z-score 평균)")
plt.ylabel("평균 사고건수(2020~2024)")
plt.title("시설 수준 vs 사고건수 사분면 분석")
plt.grid(True)
plt.legend(title="분류", loc='best')  # <-- 범례 추가
plt.show()


# =========================
# 사분면 분석 + 비교 시각화
# =========================

# 0) 준비: 라벨/집계 함수는 기존 것 재사용
labels = ['북구','남구','동구','서구','중구','수성구','달서구','달성군']
code_to_label = {i: name for i, name in enumerate(labels)}

def agg_mean(col):
    return (
        df.groupby('구군코드')[col]
          .mean()
          .reindex(range(8))
    )

# 1) 인프라 종합점수 (z-score 표준화 후 평균)
infra_cols = ['구역지정수', '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수']
infra_means = pd.DataFrame({col: agg_mean(col) for col in infra_cols})

# 표준화
infra_z = (infra_means - infra_means.mean()) / infra_means.std(ddof=0)
infra_means['인프라점수'] = infra_z.mean(axis=1)

# 목표 변수들
acc_mean = agg_mean('사고건수')
has_violation = '불법주정차위반건수' in df.columns
if has_violation:
    viol_mean = agg_mean('불법주정차위반건수')

# 2) 중앙값 기준 사분면 분류(인프라 vs 사고)
infra_med = infra_means['인프라점수'].median()
acc_med = acc_mean.median()

def quad_label(infra_val, acc_val):
    if (infra_val >= infra_med) and (acc_val >= acc_med):
        return "시설↑ & 사고↑"
    elif (infra_val < infra_med) and (acc_val < acc_med):
        return "시설↓ & 사고↓"
    elif (infra_val >= infra_med) and (acc_val < acc_med):
        return "시설↑ & 사고↓"
    else:
        return "시설↓ & 사고↑"

infra_means['사고건수'] = acc_mean
infra_means['분류'] = [
    quad_label(infra_means.loc[i,'인프라점수'], infra_means.loc[i,'사고건수'])
    for i in infra_means.index
]
if has_violation:
    infra_means['불법주정차위반건수'] = viol_mean
infra_means['구군명'] = infra_means.index.map(code_to_label)

# 3) 산점도(사분면 + 범례)
plt.figure(figsize=(8,6))
colors = {
    "시설↑ & 사고↑": 'red',
    "시설↓ & 사고↓": 'green',
    "시설↑ & 사고↓": 'blue',
    "시설↓ & 사고↑": 'orange'
}

for category, color in colors.items():
    subset = infra_means[infra_means['분류'] == category]
    plt.scatter(subset['인프라점수'], subset['사고건수'],
                color=color, s=100, label=category)
    # 점 라벨(구군명)
    for idx, row in subset.iterrows():
        plt.text(row['인프라점수']+0.02, row['사고건수'],
                 row['구군명'], fontsize=9)

plt.axvline(infra_med, color='gray', linestyle='--')
plt.axhline(acc_med, color='gray', linestyle='--')
plt.xlabel("인프라 종합점수(z-score 평균)")
plt.ylabel("평균 사고건수(2020~2024)")
plt.title("시설 수준 vs 사고건수 사분면 분석")
plt.grid(True)
plt.legend(title="분류", loc='best')
plt.tight_layout()
plt.show()


# 2') 사분면 재정의: 1사분면 = 시설↑ & 사고↓
#    y축은 '사고건수'에 음수 부호를 붙여서(= 적을수록 ↑) 사용
infra_means['y_plot'] = -infra_means['사고건수']          # 안전점수: 사고 적을수록 큼(위로)
x_med = infra_means['인프라점수'].median()
y_med = infra_means['y_plot'].median()

def quad_label_reordered(x, y):
    if (x >= x_med) and (y >= y_med):
        return "① 시설↑ & 사고↓"   # 1사분면 (좋음-좋음)
    elif (x < x_med) and (y >= y_med):
        return "② 시설↓ & 사고↓"   # 2사분면 (시설만 아쉬움)
    elif (x < x_med) and (y < y_med):
        return "③ 시설↓ & 사고↑"   # 3사분면 (둘다 나쁨)
    else:
        return "④ 시설↑ & 사고↑"   # 4사분면 (사고만 아쉬움)

infra_means['사분면'] = [
    quad_label_reordered(infra_means.loc[i, '인프라점수'],
                         infra_means.loc[i, 'y_plot'])
    for i in infra_means.index
]

# 3') 산점도(재배치된 사분면 + 범례)
plt.figure(figsize=(8,6))
colors = {
    "① 시설↑ & 사고↓": 'tab:green',
    "② 시설↓ & 사고↓": 'tab:blue',
    "③ 시설↓ & 사고↑": 'tab:red',
    "④ 시설↑ & 사고↑": 'tab:orange'
}

for category, color in colors.items():
    subset = infra_means[infra_means['사분면'] == category]
    plt.scatter(subset['인프라점수'], subset['y_plot'],
                color=color, s=110, label=category, edgecolors='black', linewidths=0.5)
    for _, row in subset.iterrows():
        plt.text(row['인프라점수']+0.02, row['y_plot'],
                 row['구군명'], fontsize=9)

plt.axvline(x_med, color='gray', linestyle='--')
plt.axhline(y_med, color='gray', linestyle='--')
plt.xlabel("인프라 종합점수 (z-score 평균, 높을수록 좋음)")
plt.ylabel("안전점수 = -사고건수 (높을수록 사고 적음)")
plt.title("시설 수준 vs 사고 건수")
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(title="사분면(해석)", loc='best')
plt.tight_layout()
plt.show()


# 사분면별 불법주정차위반건수 평균만 보기
if '불법주정차위반건수' in infra_means.columns:
    viol_summary = (
        infra_means
        .groupby('분류')['불법주정차위반건수']
        .mean()
        .reindex(["시설↑ & 사고↑", "시설↑ & 사고↓", "시설↓ & 사고↑", "시설↓ & 사고↓"])
    )

    plt.figure(figsize=(7,5))
    viol_summary.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.ylabel("평균 불법주정차위반건수(2020~2024)")
    plt.xlabel("사분면 분류")
    plt.title("사분면별 평균 불법주정차위반건수")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n[사분면별 평균 불법주정차위반건수]")
    print(viol_summary.round(2))
else:
    print("⚠ 불법주정차위반건수 컬럼이 infra_means에 없습니다.")


# (선택) 위반을 별도로 보고 싶다면: 인프라 vs 위반 사분면도
if has_violation:
    # 1) y축을 '위반 적을수록 위'로 재정의
    infra_means['y_plot_v'] = -infra_means['불법주정차위반건수']   # 안전점수(위반): 적을수록 큼
    x_med = infra_med                                            # 인프라 중앙값 재사용
    y_med_v = infra_means['y_plot_v'].median()                   # 새 y 중앙값

    # 2) 사분면 라벨 재정의: ① 시설↑ & 위반↓ 를 1사분면
    def quad_label_v_reordered(x, y):
        if (x >= x_med) and (y >= y_med_v):
            return "① 시설↑ & 위반↓"   # 좋음-좋음
        elif (x < x_med) and (y >= y_med_v):
            return "② 시설↓ & 위반↓"   # 시설만 아쉬움
        elif (x < x_med) and (y < y_med_v):
            return "③ 시설↓ & 위반↑"   # 둘 다 나쁨
        else:
            return "④ 시설↑ & 위반↑"   # 위반만 아쉬움

    infra_means['사분면_위반'] = [
        quad_label_v_reordered(infra_means.loc[i, '인프라점수'],
                               infra_means.loc[i, 'y_plot_v'])
        for i in infra_means.index
    ]

    # 3) 산점도(재배치된 사분면 + 범례)
    plt.figure(figsize=(8,6))
    colors_v = {
        "① 시설↑ & 위반↓": 'tab:green',
        "② 시설↓ & 위반↓": 'tab:blue',
        "③ 시설↓ & 위반↑": 'tab:red',
        "④ 시설↑ & 위반↑": 'tab:orange'
    }

    for category, color in colors_v.items():
        subset = infra_means[infra_means['사분면_위반'] == category]
        plt.scatter(subset['인프라점수'], subset['y_plot_v'],
                    color=color, s=110, label=category,
                    edgecolors='black', linewidths=0.5)
        for _, row in subset.iterrows():
            plt.text(row['인프라점수']+0.02, row['y_plot_v'],
                     row['구군명'], fontsize=9)

    plt.axvline(x_med, color='gray', linestyle='--')
    plt.axhline(y_med_v, color='gray', linestyle='--')
    plt.xlabel("인프라 종합점수 (z-score 평균, 높을수록 좋음)")
    plt.ylabel("안전점수 = -불법주정차위반건수 (높을수록 위반 적음)")
    plt.title("시설 수준 vs 불법주정차위반건수 관계")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(title="사분면(위반 해석)", loc='best')
    plt.tight_layout()
    plt.show()
