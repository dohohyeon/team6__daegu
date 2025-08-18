import numpy as np
import pandas as pd
import re

child=pd.read_csv('./daegu/대구어린이보호구역.csv',encoding='cp949')

child.info()

child = child.drop(columns=['Unnamed: 0'], errors='ignore')


# 찾고 싶은 구 목록
gu_list = ['북구', '남구', '동구', '서구', '중구', '수성구', '달서구', '달성군']

addr_col = '소재지도로명주소'

# 공백 정리
child[addr_col] = child[addr_col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

# 구별 데이터프레임 저장 딕셔너리
gu_dfs = {}

for target_gu in gu_list:
    pattern = rf'(?<![가-힣0-9]){target_gu}(?![가-힣0-9])'
    df_gu = child[child[addr_col].str.contains(pattern, na=False)].copy()
    gu_dfs[target_gu] = df_gu  # 각 구별 DataFrame 저장




'''
구 단위 별로 어린이보호구역 위치 파악,
해당 구의 불법주정차 내역 (2020~2024)을 통해서
어린이보호구역 내에서, 불법주정차가 몇 회 일어났는지 파악
'''

# 1. 북구
bukgu_2=pd.read_csv('./daegu/대구광역시_북구_불법주정차위반정보_20220704.csv', encoding='cp949')
bukgu_2

# 날짜 파싱 (여러 포맷: YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD, YYYYMMDD 등 자동 인식)
d = pd.to_datetime(bukgu_2['위반일자'].astype(str).str.strip(), errors='coerce')

# 2020-01-01 이상만 남기기
bukgu_2 = bukgu_2[(d >= pd.Timestamp('2020-01-01'))].copy()

bukgu_3=pd.read_csv('./daegu/대구광역시_북구_불법주정차위반정보_20230704.csv', encoding='cp949')

# 날짜 파싱 (여러 포맷: YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD, YYYYMMDD 등 자동 인식)
d = pd.to_datetime(bukgu_3['위반일자'].astype(str).str.strip(), errors='coerce')

# 2022-01-01 이상만 남기기
bukgu_3 = bukgu_3[(d >= pd.Timestamp('2022-01-01'))].copy()

bukgu_4=pd.read_csv('./daegu/대구광역시_북구_불법주정차위반정보_20231130.csv', encoding='cp949')

bukgu_5=pd.read_csv('./daegu/대구광역시_북구_불법주정차위반정보_20240603.csv', encoding='cp949')

bukgu_6=pd.read_csv('./daegu/대구광역시_북구_불법주정차위반정보_20250101.csv', encoding='cp949')

# '위반일시'를 datetime으로 변환
bukgu_6['위반일시'] = pd.to_datetime(bukgu_6['위반일시'], errors='coerce')

# 날짜와 시각으로 분리
bukgu_6['위반일자'] = bukgu_6['위반일시'].dt.date
bukgu_6['위반시간'] = bukgu_6['위반일시'].dt.time
bukgu_6 = bukgu_6.drop(columns=['위반일시'])
bukgu_6

df_bukgu = pd.concat([bukgu_2,bukgu_3,bukgu_4,bukgu_5,bukgu_6], ignore_index=True)



# 2. 남구
namgu_2=pd.read_csv('./daegu/대구광역시_남구_불법주정차단속현황_20241030.csv',encoding='cp949')

namgu_2.info()

d = pd.to_datetime(namgu_2['위반일자'].astype(str).str.strip(), errors='coerce')

namgu_2020_2024 = namgu_2[(d >= pd.Timestamp('2020-01-01'))].copy()

df_namgu = namgu_2020_2024.sort_values(by='위반일자').reset_index(drop=True)




# 3. 동구
donggu_2=pd.read_csv('./daegu/동구_20200731.csv',encoding='cp949') #2016-20200731까지
donggu_2=donggu_2.sort_values(by='위반일자').reset_index(drop=True)
# 날짜 파싱 (여러 포맷: YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD, YYYYMMDD 등 자동 인식)
d = pd.to_datetime(donggu_2['위반일자'].astype(str).str.strip(), errors='coerce')

# 2020-01-01 이상만 남기기
donggu_2 = donggu_2[(d >= pd.Timestamp('2020-01-01'))].copy()

donggu_3=pd.read_csv('./daegu/동구_20220705.csv',encoding='cp949') # 20210801- 20220630까지

donggu_4=pd.read_csv('./daegu/동구_20230531.csv',encoding='cp949') # 20210801-20230531까지. 날려야됨!

d = pd.to_datetime(donggu_4['위반일자'].astype(str).str.strip(), errors='coerce')

# 20220630 초과만 남기기
donggu_4 = donggu_4[(d > pd.Timestamp('2022-06-30'))].copy()

donggu_5=pd.read_csv('./daegu/동구_20240521.csv',encoding='cp949')
# '위반일시'를 datetime으로 변환
donggu_5['위반일시'] = pd.to_datetime(donggu_5['위반일시'], errors='coerce')

# 날짜와 시간 분리
donggu_5['위반일자'] = donggu_5['위반일시'].dt.date
donggu_5['위반시간'] = donggu_5['위반일시'].dt.strftime('%H:%M')  # HH:MM 형식

# 기존 '위반일시' 컬럼 삭제
donggu_5 = donggu_5.drop(columns=['위반일시'])

donggu_6=pd.read_csv('./daegu/동구_20240521이후.csv',encoding='cp949')
# '위반일시'를 datetime으로 변환
donggu_6['위반일시'] = pd.to_datetime(donggu_6['위반일시'], errors='coerce')

# 날짜와 시간 분리
donggu_6['위반일자'] = donggu_6['위반일시'].dt.date
donggu_6['위반시간'] = donggu_6['위반일시'].dt.strftime('%H:%M')  # HH:MM 형식
# 기존 '위반일시' 컬럼 삭제
donggu_6 = donggu_6.drop(columns=['위반일시'])

d = pd.to_datetime(donggu_6['위반일자'].astype(str).str.strip(), errors='coerce')

# 20250101 이전만 남기기
donggu_6 = donggu_6[(d < pd.Timestamp('2025-01-01'))].copy()

df_donggu = pd.concat([donggu_2,donggu_3,donggu_4,donggu_5, donggu_6], ignore_index=True)

# 중복 행 제거
df_donggu = df_donggu.drop_duplicates()

df_donggu.to_csv('동구불법주정차.csv', index=False, encoding='cp949')


# 4. 서구

seogu_2=pd.read_csv('./daegu/대구광역시_서구_불법주정차단속현황_20241231.csv',
encoding='cp949')

# 날짜 파싱 (여러 포맷: YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD, YYYYMMDD 등 자동 인식)
d = pd.to_datetime(seogu_2['위반일자'].astype(str).str.strip(), errors='coerce')

# 2020-01-01 이상만 남기기
df_seogu = seogu_2[(d >= pd.Timestamp('2020-01-01'))].copy()
df_seogu = df_seogu.sort_values(by='위반일자').reset_index(drop=True)


# 5. 중구
junggu_2=pd.read_csv('./daegu/중구_2020.csv',encoding='cp949')
junggu_3=pd.read_csv('./daegu/중구_2021.csv')
junggu_4=pd.read_csv('./daegu/중구_2022.csv',encoding='cp949')
junggu_5=pd.read_csv('./daegu/중구_2023.csv',encoding='cp949')

# 하나의 데이터프레임으로 병합 
df_junggu = pd.concat([junggu_2,junggu_3,junggu_4,junggu_5], ignore_index=True)

# 중복 행 제거
df_junggu = df_junggu.drop_duplicates()



# 6. 수성구
suseonggu_2=pd.read_csv('./daegu/수성구_2020_12.csv', encoding='cp949') # 20201231까지
suseonggu_3=pd.read_csv('./daegu/수성구_2021_04.csv',encoding='cp949') # 20210126까지
suseonggu_4=pd.read_csv('./daegu/수성구_2022_04.csv',encoding='cp949') # 20220412까지
suseonggu_5=pd.read_csv('./daegu/수성구_2024.csv',encoding='cp949') # 20241210까지

# 하나의 데이터프레임으로 병합 
df_suseouggu = pd.concat([suseonggu_2, suseonggu_3, suseonggu_4, suseonggu_5], ignore_index=True)

# 중복 행 제거
df_suseouggu = df_suseouggu.drop_duplicates()

# 날짜 컬럼 변환
df_suseouggu['단속일자'] = pd.to_datetime(
    df_suseouggu['단속일자'].astype(str).str.strip(),
    errors='coerce'
)

# 2020-01-01 이상만 남기기
df_suseonggu = df_suseouggu[df_suseouggu['단속일자'] >= pd.Timestamp('2020-01-01')]

df_suseonggu = df_suseonggu.sort_values(by='단속일자').reset_index(drop=True)


# 7. 달서구
dalseogu_2=pd.read_csv('./daegu/달서구_2021.csv', encoding='cp949') # 2020
dalseogu_3=pd.read_csv('./daegu/달서구_2022.csv',encoding='cp949') # 2017?-2021
dalseogu_4=pd.read_csv('./daegu/달서구_20230101.csv',encoding='cp949') # 2017-20221231
dalseogu_5=pd.read_csv('./daegu/달서구_20231130.csv',encoding='cp949') # 20230101-20231130
dalseogu_6=pd.read_csv('./daegu/달서구_2024.csv',encoding='cp949') # 20231201-20240731

# 하나로 합치기
df_dalseogu = pd.concat(
    [dalseogu_2, dalseogu_3, dalseogu_4, dalseogu_5, dalseogu_6],
    ignore_index=True
)

# '위반일자'를 날짜형으로 변환
df_dalseogu['위반일자'] = pd.to_datetime(
    df_dalseogu['위반일자'].astype(str).str.strip(),
    errors='coerce'
)

# 2020-01-01 이후 데이터만 남기기
df_dalseogu = df_dalseogu[df_dalseogu['위반일자'] >= pd.Timestamp('2020-01-01')]

# 중복 제거
df_dalseogu = df_dalseogu.drop_duplicates()

# 날짜 기준 오름차순 정렬 + 인덱스 재설정
df_dalseogu = df_dalseogu.sort_values(by='위반일자').reset_index(drop=True)



# 8. 달성군
dalseounggun_2=pd.read_csv('./daegu/대구광역시_달성군_불법주정차단속현황_20250321.csv', encoding='cp949')

# '단속일자'를 날짜형으로 변환
dalseounggun_2['단속일자'] = pd.to_datetime(
    dalseounggun_2['단속일자'].astype(str).str.strip(),
    errors='coerce'
)

# 2020-01-01 이후 데이터만 남기기
dalseounggun_2 = dalseounggun_2[dalseounggun_2['단속일자'] >= pd.Timestamp('2020-01-01')]

# 2024-12-31 까지만
dalseounggun_2 = dalseounggun_2[dalseounggun_2['단속일자'] < pd.Timestamp('2025-01-01')]

# 중복 제거
df_dalseounggun = dalseounggun_2.drop_duplicates()

# 날짜 기준 오름차순 정렬 + 인덱스 재설정
df_dalseounggun = df_dalseounggun.sort_values(by='단속일자').reset_index(drop=True)


'''
어린이보호구역과 연결하기
'''

# 1. 북구

gu_dfs['북구'].info()
df_bukgu.info()
df_bukgu

# --- 준비물 ---
zones = gu_dfs['북구'].copy()      # 남구 어린이보호구역 (대상시설명/주소/위경도 포함)
viol  = df_bukgu.copy()           # 남구 위반 데이터: ['위반일자','위반시간','위반장소명']


# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['위반장소'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"북구 전체 위반 건수: {len(viol):,}")
print(f"북구 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('북구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')
###################3

zones = gu_dfs['북구'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_bukgu.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['위반장소'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('북구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

df_bukgu_child =pd.read_csv('북구_어린이보호구역_위반건수.csv', encoding='cp949')

df_bukgu_child['연평균(2020-2024)'] = (df_bukgu_child['위반건수'] / 4).round(2)

df_bukgu_child.to_csv('북구_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')



# 2. 남구
gu_dfs['남구'].info()
df_namgu.info()
df_namgu

# --- 준비물 ---
zones = gu_dfs['남구'].copy()      # 남구 어린이보호구역 (대상시설명/주소/위경도 포함)
viol  = df_namgu.copy()           # 남구 위반 데이터: ['위반일자','위반시간','위반장소명']

# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['위반장소명'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"남구 전체 위반 건수: {len(viol):,}")
print(f"남구 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('남구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

######################
zones = gu_dfs['남구'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_namgu.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['위반장소명'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('남구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

df_namgu_child =pd.read_csv('남구_어린이보호구역_위반건수.csv', encoding='cp949')

df_namgu_child['연평균(2020-2024)'] = (df_namgu_child['위반건수'] / 5).round(2)

df_namgu_child.to_csv('남구_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')




# 3. 동구... 값이 다 안나옴 먼가 이상함
gu_dfs['동구'].info()

gu_dfs['동구'].to_csv('동구_어린이보호구역.csv', index=False, encoding='cp949')
df_donggu.info()
df_donggu


# --- 준비물 ---
zones= gu_dfs['동구'].copy()     
viol= df_donggu.copy()          

# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['위반장소명'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"동구 전체 위반 건수: {len(viol):,}")
print(f"동구 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('남구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')
#############################
import re
import pandas as pd

# ── 입력 DF (네가 쓰던 변수 그대로) ──────────────────────────────
zones = gu_dfs['동구'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_donggu.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['위반장소명'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('동구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')


df_donggu_child =pd.read_csv('동구_어린이보호구역_위반건수.csv', encoding='cp949')

df_donggu_child['연평균(2020-2024)'] = (df_donggu_child['위반건수'] / 4).round(2)

df_donggu_child.to_csv('동구_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')



# 4. 서구

gu_dfs['서구'].info()
df_seogu.info()
df_seogu

# --- 준비물 ---
zones = gu_dfs['서구'].copy()      # 남구 어린이보호구역 (대상시설명/주소/위경도 포함)
viol  = df_seogu.copy()           # 남구 위반 데이터: ['위반일자','위반시간','위반장소명']

# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['위반장소명'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"서구 전체 위반 건수: {len(viol):,}")
print(f"서구 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('서구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

###########################
zones = gu_dfs['서구'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_seogu.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['위반장소명'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('서구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

df_seogu_child =pd.read_csv('서구_어린이보호구역_위반건수.csv', encoding='cp949')

df_seogu_child['연평균(2020-2024)'] = (df_seogu_child['위반건수'] / 5).round(2)

df_seogu_child.to_csv('서구_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')



# 5. 중구
gu_dfs['중구'].info()
df_junggu.info()
df_junggu

# --- 준비물 ---
zones= gu_dfs['중구'].copy()     
viol= df_junggu.copy()          

# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['위반장소명'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"중구 전체 위반 건수: {len(viol):,}")
print(f"중구 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('중구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')
###################

zones = gu_dfs['중구'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_junggu.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['위반장소명'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('중구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

df_junggu_child =pd.read_csv('중구_어린이보호구역_위반건수.csv', encoding='cp949')

df_junggu_child['연평균(2020-2024)'] = (df_junggu_child['위반건수'] / 4).round(2)

df_junggu_child.to_csv('중구_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')




# 6. 수성구

gu_dfs['수성구'].info()
df_suseonggu.info()
df_suseonggu

# --- 준비물 ---
zones = gu_dfs['수성구'].copy()      # 남구 어린이보호구역 (대상시설명/주소/위경도 포함)
viol  = df_suseonggu.copy()           # 남구 위반 데이터: ['위반일자','위반시간','단속장소']

# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['단속장소'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"수성구 전체 위반 건수: {len(viol):,}")
print(f"수성구 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('수성구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

########################

zones = gu_dfs['수성구'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_suseonggu.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['단속장소'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('수성구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

df_suseonggu_child =pd.read_csv('수성구_어린이보호구역_위반건수.csv', encoding='cp949')

df_suseonggu_child['연평균(2020-2024)'] = (df_suseonggu_child['위반건수'] / 5).round(2)

df_suseonggu_child.to_csv('수성구_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')




# 7. 달서구

gu_dfs['달서구'].info()
df_dalseogu.info()
df_dalseogu

# --- 준비물 ---
zones = gu_dfs['달서구'].copy()      # 남구 어린이보호구역 (대상시설명/주소/위경도 포함)
viol  = df_dalseogu.copy()           # 남구 위반 데이터: ['위반일자','위반시간','단속장소']

# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['위반장소명'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"달서구 전체 위반 건수: {len(viol):,}")
print(f"달서구 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('달서구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

###################
zones = gu_dfs['달서구'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_dalseogu.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['위반장소명'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('달서구_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

df_dalseogu_child =pd.read_csv('달서구_어린이보호구역_위반건수.csv', encoding='cp949')

df_dalseogu_child['연평균(2020-2024)'] = (df_dalseogu_child['위반건수'] / 5).round(2)

df_dalseogu_child.to_csv('달서구_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')




# 8. 달성군

gu_dfs['달성군'].info()
df_dalseounggun.info()
df_dalseounggun

# --- 준비물 ---
zones = gu_dfs['달성군'].copy()      # 남구 어린이보호구역 (대상시설명/주소/위경도 포함)
viol  = df_dalseounggun.copy()           # 남구 위반 데이터: ['위반일자','위반시간','단속장소']

# 1) 텍스트 정규화 함수 (공백/괄호/특수문자 제거, 소문자)
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    # 괄호 안 보조정보 제거(예: '학교(본관)' -> '학교')
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    # 한글/영문/숫자만 남기고 나머지 제거
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

# 2) 보호구역 키워드 만들기
zones['키워드_시설'] = norm(zones['대상시설명'])

# 도로명주소에서 '...로/길...' 부분만 뽑아 키워드로 활용(있으면 추가)
def extract_road(addr: str) -> str:
    if not isinstance(addr, str): 
        return ''
    # 예: '대구 남구 명덕로 123' -> '명덕로'
    m = re.search(r'([가-힣0-9]+(?:로|길))', addr)
    return m.group(1) if m else ''

zones['키워드_도로'] = norm(zones['소재지도로명주소'].apply(extract_road))

# 시설명/도로명 둘 중 하나라도 매칭되면 보호구역으로 간주
keywords = pd.unique(
    pd.concat([zones['키워드_시설'], zones['키워드_도로']])
    .dropna()
    .replace('', pd.NA)
    .dropna()
)

# 3) 위반 데이터 정규화
viol['위반장소명_norm'] = norm(viol['단속장소'])

# 4) 전체 보호구역 매칭(키워드 OR 매칭) — 빠르게 하려면 정규식 하나로 합치기
# 키워드 길이 긴 것부터 매칭(부분 중복 방지에 유리)
keywords_sorted = sorted(keywords, key=len, reverse=True)
# 정규식 특수문자 이스케이프는 이미 제거했지만 혹시 몰라 한 번 더
pattern = '|'.join(map(re.escape, keywords_sorted))
mask_protected = viol['위반장소명_norm'].str.contains(pattern, na=False)

viol_in_zone = viol[mask_protected].copy()

print(f"달성군 전체 위반 건수: {len(viol):,}")
print(f"달성군 어린이보호구역 추정 위반 건수(키워드 매칭): {len(viol_in_zone):,}")

# 5) 시설별로 카운트(정확한 시설 귀속)
# 각 시설 키워드(시설명 중심)로 개별 매칭해서 count
facilities = zones[['대상시설명']].copy()
facilities['키워드_시설'] = zones['키워드_시설']

# 시설 키워드가 비어있지 않은 것만
facilities = facilities[facilities['키워드_시설'].notna() & (facilities['키워드_시설']!='')]

counts = []
for _, row in facilities.iterrows():
    key = row['키워드_시설']
    # 해당 시설 키워드가 포함된 위반만 카운트
    c = viol_in_zone['위반장소명_norm'].str.contains(re.escape(key), na=False).sum()
    counts.append((row['대상시설명'], int(c)))

df_counts = pd.DataFrame(counts, columns=['대상시설명', '위반건수']).sort_values('위반건수', ascending=False).reset_index(drop=True)

df_counts.to_csv('달성군_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

#########################
zones = gu_dfs['달성군'].copy()     # 어린이보호구역: '대상시설명' 포함
viol  = df_dalseounggun.copy()          # 위반데이터: '위반장소명' 포함

# ── 1) 정규화: 한글/영문/숫자만 남기고 소문자, 공백/괄호 제거 ──
def norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    s = s.str.replace(r'\([^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    s = s.str.replace(r'[^0-9a-z가-힣]', '', regex=True)
    return s

zones = zones.copy()
viol  = viol.copy()
zones['시설_norm'] = norm(zones['대상시설명'])
viol['장소_norm']  = norm(viol['단속장소'])

# ── 2) 시설명 변형(‘초교’→‘초등학교’, 병설 표현 통일)만 생성 ──
#     ※ 도로/동 정보는 과매칭 원인이므로 '사용하지 않음'
SUBS = [
    (r'초교병설', '초등학교병설유치원'),
    (r'초병설',   '초등학교병설유치원'),
    (r'초등병설', '초등학교병설유치원'),
    (r'초교',     '초등학교'),        # 입석초교 → 입석초등학교
    # (r'초',    '초등학교')  # 너무 광범위 → 사용하지 않음
]

TAILS = ['초등학교병설유치원', '초등학교', '유치원', '어린이집']  # 허용 접미

def facility_variants(name_norm: str) -> list:
    """시설명만으로 안전한 변형 생성 (도로/동 불사용)"""
    if not isinstance(name_norm, str) or not name_norm:
        return []

    # 공통 치환 적용
    n = name_norm
    for pat, out in SUBS:
        n = re.sub(pat, out, n)

    vars_ = {n}

    # 접미 제거한 베이스 추출 (예: 팔공초등학교 → 팔공)
    base = re.sub(r'(?:' + '|'.join(TAILS) + r')$', '', n)
    if base:
        # 대표 접미 조합(과매칭 방지 위해 ‘초’ 단독은 제외)
        for tail in TAILS:
            vars_.add(base + tail)

    # 최종 후보 정리 (2글자 이상, 중복 제거, 길이 긴 순)
    vars_ = [v for v in vars_ if len(v) >= 2]
    vars_.sort(key=len, reverse=True)
    return vars_

# 시설별 변형 목록 준비
fac_list = []
for fac in zones['대상시설명'].dropna().unique():
    v = facility_variants(norm(pd.Series([fac]))[0])
    if v:
        fac_list.append((fac, v))

# ── 3) 위반 1건당 “가장 긴 변형을 맞춘 시설” 1곳에만 배정 ──
assign = []
for idx, text in viol['장소_norm'].items():
    best_fac, best_len = None, 0
    if not isinstance(text, str) or not text:
        assign.append((idx, None))
        continue
    for fac, variants in fac_list:
        for tok in variants:
            if tok in text:
                L = len(tok)
                if L > best_len:           # 더 긴 변형을 맞춘 시설 우선
                    best_fac, best_len = fac, L
                break  # 이 시설은 더 짧은 변형 볼 필요 없음
    assign.append((idx, best_fac))

assign_df = pd.DataFrame(assign, columns=['vi_idx', '대상시설명'])
assign_df = assign_df.dropna(subset=['대상시설명'])

# ── 4) 시설별 건수 집계(중복 없음) ─────────────────────────────
df_counts = (assign_df
             .groupby('대상시설명')
             .size()
             .reset_index(name='위반건수')
             .sort_values('위반건수', ascending=False)
             .reset_index(drop=True))


# ── 5) “전체 시설 목록” 기준으로 0건도 포함해 보여주기 (여기 추가!) ──
facilities_all = zones[['대상시설명']].dropna().drop_duplicates()
df_counts_full = (facilities_all
                  .merge(df_counts, on='대상시설명', how='left')
                  .fillna({'위반건수': 0})
                  .astype({'위반건수': 'int64'})
                  .sort_values('위반건수', ascending=False)
                  .reset_index(drop=True))


# 필요시 저장
df_counts_full.to_csv('달성군_어린이보호구역_위반건수.csv', index=False, encoding='cp949')

df_dalseonggun_child =pd.read_csv('달성군_어린이보호구역_위반건수.csv', encoding='cp949')

df_dalseonggun_child['연평균(2020-2024)'] = (df_dalseonggun_child['위반건수'] / 5).round(2)

df_dalseonggun_child.to_csv('달성군_어린이보호구역_위반건수_연평균포함.csv', index=False, encoding='cp949')
