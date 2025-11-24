import joblib, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline

df = pd.read_excel("data/점수포함.xlsx")
df['사고유무'] = (df['사고건수'] > 0).astype(int)  # 사고발생=1




# ==== 전역 변수 영향도: SHAP summary (beeswarm + bar) ====
import joblib, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
from pathlib import Path

PKL_PATH   = "xgb_model.pkl"   # {'model': XGBClassifier(...), 'features': [...]} 형태 pkl
TARGET_COL = "사고유무"           # 있으면 자동 drop
MAX_DISPLAY = 20                 # 상위 n개 변수만 표시

# 1) 모델/피처
obj = joblib.load(PKL_PATH)
model     = obj["model"]
feat_list = obj.get("features")

# 2) X, y 구성 (df는 이미 존재)
df_use = df.copy()
y = df_use[TARGET_COL] if TARGET_COL in df_use.columns else None
if TARGET_COL in df_use.columns:
    df_use = df_use.drop(columns=[TARGET_COL])

if feat_list is not None:
    missing = [f for f in feat_list if f not in df_use.columns]
    if missing:
        raise ValueError(f"df에 없는 피처: {missing}")
    X = df_use[feat_list]
else:
    X = df_use

# 3) SHAP 계산 (가능하면 '확률' 단위로)
try:
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional",
        model_output="probability"
    )
    shap_vals = explainer.shap_values(X)
    base = explainer.expected_value
except Exception:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    base = explainer.expected_value

# XGBoost 이진분류에서는 list로 반환될 수 있으니 class=1 선택
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
    if isinstance(base, list):
        base = base[1]

# 4) 전역 요약 플롯 저장
out_dir = Path("shap_global"); out_dir.mkdir(exist_ok=True)

# 4-1) Beeswarm (분포 + 영향 방향)
plt.figure()
shap.summary_plot(shap_vals, X, show=False, max_display=MAX_DISPLAY)
plt.tight_layout()
plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=180, bbox_inches="tight")
plt.close()

# 4-2) Bar (평균 |SHAP| 중요도)
plt.figure()
shap.summary_plot(shap_vals, X, plot_type="bar", show=False, max_display=MAX_DISPLAY)
plt.tight_layout()
plt.savefig(out_dir / "shap_summary_bar.png", dpi=180, bbox_inches="tight")
plt.close()

# 5) 표 형태(평균 |SHAP|)도 저장
imp = (pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
         .sort_values(ascending=False)
         .rename("mean_abs_shap"))
imp.to_csv(out_dir / "shap_importance.csv", encoding="utf-8-sig")
print("상위 10개 변수(평균 |SHAP|):")
print(imp.head(10))

# (선택) Permutation Importance로 교차 확인 (y가 있으면)
try:
    if y is not None:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, scoring="roc_auc",
                                   n_repeats=10, random_state=42, n_jobs=-1)
        perm = (pd.Series(r.importances_mean, index=X.columns)
                  .sort_values(ascending=False)
                  .rename("perm_importance_auc"))
        perm.to_csv(out_dir / "permutation_importance.csv", encoding="utf-8-sig")
except Exception as e:
    print("Permutation importance는 건너뜀:", e)

print("저장 완료 →", out_dir.resolve())

# ===================================================================
PKL_PATH = "xgb_model.pkl"   # 딕셔너리(pkl)
ID_COL     = "시설명"             # 없으면 index 사용
TARGET_COL = "사고유무"           # 있으면 drop
TOPK_CASES = 3                   # 위험 확률 상위 N개 보호구역
TOPK_FEATS = 10                  # 막대그래프에 표시할 중요 변수 수

# ==== 1) 모델/피처 ====
obj = joblib.load(PKL_PATH)
model     = obj["model"]
feat_list = obj.get("features")

# ==== 2) X, id ====
df_use = df.drop(columns=[c for c in [TARGET_COL] if c in df.columns])
if feat_list is not None:
    missing = [f for f in feat_list if f not in df_use.columns]
    if missing:
        raise ValueError(f"df에 없는 피처: {missing}")
    X = df_use[feat_list].copy()
else:
    X = df_use.copy()

ids = df[ID_COL] if ID_COL in df.columns else pd.Series(df.index, name="id")

# ==== 3) 예측확률 상위 TOPK 보호구역 ====
proba   = model.predict_proba(X)[:, 1]
top_idx = np.argsort(-proba)[:TOPK_CASES]

# ==== 4) SHAP 값 계산 (가능하면 '확률' 단위로) ====
use_prob = True
try:
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional",
                                   model_output="probability")
    shap_vals = explainer.shap_values(X)     # 확률 단위 shap
    exp_val   = explainer.expected_value     # baseline 확률
except Exception:
    use_prob = False
    explainer = shap.TreeExplainer(model)    # 로그오즈 단위 shap
    shap_vals = explainer.shap_values(X)
    exp_val   = explainer.expected_value

# XGBoost 이진분류에서 shap_vals가 list로 나올 수 있으니 양성(1) 기준 선택
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
    if isinstance(exp_val, list): exp_val = exp_val[1]

# ==== 5) 막대그래프 저장 ====
out_dir = Path("shap_bar_top3"); out_dir.mkdir(exist_ok=True)

for rank, i in enumerate(top_idx, start=1):
    sv  = pd.Series(shap_vals[i, :], index=X.columns)
    val = X.iloc[i]

    # |SHAP| 상위 TOPK_FEATS 정렬
    top = (pd.DataFrame({"shap": sv, "value": val})
             .reindex(sv.abs().sort_values(ascending=False).index)
             .head(TOPK_FEATS))

    # 막대그래프 (양수=위험 확률 ↑, 음수=위험 확률 ↓)
    plt.figure(figsize=(8, 5))
    colors = np.where(top["shap"] >= 0, "crimson", "steelblue")
    plt.barh(top.index, top["shap"], color=colors)
    plt.axvline(0, lw=1, color="black")
    plt.gca().invert_yaxis()

    # 값 주석(선택)
    for y, (s, v) in enumerate(zip(top["shap"], top["value"])):
        ha = "left" if s >= 0 else "right"
        dx = 0.005 * (top["shap"].abs().max() or 1)
        plt.text(s + (dx if s >= 0 else -dx), y, f"{v:.3g}", va="center", ha=ha, fontsize=9)

    unit = "prob" if use_prob else "log-odds"
    title = f"[Top{rank}] {ids.iloc[i]}  P={proba[i]:.3f}  (baseline={exp_val:.3f}, {unit})"
    plt.title(title, fontsize=11)
    plt.xlabel("SHAP value " + ("(probability)" if use_prob else "(log-odds)"))
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_top{rank}_{i}.png", dpi=160)
    plt.close()

print("저장 경로:", out_dir.resolve())



# ==============================================

df = pd.read_excel("data/점수포함.xlsx")
df['사고유무'] = (df['사고건수'] > 0).astype(int)  # 사고발생=1

# 군/구 추출 → 새 컬럼 '구군'
df["구군"] = (
    df["주소"]
      .str.replace(r"\s+", " ", regex=True).str.strip()
      .str.extract(r"(?:대구광역시|대구시)\s*(?P<gu>[가-힣]+(?:구|군))")["gu"]
)

# (선택) 표준 카테고리로 정리
order = ["중구","동구","서구","남구","북구","수성구","달서구","달성군"]
df["구군"] = pd.Categorical(df["구군"], categories=order)

df['구군']

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1) 구군별 사고발생률 집계
tmp = (df.dropna(subset=['구군'])
         .assign(사고유무=df['사고유무'].astype(int))
         .groupby('구군')['사고유무']
         .agg(사고건수='sum', 표본수='size'))
tmp['사고발생률'] = tmp['사고건수'] / tmp['표본수']

# 정렬(발생률 내림차순)
tmp = tmp.sort_values('사고발생률', ascending=False)

# 2) 막대그래프
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(tmp.index, tmp['사고발생률'])

ax.set_title('구군별 사고발생률')
ax.set_ylabel('사고발생률')
ax.set_ylim(0, max(0.05, tmp['사고발생률'].max()*1.15))  # 머리공간

# 3) 퍼센트/표본수 주석
for i, (rate, n) in enumerate(zip(tmp['사고발생률'], tmp['표본수'])):
    ax.text(i, rate, f"{rate*100:.1f}% (n={n})",
            ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=0)
plt.tight_layout()

df.hist()
plt.show()


import seaborn as sns

df = pd.read_excel("data/점수포함.xlsx")
df = df.drop(columns=['구역지정수','pred_proba','pred_label','risk_score','속도_도로폭비','주정차위반_교통량','주정차위반_도로폭','인구밀도','교통밀도','어린이밀도','교통량_도로폭비'])


# 한글 폰트 (필요 시)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1) 수치형만 추출(사고유무는 제외)
num_df = df.drop(columns=['사고유무'], errors='ignore').select_dtypes(include='number')

# 2) 상관계수 (Pearson; 순위 상관이 필요하면 method='spearman')
corr = num_df.corr(method='pearson')

# 3) 히트맵 (모든 칸 표시)
plt.figure(figsize=(11, 9))
sns.heatmap(
    corr, annot=True, fmt=".2f",
    cmap="coolwarm", vmin=-1, vmax=1,
    linewidths=0.5, cbar_kws={"shrink": .8}
)
plt.title("수치형 변수 상관계수 (Pearson)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

THR = 0.6  # 임계값

# 1) 수치형만 (사고유무 제외)
num_df = df.drop(columns=['사고유무'], errors='ignore').select_dtypes(include='number')

# 2) 상관행렬(부호 있는 것)과 절대값
corr_signed = num_df.corr(method='pearson')
corr_abs = corr_signed.abs()

# 3) 위쪽 삼각형만 남기고 스택 → 절대값 기준 필터
upper_mask = np.triu(np.ones_like(corr_abs, dtype=bool), k=1)
hits = (
    corr_signed.where(upper_mask)  # 부호 보존
              .stack()             # (var1, var2, corr)
              .rename("corr")
              .reset_index()
)

# 4) 절대값 임계 이상만 추출
high_corr = (hits.loc[hits["corr"].abs() >= THR]
                  .assign(abs_corr=lambda d: d["corr"].abs())
                  .sort_values("abs_corr", ascending=False)
                  .reset_index(drop=True))

# 결과 미리보기
print(high_corr.head(20))

# 달서구만 분석
import joblib, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
from pathlib import Path

# ===== 설정 =====
PKL_PATH   = "xgb_model.pkl"   # dict 형태 pkl 경로
TARGET_COL = "사고유무"           # 있으면 제거
GU_VALUE   = "달서구"             # 대상 구군
MAX_DISPLAY = 20                 # 플롯에 표시할 변수 개수

# ===== 1) 모델/피처 로드 =====
obj = joblib.load(PKL_PATH)
model     = obj["model"]
feat_list = obj.get("features")

# ===== 2) 달서구 부분집합 X 구성 =====
df_sub = df.loc[df["구군"] == GU_VALUE].copy()
if df_sub.empty:
    raise ValueError(f"'{GU_VALUE}'에 해당하는 행이 없습니다.")

if TARGET_COL in df_sub.columns:
    df_sub.drop(columns=[TARGET_COL], inplace=True)

if feat_list is not None:
    missing = [f for f in feat_list if f not in df_sub.columns]
    if missing:
        raise ValueError(f"df에 없는 피처: {missing}")
    X = df_sub[feat_list]
else:
    X = df_sub.select_dtypes(include="number")  # 피처 목록이 없다면 수치형만

# ===== 3) SHAP 계산 (가능하면 '확률' 단위) =====
try:
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional",
        model_output="probability"
    )
    shap_vals = explainer.shap_values(X)
    base = explainer.expected_value
except Exception:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    base = explainer.expected_value

# XGBoost 이진분류는 list 반환 가능 → 양성(1) 선택
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
    if isinstance(base, list): base = base[1]

# ===== 4) 전역 영향도 플롯 저장 =====
out_dir = Path("shap_global_dalseo"); out_dir.mkdir(exist_ok=True)

# Beeswarm (분포 + 방향)
plt.figure()
shap.summary_plot(shap_vals, X, show=False, max_display=MAX_DISPLAY)
plt.tight_layout()
plt.savefig(out_dir / "shap_summary_beeswarm_달서구.png", dpi=180, bbox_inches="tight")
plt.close()

# Bar (평균 |SHAP| 중요도)
plt.figure()
shap.summary_plot(shap_vals, X, plot_type="bar", show=False, max_display=MAX_DISPLAY)
plt.tight_layout()
plt.savefig(out_dir / "shap_summary_bar_달서구.png", dpi=180, bbox_inches="tight")
plt.close()

# (선택) 표로도 저장
imp = (pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
         .sort_values(ascending=False)
         .rename("mean_abs_shap"))
imp.to_csv(out_dir / "shap_importance_달서구.csv", encoding="utf-8-sig")

print(f"저장 완료 → {out_dir.resolve()}")

# 위험도 점수 낮은 top3
import joblib, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
from pathlib import Path

# ===== 설정 =====
PKL_PATH    = "xgb_model.pkl"   # dict 형태 pkl
ID_COL      = "시설명"            # 식별자 없으면 index 사용
TARGET_COL  = "사고유무"          # 있으면 drop
N_LOW       = 3                   # 위험도(예측확률) 낮은 사례 상위 N
TOPK_FEATS  = 10                  # 막대에 표시할 변수 수

# ===== 1) 모델/피처 로드 =====
obj = joblib.load(PKL_PATH)
model     = obj["model"]
feat_list = obj.get("features")

# ===== 2) X 구성 =====
df_use = df.drop(columns=[c for c in [TARGET_COL] if c in df.columns]).copy()
if feat_list is not None:
    missing = [f for f in feat_list if f not in df_use.columns]
    if missing:
        raise ValueError(f"df에 없는 피처: {missing}")
    X = df_use[feat_list]
else:
    X = df_use.select_dtypes(include="number")

ids = df[ID_COL] if ID_COL in df.columns else pd.Series(df.index, name="id")

# ===== 3) 예측확률 & 낮은 N 선택 =====
proba   = model.predict_proba(X)[:, 1]
low_idx = np.argsort(proba)[:N_LOW]

print("Low-risk Top-N:")
for r, i in enumerate(low_idx, 1):
    print(f"{r}) {ids.iloc[i]}  P={proba[i]:.4f}")

# ===== 4) SHAP 계산(가능하면 확률 단위) =====
use_prob = True
try:
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional",
                                   model_output="probability")
    shap_vals = explainer.shap_values(X)
    exp_val   = explainer.expected_value
except Exception:
    use_prob = False
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    exp_val   = explainer.expected_value

# 이진분류에서 list 반환 시 class=1 선택
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
    if isinstance(exp_val, list): exp_val = exp_val[1]

# ===== 5) 개별 SHAP 막대그래프 저장 =====
out_dir = Path("shap_bar_lowN"); out_dir.mkdir(exist_ok=True)

for rank, i in enumerate(low_idx, start=1):
    sv  = pd.Series(shap_vals[i, :], index=X.columns)
    val = X.iloc[i]

    # |SHAP| 상위 TOPK_FEATS
    top = (pd.DataFrame({"shap": sv, "value": val})
             .reindex(sv.abs().sort_values(ascending=False).index)
             .head(TOPK_FEATS))

    plt.figure(figsize=(8, 5))
    # 양수(위험↑) vs 음수(위험↓) 색상
    colors = np.where(top["shap"] >= 0, "crimson", "steelblue")
    plt.barh(top.index, top["shap"], color=colors)
    plt.axvline(0, lw=1, color="black")
    plt.gca().invert_yaxis()

    # 값 주석(선택)
    for y, (s, v) in enumerate(zip(top["shap"], top["value"])):
        ha = "left" if s >= 0 else "right"
        dx = 0.005 * (top["shap"].abs().max() or 1)
        plt.text(s + (dx if s >= 0 else -dx), y, f"{v:.3g}", va="center", ha=ha, fontsize=9)

    unit = "probability" if use_prob else "log-odds"
    title = f"[Low{rank}] {ids.iloc[i]}  P={proba[i]:.3f}  (baseline={exp_val:.3f}, {unit})"
    plt.title(title, fontsize=11)
    plt.xlabel("SHAP value " + ("(probability)" if use_prob else "(log-odds)"))
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_low{rank}_{i}.png", dpi=160)
    plt.close()

print("저장 경로:", out_dir.resolve())

# ============== PPT ============================

df = pd.read_excel("data/점수포함.xlsx")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap, joblib
from matplotlib import font_manager as fm
from pathlib import Path

# ---------- 폰트: 맑은 고딕 우선 ----------
def set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    avail = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in avail:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ---------- 입력 ----------
PKL_PATH    = "xgb_model.pkl"     # {'model': XGBClassifier(...), 'features': [...]} 형태
TARGET_COL  = "사고유무"
ID_COL      = "시설명"
EXCL_LATLON = {"위도","경도"}
TOPK        = 5

# 모델 & 데이터 준비
obj = joblib.load(PKL_PATH)
model     = obj["model"]
feat_list = obj.get("features")

df_use = df.drop(columns=[c for c in [TARGET_COL] if c in df.columns]).copy()
X = df_use[feat_list] if feat_list is not None else df_use.select_dtypes(include="number")
ids = df[ID_COL] if ID_COL in df.columns else pd.Series(df.index, name="id")
proba = model.predict_proba(X)[:,1]

# 두 가지 SHAP 경로 준비: ①확률 단위 → 실패 시 ②로그오즈 기반 '정확 ΔP' 계산
def get_explanations(X):
    # ① 확률 단위 시도
    try:
        exp = shap.Explainer(model, X, algorithm="tree",
                             feature_perturbation="interventional",
                             model_output="probability")(X)
        return {"mode":"prob", "values":exp.values, "base":exp.base_values}
    except Exception:
        # ② 로그오즈 단위
        te  = shap.TreeExplainer(model)
        sv  = te.shap_values(X)             # (n,f)
        base= te.expected_value             # scalar or list
        if isinstance(sv, list): sv = sv[1]
        if isinstance(base, list): base = base[1]
        return {"mode":"logodds", "values":sv, "base":base}

expl = get_explanations(X)

def sigmoid(z): return 1.0/(1.0+np.exp(-z))

def plot_shap_idx(idx, include_latlon, title, xlabel, outfile, topk=TOPK):
    s_all = pd.Series(expl["values"][idx,:], index=X.columns)
    if not include_latlon:
        s_all = s_all.drop(index=[c for c in EXCL_LATLON if c in s_all.index])

    top_idx = s_all.abs().sort_values(ascending=False).index[:topk]
    s = s_all[top_idx].copy()
    vals = X.iloc[idx][top_idx].copy()

    # ΔP 계산
    if expl["mode"] == "prob":
        delta_p = s.values                                  # 확률 단위 SHAP = 정확 ΔP 분해
        base_p  = np.array(expl["base"])[()]
        xlab_default = "SHAP 값 (확률 기준)"
    else:
        # 로그오즈에서 '정확한 국소 ΔP' 계산: p(z) - p(z - φ_i)
        phi = s_all.values
        z   = expl["base"] + s_all.sum()
        p   = sigmoid(z)
        delta_p_full = p - sigmoid(z - s_all.values)        # 모든 특성에 대한 ΔP
        delta_p = delta_p_full[[list(s_all.index).index(c) for c in top_idx]]
        base_p  = sigmoid(expl["base"])
        xlab_default = "SHAP 값 (로그오즈 기준)"

    # 그리기
    plt.figure(figsize=(11,5))
    colors = np.where(s.values >= 0, "crimson", "steelblue")
    plt.barh(s.index, s.values, color=colors); plt.axvline(0, lw=1, color="black")
    plt.gca().invert_yaxis()

    # 값 레이블: 원값 + (ΔP=±xx.x%)
    for y, (sv, v, dp) in enumerate(zip(s.values, vals.values, delta_p)):
        ha = "left" if sv >= 0 else "right"
        dx = 0.006*(np.abs(s.values).max() or 1)
        plt.text(sv + (dx if sv>=0 else -dx), y, f"{v:.3g} (ΔP={dp*100:+.1f}%)",
                 va="center", ha=ha, fontsize=10)

    p_hat = proba[idx]
    tag = "위·경도 포함" if include_latlon else "위·경도 제외"
    plt.title(title or f"[사례 {idx}] {tag} — 예측확률 P={p_hat:.3f} (기준={base_p:.3f})", fontsize=13)
    plt.xlabel(xlabel or xlab_default)
    plt.tight_layout(); plt.savefig(outfile, dpi=170); plt.close()

# ============ 사용 예 ============
# 1) 상위 2개 인덱스 뽑기
top2 = np.argsort(-proba)[:2]
print("Top2:", top2, " / probs:", proba[top2])

# 2) 원하는 인덱스/제목/축으로 저장
i = int(top2[1])  # 직접 숫자 지정 가능
plot_shap_idx(i, True,
              title="[대광어린이집] 변수 영향도 Top5",
              xlabel="SHAP 값 (확률/로그오즈 기준)",
              outfile=f"shap_bar_idx{i}_incl_latlon.png")
plot_shap_idx(i, False,
              title="[대광어린이집] 변수 영향도 Top5",
              xlabel="SHAP 값 (확률/로그오즈 기준)",
              outfile=f"shap_bar_idx{i}_excl_latlon.png")


# ========== PDP ============

import numpy as np, matplotlib.pyplot as plt, joblib
from sklearn.inspection import PartialDependenceDisplay
from matplotlib import font_manager as fm

# 한글 폰트
def set_korean_font():
    for name in ["Malgun Gothic","NanumGothic","AppleGothic","Noto Sans CJK KR","DejaVu Sans"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = name; break
    plt.rcParams["axes.unicode_minus"] = False
set_korean_font()

def plot_pdp_single(
    df, pkl_path="xgb_model.pkl",
    feature="주행속도", xlim=None,
    title=None, ylabel="부분 의존(예측확률)",
    savepath=None, add_rug=True, target_class=1
):
    # 모델/피처
    obj = joblib.load(pkl_path)
    model     = obj["model"]
    feat_list = obj.get("features")

    X = df.drop(columns=[c for c in ["사고유무"] if c in df.columns]).copy()
    X = X[feat_list] if feat_list is not None else X.select_dtypes(include="number")
    if feature not in X.columns:
        raise ValueError(f"'{feature}' 컬럼이 없습니다.")
    fidx = list(X.columns).index(feature)

    # PDP 그림
    fig, ax0 = plt.subplots(figsize=(9, 6))
    disp = PartialDependenceDisplay.from_estimator(
        estimator=model, X=X, features=[fidx], feature_names=X.columns.tolist(),
        kind="average", grid_resolution=120,
        response_method="predict_proba", target=target_class, ax=ax0,
    )

    # ==> 실제 축 객체 다시 얻기(버전 호환)
    ax = getattr(disp, "axes_", None)
    ax = ax[0,0] if ax is not None else getattr(disp, "ax_", ax0)

    # PDP 선 데이터가 xlim 밖으로 나가면 클리핑
    if xlim is not None and hasattr(disp, "lines_") and disp.lines_:
        for ln in disp.lines_[0]:   # 이 피처의 모든 라인(보통 1개)
            x, y = ln.get_data()
            m = (x >= xlim[0]) & (x <= xlim[1])
            if m.any():
                ln.set_data(x[m], y[m])

    # 러그(표본 분포)
    if add_rug:
        vals = np.asarray(X[feature].dropna())
        if xlim is not None:
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        ymin, ymax = ax.get_ylim()
        ax.vlines(vals, ymin=ymin-(ymax-ymin)*0.015, ymax=ymin, lw=0.6, color="black")

    # 제목/라벨
    ax.set_title(title or f"[PDP] {feature}와 예측사고확률")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(feature)

    # ★ 마지막에 xlim 강제 고정(autoscale 끄기)
    if xlim is not None:
        ax.set_autoscale_on(False)
        ax.set_xlim(*xlim)
        ax.set_xbound(*xlim)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight"); plt.close()
        print(f"저장 완료: {savepath}")
    else:
        plt.show()

plot_pdp_single(
    df, feature="교통량_도로폭비",
    xlim=(0, 6000),
    title="[PDP] 교통량_도로폭비 예측사고확률",
    ylabel="부분 의존(예측확률)",
    savepath="pdp_교통량_도로폭비.png",
    target_class=1
)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

df['사고유무'] = (df['사고건수'] > 0).astype(int)  # 사고발생=1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 한글 폰트
def set_korean_font():
    for name in ["Malgun Gothic","NanumGothic","AppleGothic","Noto Sans CJK KR","DejaVu Sans"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = name; break
    plt.rcParams["axes.unicode_minus"] = False
set_korean_font()

def boxplot_nonacc(
    df, var, value=None, xlim=None,
    title=None, xlabel=None, savepath=None,
    jitter=True, whis=1.5  # ← 기본 1.5 IQR, 또는 [5,95] 같이 전달
):
    s = df.loc[df["사고유무"]==0, var].dropna().astype(float)
    if s.empty:
        raise ValueError("사고유무=0 데이터가 없거나 변수 값이 없습니다.")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.boxplot(
        s.values, vert=False, whis=whis, showfliers=True, widths=0.5,
        patch_artist=False
    )

    if jitter:
        y = np.random.normal(loc=1, scale=0.02, size=len(s))
        ax.plot(s.values, y, "o", alpha=0.18, markersize=3)

    if value is not None:
        ax.axvline(value, linestyle="--", linewidth=1.5)
        ax.text(value, 1.12, f"현재값: {value}", ha="center", va="bottom")

    ax.set_yticks([1]); ax.set_yticklabels([f"사고 미발생(N={len(s)})"])
    ax.set_xlabel(xlabel or var)
    ax.set_title(title or f"[Box Plot] 사고 미발생 구역 — {var}")
    if xlim is not None:
        ax.set_xlim(*xlim)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180, bbox_inches="tight"); plt.close()
    else:
        plt.show()
df.columns
# 1) 표준 IQR(1.5) 수염
boxplot_nonacc(df, var="보호구역도로폭", value=12, xlim=(0,30),
               title="[Box Plot] 사고 미발생 — 도로폭", xlabel="도로폭(m)", whis=[5,95])

# 2) 백분위 기반 수염(5%~95%)
boxplot_nonacc(df, var="교통밀도", value = 7356.88073394495 ,xlim=(0,12000),
               title="[Box Plot] 사고 미발생 — 교통밀도", xlabel="건수", whis=[5,95])

