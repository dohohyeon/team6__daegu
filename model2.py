# pip install mlxtend xgboost

# =========================
# 0) 준비/임포트
# =========================
import os, re
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    classification_report, balanced_accuracy_score, make_scorer, confusion_matrix
)
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, balanced_accuracy_score
)
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    f1_score, recall_score, accuracy_score, precision_score,
    roc_auc_score, classification_report, balanced_accuracy_score, make_scorer
)
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    classification_report, balanced_accuracy_score, make_scorer, confusion_matrix
)
from xgboost import XGBClassifier


# =========================
# 1) 데이터 로드 & 전처리
# =========================
df = pd.read_excel("data/dataset.xlsx", index_col="시설명")
dong_df = pd.read_excel("data/읍면동별데이터1.xlsx")

# 읍면동(세 번째 토큰) 추출 후 병합
dong_df["읍면동"] = dong_df["읍면동"].astype(str).str.extract(r'^(?:\S+\s+){2}(\S+)')
df = df.merge(dong_df, on="읍면동", how="left")



# 0 나눗셈 방지용
def nz(x, eps=1e-6):
    return np.clip(x, eps, None)

# 파생변수(분모 보호)
df["교통량_도로폭비"]   = df["교통량"] / nz(df["보호구역도로폭"])
df["어린이밀도"]       = df["어린이인구"] / nz(df["면적"])
df["인구밀도"]         = df["전체인구"]   / nz(df["면적"])
df["교통밀도"]         = df["교통량"]     / nz(df["면적"])
df["속도_도로폭비"]     = df["주행속도"] / nz(df["보호구역도로폭"])
df["주정차위반_교통량"] = df["불법주정차위반"] / nz(df["교통량"])
df["주정차위반_도로폭"] = df["불법주정차위반"] / nz(df["보호구역도로폭"])  # 보정

# 불필요 컬럼 제거(있으면만)
df = df.drop(columns=["주소","읍면동","어린이인구","전체인구","면적","구역지정수",'위도','경도'], errors="ignore")

# 수치형 변환 + inf -> NaN
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# 2) 분할
# =========================
target_col = "사고건수"
X = df.drop(columns=[target_col])
y = (df[target_col] > 0).astype(int)  # 사고발생=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3) 고정변수 & 탐색변수 나누기
# =========================
fixed_features = [
    "시설물 CCTV 수","시설물 도로표지판 수","시설물 과속방지턱 수",
    "보호구역도로폭","신호등_반경300m"
]

# 탐색 후보 = 전체 - (고정 + 타깃)
candidates = [c for c in X_train.columns if c not in fixed_features]

print(f"[INFO] 고정변수({len(fixed_features)}): {fixed_features}")
print(f"[INFO] 탐색변수 후보({len(candidates)}): {candidates}")

if len(candidates) == 0:
    raise ValueError("EFS에 넣을 유효 탐색변수가 없습니다. fixed_features 구성을 확인하세요.")

# =========================
# 4) 고정 파라미터의 XGBoost 분류기
# =========================
# 클래스 불균형 보정
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / max(pos, 1)

xgb_fixed = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.0,
    reg_lambda=5.0,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist"   # CPU 가속
)

# =========================
# 5) EFS(탐색변수에 대해서만 전수탐색)
#    - 평가 스코어는 f1_macro(양·음성 균형 반영)
#    - 최종 학습/평가 때는 [고정 + 선택] 사용
# =========================
min_class = int(y_train.value_counts().min())
n_outer = min(5, max(2, min_class))
cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

efs = EFS(
    estimator=xgb_fixed,
    min_features=1,
    max_features=len(candidates),     # 탐색변수 전체 범위
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    print_progress=True
)

# 전수탐색은 '탐색변수'만 넘김
efs = efs.fit(X_train[candidates], y_train)

# 최적 탐색변수 + 고정변수 합치기
best_candidate_names = list(efs.best_feature_names_)
best_features = fixed_features + best_candidate_names

print("\n[선택된 탐색변수]")
print(best_candidate_names)
print("[최종 사용 변수(고정+탐색)]")
print(best_features)

# =========================
# 6) 최종 학습/평가 ([고정 + 선택]만 사용)
# =========================
best_model = xgb_fixed.fit(X_train[best_features], y_train)

y_prob = best_model.predict_proba(X_test[best_features])[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n[테스트셋 성능]")
print("Accuracy        :", accuracy_score(y_test, y_pred))
print("Precision       :", precision_score(y_test, y_pred, zero_division=0))
print("Recall (positive):", recall_score(y_test, y_pred, zero_division=0))
print("F1-score        :", f1_score(y_test, y_pred, zero_division=0))
print("F1-macro        :", f1_score(y_test, y_pred, average='macro', zero_division=0))
print("Balanced Acc    :", balanced_accuracy_score(y_test, y_pred))
print("ROC-AUC         :", roc_auc_score(y_test, y_prob))

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=3))

# ================================================
# 변수 선정 후 파라미터 튜닝

# =========================
# 1) 데이터 로드 & 전처리
# =========================
df = pd.read_excel("data/dataset.xlsx", index_col="시설명")
dong_df = pd.read_excel("data/읍면동별데이터1.xlsx")
dong_df["읍면동"] = dong_df["읍면동"].astype(str).str.extract(r'^(?:\S+\s+){2}(\S+)')
df = df.merge(dong_df, on="읍면동", how="left")

def nz(x, eps=1e-6): 
    return np.clip(x, eps, None)

# 파생 (분모 보호)
df["교통량_도로폭비"]   = df["교통량"] / nz(df["보호구역도로폭"])
df["어린이밀도"]       = df["어린이인구"] / nz(df["면적"])
df["인구밀도"]         = df["전체인구"]   / nz(df["면적"])
df["교통밀도"]         = df["교통량"]     / nz(df["면적"])
df["속도_도로폭비"]     = df["주행속도"] / nz(df["보호구역도로폭"])
df["주정차위반_교통량"] = df["불법주정차위반"] / nz(df["교통량"])
df["주정차위반_도로폭"] = df["불법주정차위반"] / nz(df["보호구역도로폭"])

# 불필요 컬럼 제거(있을 때만)
df = df.drop(columns=["주소","읍면동","어린이인구","전체인구","면적","구역지정수",'위도','경도'], errors="ignore")

# 수치형 변환 + inf 처리
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# 2) 분할
# =========================
target_col = "사고건수"
X_full = df.drop(columns=[target_col])
y_full = (df[target_col] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# =========================
# 3) 사용할 변수(직접 선택) — 비워두면 전체 사용
# =========================
manual_features = ['시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수',
                   '보호구역도로폭', '신호등_반경300m',
                    '주행속도','불법주정차위반','교통량_도로폭비','교통밀도','경사도',
                    ]

if len(manual_features) == 0:
    manual_features = list(X_train.columns)  # ← 비워두면 전체 피처 사용
else:
    missing = [c for c in manual_features if c not in X_train.columns]
    assert not missing, f"다음 컬럼이 데이터에 없습니다: {missing}"

Xtr = X_train[manual_features].copy()
Xte = X_test[manual_features].copy()

# =========================
# 4) 스코어러(전체 macro 기준):
#    F1_macro ≥ 0.5 & Precision_macro ≥ 0.5 만족하는 threshold 중 Recall_macro 최대화
# =========================
def recall_at_dual_floor_macro(y_true, y_proba, f1_min=0.5, prec_min=0.5, thresholds=None):
    if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
        y_proba = y_proba[:, 1]
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)  # 0.05~0.95, step=0.005

    best_recall = -1.0
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
        if (f1m >= f1_min) and (precm >= prec_min):
            recm = recall_score(y_true, y_pred, average="macro", zero_division=0)
            if recm > best_recall:
                best_recall = recm
    return best_recall  # 전부 미달이면 -1.0 (탈락)

scorer = make_scorer(
    recall_at_dual_floor_macro,
    needs_proba=True,
    f1_min=0.5, 
    prec_min=0.5,
    thresholds=np.linspace(0.05, 0.95, 181)
)

# =========================
# 5) 튜닝 설정 (CV/불균형 보정)
# =========================
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / max(pos, 1)

min_class = int(y_train.value_counts().min())
cv_folds  = min(5, max(2, min_class))
cv        = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

xgb_base = XGBClassifier(
    random_state=42, scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", tree_method="hist"
)

param_distributions = {
    "n_estimators":       [200, 400, 600, 800, 1000],
    "max_depth":          [3, 4, 5, 6, 7, 8],
    "learning_rate":      [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample":          [0.6, 0.8, 1.0],
    "colsample_bytree":   [0.6, 0.8, 1.0],
    "min_child_weight":   [1, 3, 5, 7],
    "gamma":              [0.0, 0.2, 0.5, 1.0, 2.0],
    "reg_lambda":         [0.0, 1.0, 5.0, 10.0],
}

tuner = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=60,
    scoring=scorer,          # ★ macro dual-floor + recall_macro 최대화
    cv=cv,
    n_jobs=-1,
    refit=True,
    random_state=42,
    verbose=1,
    error_score=-1.0
)

tuner.fit(Xtr, y_train)
best_model  = tuner.best_estimator_
best_params = tuner.best_params_
best_cv     = tuner.best_score_
print("\n[튜닝 완료] best_cv(recall_macro @ F1_macro≥0.5 & Prec_macro≥0.5) =", round(best_cv, 5))
print("[best_params]", best_params)

# =========================
# 6) 최종 threshold 선택 (OOF 확률로 동일 규칙 적용)
# =========================
def pick_threshold_dual_floor_macro(y_true, y_proba, f1_min=0.5, prec_min=0.5, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    best = {"thr": None, "recall_macro": -1.0, "f1_macro": None, "prec_macro": None}
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
        if (f1m >= f1_min) and (precm >= prec_min):
            recm = recall_score(y_true, y_pred, average="macro", zero_division=0)
            if recm > best["recall_macro"]:
                best = {"thr": thr, "recall_macro": recm, "f1_macro": f1m, "prec_macro": precm}
    return best

y_proba_oof = cross_val_predict(best_model, Xtr, y_train, cv=cv, method="predict_proba")[:, 1]
pick = pick_threshold_dual_floor_macro(y_train, y_proba_oof, f1_min=0.5, prec_min=0.5)
final_thr = 0.5 if pick["thr"] is None else pick["thr"]
print(f"\n[최종 threshold] {final_thr:.3f} | OOF recall_macro={pick['recall_macro']}, "
      f"OOF f1_macro={pick['f1_macro']}, OOF prec_macro={pick['prec_macro']}")

# =========================
# 7) 성능 평가: TRAIN(in-sample) / OOF / TEST (모두 '전체 macro' 기준)
# =========================
def report_block(name, y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n[{name} 성능(전체 macro 기준)]")
    print("Precision-macro :", precision_score(y_true, y_pred, average='macro', zero_division=0))
    print("Recall-macro    :", recall_score(y_true, y_pred, average='macro', zero_division=0))  # = Balanced Acc(이진)
    print("F1-macro        :", f1_score(y_true, y_pred, average='macro', zero_division=0))
    print("Balanced Acc    :", balanced_accuracy_score(y_true, y_pred))
    if name == "TEST":
        print("ROC-AUC         :", roc_auc_score(y_true, y_prob))
    print("Accuracy        :", accuracy_score(y_true, y_pred))
    print(f"\n[참고: 양성(1) 기준 - {name}]")
    print("Precision(1)    :", precision_score(y_true, y_pred, zero_division=0))
    print("Recall(1)       :", recall_score(y_true, y_pred, zero_division=0))
    print("F1(1)           :", f1_score(y_true, y_pred, zero_division=0))
    print(f"Confusion Matrix {name} (tn, fp, fn, tp) = ({tn}, {fp}, {fn}, {tp})")
    if name == "TEST":
        print(f"\n[Classification Report - {name}]")
        print(classification_report(y_true, y_pred, digits=3))

# TRAIN (in-sample: 낙관적일 수 있음)
y_prob_tr = best_model.predict_proba(Xtr)[:, 1]
report_block("TRAIN", y_train, y_prob_tr, final_thr)

# OOF (누수 없는 훈련셋 추정)
y_pred_oof = (y_proba_oof >= final_thr).astype(int)
print("\n[OOF 성능(훈련셋, 누수 없음, 전체 macro 기준)]")
print("Precision-macro :", precision_score(y_train, y_pred_oof, average='macro', zero_division=0))
print("Recall-macro    :", recall_score(y_train, y_pred_oof, average='macro', zero_division=0))
print("F1-macro        :", f1_score(y_train, y_pred_oof, average='macro', zero_division=0))
print("Balanced Acc    :", balanced_accuracy_score(y_train, y_pred_oof))
print("Accuracy        :", accuracy_score(y_train, y_pred_oof))
print("\n[참고: 양성(1) 기준 - OOF]")
print("Precision(1)    :", precision_score(y_train, y_pred_oof, zero_division=0))
print("Recall(1)       :", recall_score(y_train, y_pred_oof, zero_division=0))
print("F1(1)           :", f1_score(y_train, y_pred_oof, zero_division=0))
tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_train, y_pred_oof).ravel()
print(f"Confusion Matrix OOF (tn, fp, fn, tp) = ({tn_o}, {fp_o}, {fn_o}, {tp_o})")

# TEST
y_prob_te = best_model.predict_proba(Xte)[:, 1]
report_block("TEST", y_test, y_prob_te, final_thr)

import joblib
from datetime import datetime

artifact = {
    "model": best_model,            # XGBClassifier (튜닝 완료 모델)
    "features": manual_features,    # 학습에 사용한 컬럼명 리스트
    "threshold": final_thr,         # 최종 분류 임계값
    "best_params": best_params,     # 최적 하이퍼파라미터
    "cv_best_score": best_cv,       # 튜닝 스코어(옵션)
    "created_at": datetime.now().isoformat(),
    "notes": "F1_macro≥0.5 & Precision_macro≥0.5 floor, Recall_macro 최대화"
}

save_path = "xgb_model.pkl"
joblib.dump(artifact, save_path)
print(f"Saved to {save_path}")

# LightGBM
# =========================
# 1) 데이터 로드 & 전처리
# =========================
df = pd.read_excel("data/dataset.xlsx", index_col="시설명")
dong_df = pd.read_excel("data/읍면동별데이터1.xlsx")
dong_df["읍면동"] = dong_df["읍면동"].astype(str).str.extract(r'^(?:\S+\s+){2}(\S+)')
df = df.merge(dong_df, on="읍면동", how="left")

def nz(x, eps=1e-6): 
    return np.clip(x, eps, None)

# 파생 (분모 보호)
df["교통량_도로폭비"]   = df["교통량"] / nz(df["보호구역도로폭"])
df["어린이밀도"]       = df["어린이인구"] / nz(df["면적"])
df["인구밀도"]         = df["전체인구"]   / nz(df["면적"])
df["교통밀도"]         = df["교통량"]     / nz(df["면적"])
df["속도_도로폭비"]     = df["주행속도"] / nz(df["보호구역도로폭"])
df["주정차위반_교통량"] = df["불법주정차위반"] / nz(df["교통량"])
df["주정차위반_도로폭"] = df["불법주정차위반"] / nz(df["보호구역도로폭"])

# 불필요 컬럼 제거(있을 때만)
df = df.drop(columns=["주소","읍면동","어린이인구","전체인구","면적","구역지정수"], errors="ignore")

# 수치형 변환 + inf 처리
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# 2) 분할
# =========================
target_col = "사고건수"
X_full = df.drop(columns=[target_col])
y_full = (df[target_col] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# =========================
# 3) 사용할 변수(직접 선택) — 비워두면 전체 사용
# =========================
manual_features = [
    '시설물 CCTV 수','시설물 도로표지판 수','시설물 과속방지턱 수',
    '보호구역도로폭','위도','경도','신호등_반경300m',
    '주행속도','불법주정차위반','교통량_도로폭비','교통밀도','경사도'
]
if len(manual_features) == 0:
    manual_features = list(X_train.columns)
else:
    missing = [c for c in manual_features if c not in X_train.columns]
    assert not missing, f"다음 컬럼이 데이터에 없습니다: {missing}"

Xtr = X_train[manual_features].copy()
Xte = X_test[manual_features].copy()

# =========================
# 4) 스코어러(전체 macro 기준):
#    F1_macro ≥ 0.5 & Precision_macro ≥ 0.5 만족하는 threshold 중 Recall_macro 최대화
# =========================
def recall_at_dual_floor_macro(y_true, y_proba, f1_min=0.5, prec_min=0.5, thresholds=None):
    if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
        y_proba = y_proba[:, 1]
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)  # 0.05~0.95, step=0.005

    best_recall = -1.0
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
        if (f1m >= f1_min) and (precm >= prec_min):
            recm = recall_score(y_true, y_pred, average="macro", zero_division=0)
            if recm > best_recall:
                best_recall = recm
    return best_recall  # 전부 미달이면 -1.0 (탈락)

from sklearn.metrics import make_scorer
scorer = make_scorer(
    recall_at_dual_floor_macro,
    needs_proba=True,
    f1_min=0.5, 
    prec_min=0.5,
    thresholds=np.linspace(0.05, 0.95, 181)
)

# =========================
# 5) 튜닝 설정 (CV/불균형 보정)
# =========================
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / max(pos, 1)

min_class = int(y_train.value_counts().min())
cv_folds  = min(5, max(2, min_class))
cv        = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

lgbm_base = LGBMClassifier(
    objective="binary",
    random_state=42,
    n_jobs=-1,
    # 불균형 보정
    scale_pos_weight=scale_pos_weight,
    # LightGBM은 NaN을 내부적으로 처리 가능
)

# LightGBM 하이퍼파라미터 공간
param_distributions = {
    "n_estimators":       [300, 500, 800, 1000, 1500],
    "learning_rate":      [0.01, 0.03, 0.05, 0.1],
    "num_leaves":         [15, 31, 47, 63, 95],
    "max_depth":          [-1, 3, 5, 7, 9],
    "min_child_samples":  [5, 10, 20, 30, 50, 100],
    "subsample":          [0.6, 0.8, 1.0],     # (= bagging_fraction)
    "subsample_freq":     [0, 1, 3, 5],        # (= bagging_freq)
    "colsample_bytree":   [0.6, 0.8, 1.0],     # (= feature_fraction)
    "reg_lambda":         [0.0, 1.0, 5.0, 10.0],
    "reg_alpha":          [0.0, 0.1, 0.5, 1.0],
}

tuner = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_distributions,
    n_iter=60,                 # ↑ 늘리면 정밀/시간 증가
    scoring=scorer,            # ★ macro dual-floor + recall_macro 최대화
    cv=cv,
    n_jobs=-1,
    refit=True,
    random_state=42,
    verbose=1,
    error_score=-1.0
)

tuner.fit(Xtr, y_train)
best_model  = tuner.best_estimator_
best_params = tuner.best_params_
best_cv     = tuner.best_score_
print("\n[튜닝 완료] best_cv(recall_macro @ F1_macro≥0.5 & Prec_macro≥0.5) =", round(best_cv, 5))
print("[best_params]", best_params)

# =========================
# 6) 최종 threshold 선택 (OOF 확률로 동일 규칙 적용)
# =========================
def pick_threshold_dual_floor_macro(y_true, y_proba, f1_min=0.5, prec_min=0.5, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    best = {"thr": None, "recall_macro": -1.0, "f1_macro": None, "prec_macro": None}
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
        if (f1m >= f1_min) and (precm >= prec_min):
            recm = recall_score(y_true, y_pred, average="macro", zero_division=0)
            if recm > best["recall_macro"]:
                best = {"thr": thr, "recall_macro": recm, "f1_macro": f1m, "prec_macro": precm}
    return best

# LightGBM OOF 확률
y_proba_oof = cross_val_predict(best_model, Xtr, y_train, cv=cv, method="predict_proba")[:, 1]
pick = pick_threshold_dual_floor_macro(y_train, y_proba_oof, f1_min=0.5, prec_min=0.5)
final_thr = 0.5 if pick["thr"] is None else pick["thr"]
print(f"\n[최종 threshold] {final_thr:.3f} | OOF recall_macro={pick['recall_macro']}, "
      f"OOF f1_macro={pick['f1_macro']}, OOF prec_macro={pick['prec_macro']}")

# =========================
# 7) 성능 평가: TRAIN(in-sample) / OOF / TEST (모두 '전체 macro' 기준)
# =========================
def report_block(name, y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n[{name} 성능(전체 macro 기준)]")
    print("Precision-macro :", precision_score(y_true, y_pred, average='macro', zero_division=0))
    print("Recall-macro    :", recall_score(y_true, y_pred, average='macro', zero_division=0))  # = Balanced Acc(이진)
    print("F1-macro        :", f1_score(y_true, y_pred, average='macro', zero_division=0))
    print("Balanced Acc    :", balanced_accuracy_score(y_true, y_pred))
    if name == "TEST":
        print("ROC-AUC         :", roc_auc_score(y_true, y_prob))
    print("Accuracy        :", accuracy_score(y_true, y_pred))
    print(f"\n[참고: 양성(1) 기준 - {name}]")
    print("Precision(1)    :", precision_score(y_true, y_pred, zero_division=0))
    print("Recall(1)       :", recall_score(y_true, y_pred, zero_division=0))
    print("F1(1)           :", f1_score(y_true, y_pred, zero_division=0))
    print(f"Confusion Matrix {name} (tn, fp, fn, tp) = ({tn}, {fp}, {fn}, {tp})")
    if name == "TEST":
        print(f"\n[Classification Report - {name}]")
        print(classification_report(y_true, y_pred, digits=3))

# TRAIN
y_prob_tr = best_model.predict_proba(Xtr)[:, 1]
report_block("TRAIN", y_train, y_prob_tr, final_thr)

# OOF
y_pred_oof = (y_proba_oof >= final_thr).astype(int)
print("\n[OOF 성능(훈련셋, 누수 없음, 전체 macro 기준)]")
print("Precision-macro :", precision_score(y_train, y_pred_oof, average='macro', zero_division=0))
print("Recall-macro    :", recall_score(y_train, y_pred_oof, average='macro', zero_division=0))
print("F1-macro        :", f1_score(y_train, y_pred_oof, average='macro', zero_division=0))
print("Balanced Acc    :", balanced_accuracy_score(y_train, y_pred_oof))
print("Accuracy        :", accuracy_score(y_train, y_pred_oof))
print("\n[참고: 양성(1) 기준 - OOF]")
print("Precision(1)    :", precision_score(y_train, y_pred_oof, zero_division=0))
print("Recall(1)       :", recall_score(y_train, y_pred_oof, zero_division=0))
print("F1(1)           :", f1_score(y_train, y_pred_oof, zero_division=0))
tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_train, y_pred_oof).ravel()
print(f"Confusion Matrix OOF (tn, fp, fn, tp) = ({tn_o}, {fp_o}, {fn_o}, {tp_o})")

# TEST
y_prob_te = best_model.predict_proba(Xte)[:, 1]
report_block("TEST", y_test, y_prob_te, final_thr)

# =========================
# 8) 아티팩트 저장 (.pkl)
# =========================
artifact = {
    "model": best_model,            # LGBMClassifier (튜닝 완료 모델)
    "features": manual_features,    # 학습에 사용한 컬럼명 리스트
    "threshold": final_thr,         # 최종 분류 임계값
    "best_params": best_params,     # 최적 하이퍼파라미터
    "cv_best_score": best_cv,       # 튜닝 스코어(옵션)
    "created_at": datetime.now().isoformat(),
    "notes": "LightGBM | F1_macro≥0.5 & Precision_macro≥0.5 floor, Recall_macro 최대화"
}

save_path = "lgbm_model.pkl"
joblib.dump(artifact, save_path)
print(f"Saved to {save_path}")

from matplotlib import font_manager, rcParams
from sklearn.inspection import PartialDependenceDisplay
import joblib

# ===== 1) 한글 폰트/마이너스 처리 =====
font_candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
available = {f.name for f in font_manager.fontManager.ttflist}
for f in font_candidates:
    if f in available:
        rcParams["font.family"] = f
        break
rcParams["axes.unicode_minus"] = False

# ===== 2) pkl 아티팩트 로드 (모델/피처/임계값 등) =====
artifact_path = "xgb_model.pkl"     # ← 저장한 파일명
artifact = joblib.load(artifact_path)
model = artifact["model"]
feat_all = artifact["features"]     # 학습에 사용한 피처 목록(순서 포함)

# ===== 3) 데이터 로드 & 전처리 (학습 시와 동일하게) =====
#    * 파이프라인을 저장하지 않았다면, 학습 때 했던 파생/전처리를 동일하게 수행해야 함
df = pd.read_excel("data/dataset.xlsx", index_col="시설명")
dong_df = pd.read_excel("data/읍면동별데이터1.xlsx")
dong_df["읍면동"] = dong_df["읍면동"].astype(str).str.extract(r'^(?:\S+\s+){2}(\S+)')
df = df.merge(dong_df, on="읍면동", how="left")

def nz(x, eps=1e-6): return np.clip(x, eps, None)
df["교통량_도로폭비"]   = df["교통량"] / nz(df["보호구역도로폭"])
df["어린이밀도"]       = df["어린이인구"] / nz(df["면적"])
df["인구밀도"]         = df["전체인구"]   / nz(df["면적"])
df["교통밀도"]         = df["교통량"]     / nz(df["면적"])
df["속도_도로폭비"]     = df["주행속도"] / nz(df["보호구역도로폭"])
df["주정차위반_교통량"] = df["불법주정차위반"] / nz(df["교통량"])
df["주정차위반_도로폭"] = df["불법주정차위반"] / nz(df["보호구역도로폭"])
# 불필요 컬럼 제거(있을 때만)
df = df.drop(columns=["주소","읍면동","어린이인구","전체인구","면적","구역지정수"], errors="ignore")
# 수치형 변환 + inf 처리
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)

X_pdp = df[feat_all].copy()

# ========= 3) 개별 PDP로 그릴 변수 목록 =========
features_to_plot = [
    '시설물 CCTV 수','시설물 도로표지판 수','시설물 과속방지턱 수',
    '보호구역도로폭','신호등_반경300m','주행속도',
    '불법주정차위반','교통량_도로폭비','교통밀도'
]
features_to_plot = [f for f in features_to_plot if f in X_pdp.columns]
assert features_to_plot, "PDP로 그릴 유효한 변수명이 없습니다."

# ========= 4) 저장 폴더(선택) =========
save_dir = "pdp_plots"
os.makedirs(save_dir, exist_ok=True)

# ========= 5) 한 변수씩 개별 Figure로 PDP 그리기 =========
for f in features_to_plot:
    idx = X_pdp.columns.get_loc(f)  # 버전 호환을 위해 인덱스로 전달
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    disp = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_pdp,
        features=[idx],               # 단일 피처만
        percentiles=(0.05, 0.95),
        grid_resolution=100,          # 곡선 매끄럽게
        kind="average",               # ICE도 보고 싶으면 "both"
        target=1,                     # 양성(1) 기준 PDP
        ax=ax
    )

    ax.set_title(f"PDP - {f}", fontsize=14, pad=10)
    ax.tick_params(axis="x", labelrotation=20, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.margins(x=0.03)

    # 파일 저장(원치 않으면 주석)
    safe = re.sub(r'[^0-9A-Za-z가-힣_.+-]+', '_', f)
    out  = os.path.join(save_dir, f"pdp_{safe}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


df.info()

# ===== 1) 아티팩트 로드 =====
artifact_path = "xgb_model.pkl"  # ← 네 pkl 경로
artifact  = joblib.load(artifact_path)
model     = artifact["model"]            # 학습된 XGBClassifier
feat_list = list(artifact["features"])   # 학습에 사용한 컬럼명
thr       = float(artifact.get("threshold", 0.5))  # 최종 임계값(없으면 0.5)

# ===== 2) (필요 시) 누락 파생컬럼 자동 생성 =====
def _nz(s, eps=1e-6):
    return np.clip(s.astype(float), eps, None)

def ensure_features(df: pd.DataFrame, need: list[str]) -> pd.DataFrame:
    df = df.copy()
    # 숫자형 강제 변환 (비수치 → NaN)
    for c in need:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2-1) 교통량_도로폭비 = 교통량 / 보호구역도로폭
    if ("교통량_도로폭비" in need) and ("교통량_도로폭비" not in df.columns):
        if {"교통량", "보호구역도로폭"}.issubset(df.columns):
            df["교통량_도로폭비"] = df["교통량"].astype(float) / _nz(df["보호구역도로폭"])
        # else: 못 만들면 아래에서 누락 리스트로 안내

    # 2-2) 교통밀도 = 교통량 / 면적
    if ("교통밀도" in need) and ("교통밀도" not in df.columns):
        if {"교통량", "면적"}.issubset(df.columns):
            df["교통밀도"] = df["교통량"].astype(float) / _nz(df["면적"])
        # 면적이 아예 없다면 아래에서 누락 안내

    # 필요하면 추가 파생 규칙 여기에 …

    # 최종 누락 점검
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(
            f"예측에 필요한 컬럼이 없습니다: {missing}\n"
            f"데이터에 존재하는 컬럼 예: {list(df.columns)[:15]} ..."
        )
    return df

df_ready = ensure_features(df, feat_list)

# ===== 3) 입력행렬 구성(학습 때와 동일한 피처 순서) + 안전 변환 =====
X_inf = df_ready[feat_list].apply(pd.to_numeric, errors="coerce")
# XGBoost는 NaN을 처리할 수 있으므로 별도 대치 없이 진행(원하면 SimpleImputer 추가 가능)

# ===== 4) 예측 확률/라벨 산출 =====
y_prob = model.predict_proba(X_inf)[:, 1]
y_pred = (y_prob >= thr).astype(int)

# ===== 5) 원본 df에 컬럼 추가 (위험도 점수도 함께) =====
df["pred_proba"] = y_prob
df["pred_label"] = y_pred
df["risk_score"] = (y_prob * 100).round(1)  # 0~100 점수
df.head()
print(f"[DONE] 예측 완료  |  threshold={thr:.3f}")
print(df[["pred_label", "pred_proba", "risk_score"]].head())
df.to_excel("점수포함.xlsx", index=False)


import xgboost as xgb
import shap

print("xgboost:", xgb.__version__)
print("shap   :", shap.__version__)

df = pd.read_excel("data/점수포함.xlsx")
df['사고유무'] = (df['사고건수'] > 0).astype(int)  # 사고발생=1

# =============================================
# ==== LightGBM: 튜닝 → 임계값 선택(OOF) → Train/Test 평가 ====
# ==== CatBoost: 튜닝 → 임계값 선택(OOF) → Train/Test 평가 ====
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             balanced_accuracy_score, accuracy_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             make_scorer)

# CatBoost 설치 확인
try:
    from catboost import CatBoostClassifier
except Exception as e:
    raise RuntimeError("catboost 패키지가 필요합니다. pip install catboost 로 설치하세요.") from e

# --- 커스텀 스코어러: F1_macro ≥ 0.5를 만족하는 임계값들 중 Recall_macro 최대화 ---
def recall_at_f1_floor_macro(y_true, y_proba, f1_min=0.5, thresholds=None):
    if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
        y_proba = y_proba[:, 1]
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)  # 0.05~0.95 step 0.005
    best = -1.0
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        if f1_score(y_true, y_pred, average="macro", zero_division=0) >= f1_min:
            recm = recall_score(y_true, y_pred, average="macro", zero_division=0)
            if recm > best:
                best = recm
    return best

def pick_threshold_f1_floor(y_true, y_proba, f1_min=0.5, thresholds=None):
    if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
        y_proba = y_proba[:, 1]
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    best = {"thr": None, "recall_macro": -1.0, "f1_macro": None}
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1m >= f1_min:
            recm = recall_score(y_true, y_pred, average="macro", zero_division=0)
            if recm > best["recall_macro"]:
                best = {"thr": thr, "recall_macro": recm, "f1_macro": f1m}
    return best

scorer = make_scorer(
    recall_at_f1_floor_macro, needs_proba=True,
    f1_min=0.5, thresholds=np.linspace(0.05, 0.95, 181)
)

# --- CV/불균형 보정 (y_train 필요) ---
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / max(pos, 1)
min_class = int(y_train.value_counts().min())
cv_folds  = min(5, max(2, min_class))
cv        = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

# --- 모델 & 탐색공간 ---
cat = CatBoostClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    thread_count=-1,
    verbose=False,
    loss_function="Logloss"
)
param_distributions = {
    "iterations":   [400, 600, 800, 1000, 1200],
    "depth":        [4, 6, 8, 10],
    "learning_rate":[0.01, 0.03, 0.05, 0.1],
    "l2_leaf_reg":  [1.0, 3.0, 5.0, 10.0],
    "subsample":    [0.6, 0.8, 1.0],
    "border_count": [32, 64, 128]
}

# --- 튜닝 ---
tuner = RandomizedSearchCV(
    estimator=cat,
    param_distributions=param_distributions,
    n_iter=60,
    scoring=scorer,       # F1_macro≥0.5 하에서 Recall_macro 최대화
    cv=cv,
    n_jobs=-1,
    refit=True,
    random_state=42,
    verbose=1,
    error_score=-1.0
)
tuner.fit(Xtr, y_train)
cat_best  = tuner.best_estimator_
cat_params = tuner.best_params_
cat_cv     = tuner.best_score_
print(f"[CatBoost] best_cv(recall_macro @ F1_macro>=0.5) = {cat_cv:.5f}")
print(f"[CatBoost] best_params = {cat_params}")

# --- 임계값 선택(OOF 확률) ---
y_proba_oof = cross_val_predict(cat_best, Xtr, y_train, cv=cv, method="predict_proba")[:, 1]
pick = pick_threshold_f1_floor(y_train, y_proba_oof, f1_min=0.5)
cat_thr = 0.5 if pick["thr"] is None else pick["thr"]
print(f"[CatBoost] 최종 threshold = {cat_thr:.3f} | "
      f"OOF recall_macro={pick['recall_macro']} | OOF f1_macro={pick['f1_macro']}")

# --- 공통 리포트 함수 ---
def report_block(name, y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n[{name}]")
    print("Precision-macro :", precision_score(y_true, y_pred, average='macro', zero_division=0))
    print("Recall-macro    :", recall_score(y_true, y_pred, average='macro', zero_division=0))
    print("F1-macro        :", f1_score(y_true, y_pred, average='macro', zero_division=0))
    print("Balanced Acc    :", balanced_accuracy_score(y_true, y_pred))
    print("Accuracy        :", accuracy_score(y_true, y_pred))
    print("ROC-AUC (prob)  :", roc_auc_score(y_true, y_prob))
    print(f"Confusion Matrix: (tn, fp, fn, tp) = ({tn}, {fp}, {fn}, {tp})")
    print(classification_report(y_true, y_pred, digits=3))

# --- Train/Test 평가 (Xtr, Xte / y_train, y_test 필요) ---
y_prob_tr = cat_best.predict_proba(Xtr)[:, 1]
report_block("CatBoost-TRAIN", y_train, y_prob_tr, cat_thr)

y_prob_te = cat_best.predict_proba(Xte)[:, 1]
report_block("CatBoost-TEST", y_test, y_prob_te, cat_thr)