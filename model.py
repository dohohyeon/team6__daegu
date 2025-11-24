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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             make_scorer)
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import optuna
import joblib
import shap
from sklearn.linear_model import LogisticRegression


df = pd.read_excel("data/dataset.xlsx", index_col="시설명")
dong_df = pd.read_excel("data/읍면동별데이터1.xlsx")

df['읍면동']
dong_df['읍면동'] = dong_df["읍면동"].str.extract(r'^(?:\S+\s+){2}(\S+)')


df = df.merge(dong_df, on="읍면동", how="left")
df.columns
df.to_excel("df.xlsx", index=False)


df["교통량_도로폭비"] = df["교통량"] / (df["보호구역도로폭"])
df["어린이밀도"] = df["어린이인구"] / (df["면적"])
df["인구밀도"] = df["전체인구"] / (df["면적"])
df["교통밀도"] = df["교통량"] / (df["면적"])
df["속도_도로폭비"] = df["주행속도"] / (df["보호구역도로폭"])
df["주정차위반_교통량"] = df["불법주정차위반"] / (df["교통량"])
df["주정차위반_도로폭"] = df["불법주정차위반"] / (df["교통량"])


df = df.drop(columns=['주소','읍면동','어린이인구','전체인구','면적',])
df.columns


# xgboost에서 최적의 변수 선택(EFS)

# X, y 준비 (사고건수 제외한 독립변수 전체)
target_col = "사고건수"
X = df.drop(columns=[target_col])   # 사고건수 제외
y = (df[target_col] > 0).astype(int)

# Train/Test 분할 (8:2 stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 클래스 불균형 보정
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# -------------------
# 2) XGBoost 분류기 정의
# -------------------
xgb_clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

fixed_features = [
    '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수',
    '보호구역도로폭', '위도', '경도', '신호등_반경300m'
]

candidates = [col for col in X_train.columns if col not in fixed_features]

# -------------------
# 2) EFS는 candidates에서만 실행
# -------------------
efs = EFS(
    estimator=xgb_clf,
    min_features=1,  # 최소 선택 개수 보정
    max_features=len(candidates),
    scoring=make_scorer(f1_score),
    cv=5,
    n_jobs=-1
)

efs = efs.fit(X_train[candidates], y_train)

# -------------------
# 3) 최종 선택된 변수 = 고정 변수 + 탐색 변수 결과
# -------------------
best_features = fixed_features + list(efs.best_feature_names_)
print("최적 변수 조합:", best_features)
print("Train 교차검증 최고 recall-score:", efs.best_score_)

# -------------------
# 4) 최적 변수로 학습/평가
# -------------------
best_model = xgb_clf.fit(X_train[best_features], y_train)

y_prob = best_model.predict_proba(X_test[best_features])[:, 1]

# 2) threshold = 0.3 적용
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

print("\n[테스트셋 성능 평가]")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred))

import optuna
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500), # 2000
        "max_depth": trial.suggest_int("max_depth", 3, 8), # 2~ 15
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), # 0.5~
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), # 0.5~ 
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6), # ~ 20
        "gamma": trial.suggest_float("gamma", 0.0, 2.0), # 10
        "scale_pos_weight": scale_pos_weight,   # 클래스 불균형 보정 그대로 사용
        "random_state": 42,
        "eval_metric": "logloss"
    }
    
    # XGBoost 분류기 생성
    model = XGBClassifier(**params)
    
    # Recall 기준 5-Fold 교차검증
    scores = cross_val_score(
        model, X_train[best_features], y_train,
        cv=5,
        scoring=make_scorer(f1_score)
    )
    return scores.mean()

# -------------------
# 6) Optuna 실행
# -------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)   # 탐색 횟수 조정 가능 (예: 50~100)

print("Best params:", study.best_params)
print("Best CV Recall:", study.best_value)

# -------------------
# 7) 최적 파라미터로 모델 학습
# -------------------
best_params = study.best_params
best_params.update({"scale_pos_weight": scale_pos_weight, "random_state": 42, "eval_metric": "logloss"})

best_model = XGBClassifier(**best_params)
best_model.fit(X_train[best_features], y_train)

# -------------------
# 8) 테스트셋 예측 (threshold=0.3)
# -------------------
y_prob = best_model.predict_proba(X_test[best_features])[:, 1]
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

print("\n[테스트셋 성능 평가 - Optuna 최적 파라미터]")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=3))

best_features
df.columns


# -------------------
# 1) 데이터 준비
# -------------------
target_col = "사고건수"
best_features = ['시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수',
                 '보호구역도로폭', '위도', '경도', '신호등_반경300m', '불법주정차위반', '경사도',
                 '면적', '주행속도', '어린이밀도', '인구밀도']


X = df[best_features]
y = (df[target_col] > 0).astype(int)

# Train/Test 분할 (8:2, stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 클래스 불균형 보정 weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# -------------------
# 2) Optuna objective 함수 정의
# -------------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring=make_scorer(f1_score)
    )
    return scores.mean()

# -------------------
# 3) Optuna 실행
# -------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  # 탐색 횟수 조정 가능

print("Best params:", study.best_params)
print("Best CV f1-score:", study.best_value)

# -------------------
# 4) 최적 파라미터로 모델 학습
# -------------------
best_params = study.best_params
best_params.update({
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "eval_metric": "logloss",
    "use_label_encoder": False
})

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# -------------------
# 5) 테스트셋 평가
# -------------------
y_prob = best_model.predict_proba(X_test)[:, 1]
threshold = 0.5  # 원하는 threshold
y_pred = (y_prob >= threshold).astype(int)

print("\n[테스트셋 성능 평가 - Optuna 최적 파라미터]")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=3))

# -------------------
# 6) 모델 저장 (옵션)
# -------------------
joblib.dump(best_model, "best_model.pkl")
print("모델 저장 완료: best_model.pkl")


# Logistic Regression

df = pd.read_excel("data/dataset.xlsx", index_col="시설명")
dong_df = pd.read_excel("data/읍면동별데이터1.xlsx")

df['읍면동']
dong_df['읍면동'] = dong_df["읍면동"].str.extract(r'^(?:\S+\s+){2}(\S+)')


df = df.merge(dong_df, on="읍면동", how="left")

df["교통량_도로폭비"] = df["교통량"] / (df["보호구역도로폭"])
df["어린이밀도"] = df["어린이인구"] / (df["면적"])
df["교통밀도"] = df["교통량"] / (df["면적"])
df["속도_도로폭비"] = df["주행속도"] / (df["보호구역도로폭"])

df = df.drop(columns=['주소','읍면동'])
target_col = "사고건수"
best_features = [
    '시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수',
    '보호구역도로폭', '위도', '경도', '신호등_반경300m',
    '경사도', '면적', '교통량', '주행속도', '어린이밀도'
]

X = df[best_features]
y = (df[target_col] > 0).astype(int)

# Train/Test 분할 (8:2 stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------
# 2) Optuna objective 함수 정의
# -------------------
def objective(trial):
    params = {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),  # 규제 강도
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),  # 규제 방식
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),  
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    }

    model = LogisticRegression(**params)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring=make_scorer(f1_score)
    )
    return scores.mean()

# -------------------
# 3) Optuna 실행
# -------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Logistic은 가볍기 때문에 50회면 충분

print("Best params:", study.best_params)
print("Best CV f1-score:", study.best_value)

# -------------------
# 4) 최적 파라미터로 모델 학습
# -------------------
best_params = study.best_params
best_params.update({
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42
})

best_model = LogisticRegression(**best_params)
best_model.fit(X_train, y_train)

# -------------------
# 5) 테스트셋 평가
# -------------------
y_prob = best_model.predict_proba(X_test)[:, 1]
threshold = 0.  # 원하는 threshold
y_pred = (y_prob >= threshold).astype(int)

print("\n[테스트셋 성능 평가 - Optuna 최적 파라미터]")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=3))

# 회귀계수 추출
coef_df = pd.DataFrame({
    "Feature": best_features,
    "Coefficient": best_model.coef_[0]
})

# 절편(Intercept)
intercept = best_model.intercept_[0]
coef_df.sort_values('Coefficient', ascending=False)


print("Intercept:", intercept)
print("\n회귀계수:")
print(coef_df)









# 1) 저장된 모델 불러오기
loaded_model = joblib.load("best_model.pkl")

# 2) 전체 데이터에 대해 예측 수행
#    확률 예측 (y=1일 확률)
df["y_prob"] = loaded_model.predict_proba(df[best_features])[:, 1]
df["위험도점수"] = np.round(df["y_prob"] * 100, 0).astype(int)

df.to_csv("점수포함.csv", index=False, encoding="utf-8-sig")



plt.rc("font", family="Malgun Gothic")   # 맑은 고딕
plt.rc("axes", unicode_minus=False)  # 마이너스(-) 기호 깨짐 방지

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

loaded_model = joblib.load("best_model.pkl")

# 2) PDP를 그리고 싶은 변수들 지정 (예시: 2개 변수)
features_to_plot = ['시설물 CCTV 수',
 '시설물 도로표지판 수',
 '시설물 과속방지턱 수',
 '보호구역도로폭',
 '위도',
 '경도',
 '신호등_반경300m',
 '경사도',
 '면적',
 '교통량',
 '주행속도',
 '어린이밀도']

# 3) PDP Plot
fig, ax = plt.subplots(figsize=(20, 12))
PartialDependenceDisplay.from_estimator(
    estimator=loaded_model,             # 불러온 모델
    X=X_train[best_features],           # 학습에 사용한 독립변수 데이터
    features=features_to_plot,          # PDP 보고 싶은 변수 리스트
    percentiles=(0.05, 0.95),           # 극단값 제외
    ax=ax
)

plt.suptitle("Partial Dependence Plot", fontsize=16)
plt.tight_layout()
plt.show()


# 1) 모델 불러오기
loaded_model = joblib.load("best_model.pkl")

# 2) SHAP explainer 생성 (Tree 기반 모델 전용)
explainer = shap.TreeExplainer(loaded_model)

# 3) SHAP value 계산 (예: 테스트셋 기준)
shap_values = explainer.shap_values(X_test[best_features])

# -----------------------------
# 📊 (1) Summary Plot (dot plot)
# -----------------------------
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test[best_features], show=False)
plt.title("SHAP Summary Plot (변수별 기여도 분포)", fontsize=14)
plt.tight_layout()
plt.show()

# -----------------------------
# 📊 (2) Bar Plot (평균 절댓값 중요도)
# -----------------------------
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test[best_features], plot_type="bar", show=False)
plt.title("SHAP Bar Plot (변수별 평균 중요도)", fontsize=14)
plt.tight_layout()
plt.show()

# -----------------------------
# 📊 (3) Dependence Plot (특정 변수)
# -----------------------------
shap.dependence_plot("교통량", shap_values, X_test[best_features])

# -----------------------------
# 📊 (4) Force Plot (개별 샘플)
# -----------------------------
i = 0  # 샘플 인덱스
shap.force_plot(
    explainer.expected_value,
    shap_values[i],
    X_test[best_features].iloc[i],
    matplotlib=True
)

# -----------------------------
# 📊 (5) Waterfall Plot (개별 샘플)
# -----------------------------
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[i],
        base_values=explainer.expected_value,
        data=X_test[best_features].iloc[i].values,
        feature_names=X_test[best_features].columns
    )
)