from shiny.express import ui, input, render
from shiny import ui as xui, reactive
import pandas as pd, re, unicodedata
import matplotlib.pyplot as plt
import math
import numpy as np
import textwrap
import seaborn as sns
import plotly.express as px
from shinywidgets import render_plotly
from faicons import icon_svg
import os
from matplotlib import font_manager

# 앱의 루트 디렉토리 기준
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def _set_korean_font():
    """배포 환경에서도 깨지지 않도록 한글 폰트를 www/fonts에서 강제 설정"""
    font_path = os.path.join(APP_DIR, "www", "fonts", "NanumGothic-Regular.ttf")

    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"   # Matplotlib
        print(f"✅ 한글 폰트 적용됨: {font_path}")
    else:
        print(f"⚠️ 경고: 한글 폰트 파일 없음 → {font_path}")
        plt.rcParams["font.family"] = "sans-serif"

    # 마이너스 부호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False

    # ✅ Plotly 기본 폰트도 NanumGothic으로 설정
    import plotly.io as pio
    pio.templates["nanum"] = pio.templates["plotly_white"].update(
        layout_font=dict(family="NanumGothic")
    )
    pio.templates.default = "nanum"


_set_korean_font()


# ── 데이터 로드/전처리 ─────────────────────────
df = pd.read_csv("./data/dataset.csv")        # 분석용(기존)
df_overview = pd.read_csv("./data/dataset.csv") # 개요 탭 전용





def normalize(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00A0"," ").replace("\u3000"," ")
    return re.sub(r"\s+", " ", s).strip()

# 주소에서 구/군 추출
pat = re.compile(r'(달서\s*구|달성\s*군|수성\s*구|북\s*구|남\s*구|동\s*구|서\s*구|중\s*구)\b')
df["구군"] = (
    df["주소"].astype(str).map(normalize)
      .str.extract(pat, expand=False)
      .str.replace(r"\s+", "", regex=True)
      .astype(str).str.strip()
)

label_to_code = {"북구":0,"남구":1,"동구":2,"서구":3,"중구":4,"수성구":5,"달서구":6,"달성군":7}
code_to_label = {v:k for k,v in label_to_code.items()}
df["구군코드"] = df["구군"].map(label_to_code)


# ── 집계/그리기 유틸 ───────────────────────────
def agg_mean(col):
    return (
        df.groupby("구군코드")[col]
          .mean()
          .reindex(range(8))
    )

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.text(
                p.get_x() + p.get_width()/2, h, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9
            )

def make_bargrid(series_triplets, ncols=2, title=None, figsize=(8.0, 2.0)):
    n = len(series_triplets)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(figsize[0], figsize[1] * nrows))

    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    for i, (series, stitle, ylabel) in enumerate(series_triplets):
        ax = axes[i]
        ax.bar(series.index, series.values, edgecolor="black")
        ax.set_title(stitle, pad=6, fontweight="bold")
        ax.set_xlabel("구군코드 (0=북구, …, 7=달성군)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"{k}\n({code_to_label[k]})" for k in range(8)])
        annotate_bars(ax)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=0.97, fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.28)
    return fig

# ── 선택값 → 그래프 구성 매핑 ───────────────────
pos_map = {
    "시설물 CCTV 수":      (agg_mean("시설물 CCTV 수"),      "구군(코드)별 평균 시설물 CCTV 수",     "평균 CCTV 수"),
    "시설물 도로표지판 수": (agg_mean("시설물 도로표지판 수"), "구군(코드)별 평균 도로표지판 수",       "평균 도로표지판 수"),
    "시설물 과속방지턱 수": (agg_mean("시설물 과속방지턱 수"), "구군(코드)별 평균 과속방지턱 수",       "평균 과속방지턱 수"),
    "보호구역도로폭":        (agg_mean("보호구역도로폭"),        "구군(코드)별 평균 보호구역도로폭",      "평균 보호구역도로폭"),
    "신호등 300m":          (agg_mean("신호등 300m"),          "구군(코드)별 300m 이내 평균 신호등 개수",   "300m 이내 평균 신호등 개수"),
}

neg_map = {
    "평균주행속도":       (agg_mean("평균주행속도"),       "구군(코드)별 평균 주행속도",            "평균 주행속도"),
    "불법주정차위반건수": (agg_mean("불법주정차위반건수"), "구군(코드)별 평균 불법 주정차 위반건수", "평균 불법 주정차 위반건수"),
    "사고건수":           (agg_mean("사고건수"),           "구군(코드)별 평균 사고건수",            "평균 사고건수"),
}

# ── 사분면(인프라 vs 사고) 유틸 ──────────────────
def _compute_quadrant_df():
    pos_cols = ['시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수', '보호구역도로폭', '신호등 300m']
    neg_cols = ['평균주행속도', '불법주정차위반건수', '사고건수']
    infra_cols = pos_cols + neg_cols

    infra_means = pd.DataFrame({col: agg_mean(col) for col in infra_cols})

    def minmax(s):
        vmin, vmax = s.min(), s.max()
        if vmax == vmin:
            return pd.Series(0.0, index=s.index)
        return (s - vmin) / (vmax - vmin)
    infra_norm01 = infra_means.apply(minmax, axis=0)

    for c in neg_cols:
        infra_norm01[c] = 1 - infra_norm01[c]

    infra_means['인프라점수'] = infra_norm01.mean(axis=1)

    acc_mean = agg_mean('사고건수')
    infra_total = infra_means['인프라점수'].mean()
    acc_total   = acc_mean.mean()

    infra_means['사고건수'] = acc_mean
    infra_means['구군명'] = infra_means.index.map(code_to_label)

    def classify(row):
        infra_high = row['인프라점수'] >= infra_total
        acc_high   = row['사고건수'] > acc_total
        if infra_high and acc_high:   return "인프라↑ & 사고↑"
        if (not infra_high) and (not acc_high): return "인프라↓ & 사고↓"
        if infra_high and (not acc_high):       return "인프라↑ & 사고↓"
        return "인프라↓ & 사고↑"

    infra_means['분류'] = infra_means.apply(classify, axis=1)
    return infra_means, infra_total, acc_total

def make_quadrant_fig(scale=4, names=None):
    dfq, infra_total, acc_total = _compute_quadrant_df()

    infra_min, infra_max = dfq["인프라점수"].min(), dfq["인프라점수"].max()
    acc_min, acc_max     = dfq["사고건수"].min(), dfq["사고건수"].max()

    if names is not None:
        dfq = dfq[dfq["구군명"].isin(names)]

    base = 7.2
    size = base * scale
    fig, ax = plt.subplots(figsize=(size, size))
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    colors = {
        "인프라↑ & 사고↑": 'red',
        "인프라↓ & 사고↓": 'green',
        "인프라↑ & 사고↓": 'blue',
        "인프라↓ & 사고↑": 'orange',
    }

    for cat, color in colors.items():
        sub = dfq[dfq['분류'] == cat]
        ax.scatter(
            sub['인프라점수'], sub['사고건수'],
            color=color, s=110, alpha=0.9, edgecolors="none", label=cat
        )
        for _, r in sub.iterrows():
            ax.text(r['인프라점수'] + 0.004, r['사고건수'], r['구군명'], fontsize=10)

    ax.axvline(infra_total, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='대구 인프라 평균')
    ax.axhline(acc_total,   color='gray', linestyle='--', linewidth=2, alpha=0.7, label='대구 사고 평균')

    ax.set_xlabel("인프라 종합점수\n(Min–Max scale)", labelpad=12)
    ax.set_ylabel("평균 사고건수(2020–2024)", labelpad=10)

    pad_x = (infra_max - infra_min) * 0.05
    pad_y = (acc_max - acc_min) * 0.05
    ax.set_xlim(infra_min - pad_x, infra_max + pad_x)
    ax.set_ylim(acc_min - pad_y, acc_max + pad_y)
    
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True, framealpha=0.95,
        facecolor="white", edgecolor="#ddd",
        fontsize=11, borderaxespad=0.0, handlelength=2.4,
    )

    ax.grid(True, linestyle=':', alpha=0.6)
    fig.subplots_adjust(left=0.12, right=0.76, top=0.98, bottom=0.22)
    return fig

# ── 레이더(절대 기준 점수화) 유틸 ─────────────────
_good_cols = ['시설물 CCTV 수', '시설물 도로표지판 수', '시설물 과속방지턱 수', '보호구역도로폭', '신호등 300m']
_bad_cols  = ['평균주행속도', '불법주정차위반건수', '사고건수']
_all_cols  = _good_cols + _bad_cols

# 절대 기준 점수 규칙 (👉 _rules_abs 사용)
_rules_abs = {
    '시설물 CCTV 수':        ([0, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, np.inf],   [10,20,30,40,50,60,70,80,90,100]),
    '시설물 도로표지판 수':  ([0,10,14,18,22,26,28,30,32,34, np.inf],           [10,20,30,40,50,60,70,80,90,100]),
    '시설물 과속방지턱 수': ([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf],           [10,20,30,40,50,60,70,80,90,100]),
    '보호구역도로폭':        ([0, 9,10,11,12,13,14,15,16,17, np.inf],            [10,20,30,40,50,60,70,80,90,100]),
    '신호등 300m':          ([0, 6, 7, 8, 9,10,11,12,13,14, np.inf],             [10,20,30,40,50,60,70,80,90,100]),
    # 부정지표: 역방향 라벨
    '평균주행속도':          ([0,30,32,34,36,38,40,42,44,46, np.inf],            [100,90,80,70,60,50,40,30,20,10]),
    '불법주정차위반건수':    ([0,20,30,40,50,60,70,80,90,100, np.inf],           [100,90,80,70,60,50,40,30,20,10]),
    '사고건수':              ([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50, np.inf], [100,90,80,70,60,50,40,30,20,10]),
}

_label_map = {'평균주행속도':'안전 주행 수준','불법주정차위반건수':'불법 주정차 안전 수준','사고건수':'사고 안전 수준'}
_radar_features = [_label_map.get(c, c) for c in _all_cols]

def _apply_absolute_scoring(df_means):
    out = pd.DataFrame(index=df_means.index)
    for col, (bins, labels) in _rules_abs.items():
        out[col] = pd.cut(df_means[col], bins=bins, labels=labels, include_lowest=True).astype(int)
    out = out.rename(columns=_label_map)
    return out

def _radar_base_table():
    all_means = pd.DataFrame({c: agg_mean(c) for c in _all_cols})
    all_means['구군명'] = all_means.index.map(code_to_label)

    total_row = all_means[_all_cols].mean().to_frame().T
    total_row['구군명'] = '대구 전체'
    total_row.index = [len(all_means)]

    all_means = pd.concat([all_means, total_row], ignore_index=False)
    score_table = _apply_absolute_scoring(all_means[_all_cols])
    score_table['구군명'] = all_means['구군명']
    return score_table

def _make_angles(n):
    ang = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    return ang

def _wrap_labels(labels, width=6):
    return ['\n'.join(textwrap.wrap(lbl, width=width)) for lbl in labels]

def _style_radar(ax, feature_names, angles, label_fs=11, pad_px=14):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), _wrap_labels(feature_names, 6), fontsize=label_fs)
    ax.tick_params(axis="x", pad=pad_px)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

def _plot_one(ax, series_vals, angles, label=None, alpha=0.18, linewidth=2):
    vals = series_vals.tolist() + series_vals.tolist()[:1]
    ax.plot(angles, vals, linewidth=linewidth, label=label, marker="o", markersize=4)
    ax.fill(angles, vals, alpha=alpha)

def make_radar_figure(names, include_total=True, scale=1.0):
    scores = _radar_base_table().set_index("구군명")
    plot_names = list(dict.fromkeys([*names, "대구 전체" if include_total else None]))
    plot_names = [n for n in plot_names if n in scores.index and n]
    if not plot_names:
        raise ValueError("선택한 구군이 없습니다.")

    angles = _make_angles(len(_radar_features))

    base_w, base_h = 10.0, 8.0
    w, h = base_w * scale, base_h * scale
    fig, ax = plt.subplots(figsize=(w, h), subplot_kw={"polar": True})

    legend_frac = 0.26
    margin_r    = 0.02
    left, bottom, top = 0.06, 0.12, 0.92
    right = 1.0 - legend_frac - margin_r

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    ax.set_position([left, bottom, right - left, top - bottom])

    _style_radar(ax, _radar_features, angles, label_fs=12, pad_px=14)

    sel = scores.loc[plot_names, _radar_features]
    for nm, row in sel.iterrows():
        _plot_one(ax, row, angles, label=nm, alpha=0.20, linewidth=2.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.03),
        ncol=1,
        frameon=True, framealpha=0.95,
        facecolor="white", edgecolor="#e5e7eb",
        fontsize=11, handlelength=2.4, labelspacing=0.8,
    )

# --- 부록용 그룹 (지표명은 _rules_abs의 key와 동일) ---
groups = {
    "사고·행동 위험": ["사고건수", "평균주행속도", "불법주정차위반건수"],
    "가시성·안내":   ["시설물 CCTV 수", "시설물 도로표지판 수"],
    "속도저감·제어": ["시설물 과속방지턱 수", "신호등 300m"],
    "도로환경":       ["보호구역도로폭"],
}
units_map = {"보호구역도로폭": "m"}

def indicator_table_html(name):
    bins, scores = _rules_abs[name]
    unit = units_map.get(name, "")
    rows_html = []
    for i, sc in enumerate(scores):
        lo, hi = bins[i], bins[i+1]
        if i == 0:
            rng = f"x < {hi}{unit}"
        elif hi is np.inf:
            rng = f"x ≥ {lo}{unit}"
        else:
            rng = f"{lo}{unit} ≤ x < {hi}{unit}"
        rows_html.append(f"<tr><td>{rng}</td><td>{sc}점</td></tr>")
    table = f"""
    <table class="table table-striped table-bordered" style="margin-bottom:0;">
      <thead><tr><th style="width:70%;">구간</th><th>점수</th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    """
    return table

def group_card_html(title, indicators):
    items = []
    for name in indicators:
        items.append(f"""
        <details>
          <summary style="cursor:pointer; font-weight:600; padding:6px 0;">{name}</summary>
          <div class="scroll-box" style="max-height:220px; overflow:auto; margin-top:6px;">
            {indicator_table_html(name)}
          </div>
        </details>
        """)
    return f"""
    <div>
      <h3 class="sub-title" style="margin-top:0;">{title}</h3>
      {' '.join(items)}
    </div>
    """

# ── 페이지 옵션/스타일 ─────────────────────────
ui.page_opts(title="🚸 어린이 보호구역 사고 대시보드", fillable=True)

ui.head_content(
    ui.tags.style(
        """
        .section-title { font-size: 1.75rem; font-weight: 800; margin: 6px 0 12px; }
        .sub-title     { font-size: 1.15rem; font-weight: 700; margin: 2px 0 10px; color: #374151; }
        .muted         { color: #6b7280; }
        .placeholder   { border: 2px dashed #cbd5e1; border-radius: 12px; padding: 22px; text-align: center; }
        .grid-gap > .col { padding: 6px !important; }

        /* 🔧 matplotlib 이미지가 카드에서 잘리지 않도록 */
        .card .recalculating img, .card img.plot-image, .shiny-plot-output img {
            max-width: 100% !important;
            height: auto !important;
            display: block;
        }
        /* 카드 패딩 살짝 줄여서 가시 면적 확대 */
        .card .card-body { padding: 12px 14px; }
        /* 인라인 입력 요소 줄 간격/여백 축소 */
        .form-check-inline { margin-right: 10px; }
        .form-label { margin-bottom: 6px; }
        """
    )
)

ui.head_content(
    ui.tags.style(
        """
        /* 이미지가 섹션 높이에 맞춰 너무 커지지 않게 */
        .shiny-plot-output img { max-width: 100%; height: auto !important; display:block; }

        /* 카드 안쪽 여백 살짝만 (가로 여백 과다 방지) */
        .card .card-body { padding-left:12px; padding-right:12px; }
        """
    )
)

ui.head_content(
    ui.tags.style(
        """
        /* 부록 2열 컬럼 스택: 각 컬럼 높이를 동일하게, 카드 2개를 반반 */
        .appendix { --appendix-col-h: 820px; } /* 전체 높이 원하는 값으로 조절 */
        .appendix .col > .appendix-col {
            height: var(--appendix-col-h);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .appendix .half-card {
            flex: 1 1 0;
            display: flex;
            flex-direction: column;
        }
        .appendix .half-card .card-body {
            display: flex;
            flex-direction: column;
            min-height: 0;      /* 내부 스크롤 허용 */
        }
        .appendix .half-card .scroll-box {
            flex: 1;
            overflow: auto;     /* 카드 안에서만 스크롤 */
        }

        /* 요약 매트릭스 표 가독성 */
        .rules-matrix thead th { text-align:center; white-space:nowrap; }
        .rules-matrix tbody td { font-size:12px; vertical-align:middle; white-space:nowrap; }
        .scroll-x { overflow-x:auto; }
        """
    )
)

ui.head_content(
    ui.tags.style(
        """
        /* 기존 equal-card는 그대로 두고, 1행에는 auto-card로 자동 높이 */
        .appendix .auto-card{height:auto;display:flex;flex-direction:column;}
        .appendix .auto-card .card-body{display:block;min-height:0;}
        .appendix .auto-card .scroll-box{overflow:visible;max-height:none;}
        """
    )
)

# ── 부록 카드 균등 높이 + 표 가독성 CSS (한 번만)
ui.head_content(
    ui.tags.style(
        """
        .appendix { --appendix-card-h: 680px; }
        .appendix .equal-card { height: var(--appendix-card-h); display:flex; flex-direction:column; }
        .appendix .equal-card .card-body { display:flex; flex-direction:column; min-height:0; }
        .appendix .equal-card .scroll-box { flex:1; overflow:auto; }

        /* 매트릭스 표(점수 헤더) 가독성 개선 + sticky */
        .rules-wrap { overflow-x:auto; position:relative; }
        table.rules-matrix { font-size:12px; }
        table.rules-matrix th, table.rules-matrix td { padding:6px 8px; text-align:center; }
        table.rules-matrix thead th {
            position:sticky; top:0; z-index:2; background:#fff;
            border-bottom:2px solid #e5e7eb;
        }
        table.rules-matrix td:first-child, table.rules-matrix th:first-child {
            position:sticky; left:0; z-index:1; background:#fff; text-align:left;
        }
        table.rules-matrix tbody tr:nth-child(odd) td { background:#f8fafc; }
        table.rules-matrix tbody tr:nth-child(even) td { background:#f1f5f9; }
        .rules-footnote { font-size:11px; color:#6b7280; margin-top:6px; }
        """
    )
)


# === 전체 선택 / 전체 해제 동작 (분석 2) ===
@reactive.effect
@reactive.event(input.quad_all)
def quad_select_all():
    xui.update_checkbox_group(
        "quad_pick",
        selected=[code_to_label[i] for i in range(8)],
    )

@reactive.effect
@reactive.event(input.quad_clear)
def quad_clear_all():
    xui.update_checkbox_group(
        "quad_pick",
        selected=[],
    )

# ── UI ─────────────────────────────────────────────────────
with ui.navset_card_pill(id="tab"):

    # 1) 개요 (신규)
    with ui.nav_panel("개요"):
        center_lat = float(df_overview["위도"].mean())
        center_lon = float(df_overview["경도"].mean())

        with ui.layout_sidebar():
            with ui.sidebar(width=220):
                ui.h2("정보", class_="section-title")

                ui.input_action_button(
                    "btn1", "어린이보호구역이란?",
                    class_="btn-primary",
                    style="font-size:13px; white-space:nowrap;"
                )
                @reactive.effect
                @reactive.event(input.btn1)
                def show_modal_btn1():
                    ui.modal_show(ui.modal(
                        ui.markdown(
                            "어린이 보호구역은 학교·유치원 등 주변 반경 300m 이내 도로에서 "
                            "어린이를 교통사고로부터 보호하기 위해 속도 제한(30km), "
                            "주·정차 금지, 안전시설 설치 등을 의무화한 특별 구역입니다."
                        ),
                        title="어린이보호구역이란",
                        easy_close=True,
                        footer=ui.modal_button("닫기")
                    ))

                ui.input_action_button(
                    "btn2", "심각성", class_="btn-info",
                    style="font-size:13px; white-space:nowrap;"
                )
                @reactive.effect
                @reactive.event(input.btn2)
                def show_modal_btn2():
                    ui.modal_show(ui.modal(
                        ui.tags.img(src="gisa.png", style="width:100%; height:auto;"),
                        title="심각성",
                        easy_close=True,
                        footer=ui.modal_button("닫기"),
                        size="l"
                    ))

                ui.input_action_button(
                    "btn3", "사례", class_="btn-success",
                    style="font-size:13px; white-space:nowrap;"
                )
                @reactive.effect
                @reactive.event(input.btn3)
                def show_modal_btn3():
                    ui.modal_show(ui.modal(
                        ui.tags.img(src="gisa1.png", style="width:100%; height:auto;"),
                        title="사례",
                        easy_close=True,
                        footer=ui.modal_button("닫기"),
                        size="l"
                    ))

            with ui.layout_columns(col_widths=(9, 3)):
                with ui.card(full_screen=True):
                    ui.card_header("어린이 보호구역 위험도 지도")
                    ui.tags.iframe(
                        src="gido.html",
                        style="width:100%; height:600px; border:0;",
                        loading="lazy"
                    )

                with ui.layout_column_wrap(width=1):
                    with ui.value_box(showcase=icon_svg("users")):
                        "보호구역 수"
                        @render.text
                        def vb_count():
                            return f"{len(df_overview):,} 개"

                    with ui.value_box(showcase=icon_svg("gauge-high")):
                        "위험도 평균"
                        @render.text
                        def vb_mean_risk():
                            return f"{df_overview['위험도점수'].mean():.2f}"

                    with ui.value_box(showcase=icon_svg("child")):
                        "대구시 어린이 인구"
                        "226,763명"


    # 5) 분석 1
    with ui.nav_panel("분석 1"):
        ui.h2("상관 관계 지표 및 인프라 평균 분석", class_="section-title")
        with ui.div(class_="panel-scroll"):
            with ui.layout_columns(class_="grid-gap"):
                with ui.card(full_screen=True):
                    ui.card_header("선택 변수 상관계수 히트맵")
                    @render.plot
                    def corr_heatmap():
                        cols = [
                            "구역지정수", "시설물 CCTV 수", "시설물 도로표지판 수", "시설물 과속방지턱 수",
                            "보호구역도로폭", "사고건수", "평균주행속도", "불법주정차위반건수",
                            "전체인구", "어린이인구", "신호등 300m", "어린이비율"
                        ]
                        num_df = df[cols].select_dtypes(include=["int64", "float64"]).copy()
                        corr = num_df.corr()
                        fig, ax = plt.subplots(figsize=(11, 7))
                        sns.heatmap(
                            corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                            cbar=True,
                            annot_kws={"size": 8},
                            xticklabels=corr.columns, yticklabels=corr.columns
                        )
                        ax.set_aspect("auto")
                        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha="right")
                        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
                        plt.tight_layout()
                        return fig

                with ui.card(full_screen=True):
                    ui.card_header("구군별 보호구역 개수")
                    @render_plotly
                    def gu_bar():
                        if "구군" not in df.columns:
                            return px.bar(title="구군 컬럼 없음")
                        counts = df["구군"].value_counts().reset_index()
                        counts.columns = ["구군", "개수"]
                        fig = px.bar(
                            counts, x="구군", y="개수",
                            text="개수",
                            color="개수",
                            color_continuous_scale="Greens",
                        )
                        fig.update_traces(textposition="outside")
                        fig.update_layout(
                            xaxis_title="구군",
                            yaxis_title="개수",
                            coloraxis_showscale=False,
                            margin=dict(t=60, r=20, l=20, b=40)
                        )
                        return fig

            with ui.layout_columns(col_widths=(6, 6), class_="grid-gap"):
                with ui.card():
                    ui.h3("인프라 (+) 요소", class_="sub-title")
                    ui.input_select(
                        id="pick_pos",
                        label="표시 지표 선택",
                        choices=list(pos_map.keys()),
                        selected="시설물 CCTV 수",
                    )
                    @render.plot
                    def infra_pos_plot_daegu():
                        key = input.pick_pos()
                        return make_bargrid(
                            [pos_map[key]],
                            ncols=1,
                            title=f"대구 구군별 인프라(+) • {key}",
                        )

                with ui.card():
                    ui.h3("인프라 (–) 요소", class_="sub-title")
                    ui.input_select(
                        id="pick_neg",
                        label="표시 지표 선택",
                        choices=list(neg_map.keys()),
                        selected="평균주행속도",
                    )
                    @render.plot
                    def infra_neg_plot_daegu():
                        key = input.pick_neg()
                        return make_bargrid(
                            [neg_map[key]],
                            ncols=1,
                            title=f"대구 구군별 인프라(–) • {key}",
                        )

    # 6) 분석 2
    with ui.nav_panel("분석 2"):
        ui.h2("대구 전체 사분면 분석 및 레이더 차트", class_="section-title")
        with ui.layout_sidebar():
            with ui.sidebar(open="closed"):
                ui.h4("사분면 표시 대상", class_="sub-title")
                ui.input_checkbox_group(
                    id="quad_pick",
                    label="표시할 구·군 선택",
                    choices=[code_to_label[i] for i in range(8)],
                    selected=[code_to_label[i] for i in range(8)],
                    inline=False,
                )
                with ui.div(style="margin-top:6px;"):
                    ui.input_action_button("quad_all", "전체 선택")
                    ui.input_action_button("quad_clear", "전체 해제")
                ui.hr()

                ui.h4("레이더 차트 비교", class_="sub-title")
                ui.input_checkbox_group(
                    id="radar_pick",
                    label="비교할 구·군 (최대 2개)",
                    choices=[code_to_label[i] for i in range(8)],
                    selected=["달서구", "북구"],
                    inline=False,
                )
                ui.input_checkbox(
                    id="radar_with_total",
                    label="대구 전체(평균) 기준선 포함",
                    value=True,
                )

            with ui.div(class_="panel-scroll"):
                with ui.layout_columns(col_widths=(6, 6), class_="grid-gap"):
                    with ui.card():
                        ui.h3("인프라 수준 vs 사고건수 사분면 분석", class_="sub-title")
                        @render.plot
                        def quad_plot_daegu():
                            picks = input.quad_pick() or []
                            if isinstance(picks, str):
                                picks = [picks]
                            return make_quadrant_fig(scale=4, names=picks)

                    with ui.card():
                        ui.h3("구군별 인프라·안전 지표 (점수, 0~100)", class_="sub-title")
                        @render.plot
                        def radar_plot_daegu():
                            picks = input.radar_pick() or []
                            if isinstance(picks, str):
                                picks = [picks]
                            names = picks[:2]
                            return make_radar_figure(
                                names,
                                include_total=input.radar_with_total(),
                                scale=1.6,
                            )



    # 7) 부록 — 표 섹션 (2×2 레이아웃, 순서: 1행 출처/점수표 · 2행 산출기준/변수설명)
    with ui.nav_panel("부록"):
        ui.h2("데이터 설명", class_="section-title")

        with ui.div(class_="appendix"):

            # ── 1행: (좌) 데이터 출처  /  (우) 지표별 점수 구간표 ──
            with ui.layout_columns(col_widths=[6, 6], class_="grid-gap"):

                # 1행 1열: 데이터 출처
                with ui.card(class_="auto-card"):
                    ui.h3("데이터 출처", class_="sub-title")

                    @render.ui
                    def show_weight_table():
                        df_w = pd.DataFrame({
                            "데이터 항목": [
                                "어린이보호구역 정보",
                                "행정동별 인구 현황",
                                "행정동별 평균 주행속도",
                                "불법주정차 단속 현황",
                                "어린이보호구역 내 교통사고 현황",
                                "도로폭"
                            ],
                            "출처": [
                                "공공데이터포털",
                                "행정안전부",
                                "공공데이터포털",
                                "공공데이터포털",
                                "교통사고정보개방시스템",
                                "국토정보플랫폼"
                            ]
                        })
                        html = df_w.to_html(escape=False, index=False,
                                            classes="table table-striped table-bordered", border=0)
                        return ui.HTML(f'<div class="scroll-box">{html}</div>')

                # 1행 2열: 지표별 점수 구간표 (열=점수, 셀=사람친화 레이블)
                with ui.card(class_="auto-card"):
                    ui.h3("인프라점수 산출 기준표", class_="sub-title")

                    @render.ui
                    def show_threshold_table():
                        units_map = {"보호구역도로폭": "m"}  # 단위 필요한 지표만
                        score_cols = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]  # 왼→오: 좋은 점수→낮은 점수

                        def fmt_num(v):
                            if v is np.inf: return "∞"
                            return str(int(v)) if abs(v - int(v)) < 1e-9 else str(v)

                        def human_range(lo, hi, unit):
                            if lo == -np.inf: return f"{fmt_num(hi)}{unit} 미만"
                            if hi is np.inf:  return f"{fmt_num(lo)}{unit} 이상"
                            a, b = fmt_num(lo), fmt_num(hi)
                            return f"{a}{unit}–{b}"

                        def exact_range(lo, hi, unit):
                            if lo == -np.inf: return f"x < {fmt_num(hi)}{unit}"
                            if hi is np.inf:  return f"x ≥ {fmt_num(lo)}{unit}"
                            return f"{fmt_num(lo)}{unit} ≤ x < {fmt_num(hi)}{unit}"

                        rows = []
                        for name, (bins, scores) in _rules_abs.items():
                            unit = units_map.get(name, "")
                            sc2cell = {}
                            for i, sc in enumerate(scores):
                                lo = bins[i] if np.isfinite(bins[i]) else -np.inf
                                hi = bins[i+1]
                                label = human_range(lo, hi, unit)
                                tip   = exact_range(lo, hi, unit)
                                sc2cell[int(sc)] = f'<span title="{tip}">{label}</span>'
                            rows.append([name] + [sc2cell.get(s, "") for s in score_cols])

                        df_matrix = pd.DataFrame(rows, columns=["지표"] + [f"{s}점" for s in score_cols])
                        html = df_matrix.to_html(
                            escape=False, index=False,
                            classes="table table-striped table-bordered rules-matrix", border=0
                        )
                        foot = "  "
                        return ui.HTML(f"<div class='scroll-box rules-wrap'>{html}</div>{foot}")

            # ── 2행: (좌) 위험도 산출 기준표  /  (우) 변수 설명 ──
            with ui.layout_columns(col_widths=[6, 6], class_="grid-gap"):

                # 2행 1열: 위험도 산출 기준표
                with ui.card(class_="equal-card"):
                    ui.h3("위험도점수 산출 기준표", class_="sub-title")

                    @render.ui
                    def show_score_table():
                        df_tbl = pd.DataFrame({
                            "변수명": [
                                "사고 건수", "평균 주행속도", "불법주정차 위반건수", "CCTV 수", "과속방지턱 수",
                                "신호등 개수", "어린이 비율", "평균 도로폭", "도로 표지판 수"
                            ],
                            "점수": ["30점","13점","10점","10점","10점","10점","7점","5점","5점"],
                            "설명": [
                                "• 과거 사고 발생이 많을수록 잠재 위험이 높음을 의미<br>• 반복 사고 구간일 가능성이 높음",
                                "• 주행속도가 높을수록 사고 위험성 증가",
                                "• 불법주정차가 많으면 시야 방해·보행자 돌발 위험 증가<br>• 사고 발생 확률을 높이는 주요 요인",
                                "• CCTV는 억제 효과가 있으나 부족 시 단속 사각지대 발생<br>• 범칙행위 및 사고 발생 감시 한계",
                                "• 과속방지턱은 속도 저감 효과<br>",
                                "• 신호등은 보행자 보호·교통 조절 기능<br>• 부족하면 횡단 중 사고 위험 증가",
                                "• 어린이 비율이 높으면 사고 발생 시 피해 심각성 증가<br>• 보호 필요성이 상대적으로 큼",
                                "• 도로폭이 좁으면 보행자와 차량 간 간격 축소<br>• 교차 및 돌발 위험이 커짐",
                                "• 도로 표지판은 운전자 주의 환기 및 규제 안내<br>• 부족하면 규칙 준수·주의 환기 미흡"
                            ]
                        })
                        html = df_tbl.to_html(escape=False, index=False,
                                              classes="table table-striped table-bordered", border=0)
                        return ui.HTML(f'<div class="scroll-box">{html}</div>')

                # 2행 2열: 변수 설명(=변수 정의)
                with ui.card(class_="equal-card"):
                    ui.h3("변수 설명", class_="sub-title")

                    @render.ui
                    def show_var_def_table():
                        df_def = pd.DataFrame({
                            "변수": [
                                "시설명", "주소", "구역지정수", "시설물 CCTV 수", "시설물 도로표지판 수",
                                "시설물 과속방지턱 수", "보호구역도로폭", "사고건수", "위도", "경도",
                                "평균주행속도", "불법주정차위반건수", "읍면동", "전체인구", "어린이인구", "신호등 300m"
                            ],
                            "정의": [
                                "해당 보호구역 시설의 이름",
                                "해당 시설의 행정 주소",
                                "지정된 보호구역의 개수",
                                "보호구역 내 또는 주변에 설치된 CCTV 수",
                                "보호구역 내 설치된 도로 표지판 수",
                                "보호구역 내 설치된 과속방지턱 수",
                                "보호구역 도로의 유효 폭원(m)",
                                "관측기간 동안 해당 구간에서 발생한 사고 건수",
                                "시설의 위도 좌표",
                                "시설의 경도 좌표",
                                "해당 구간의 평균 주행 속도(km/h)",
                                "해당 구간에서 단속된 불법 주정차 건수",
                                "해당 시설이 속한 읍/면/동 행정 구역",
                                "해당 읍/면/동 전체 주민 인구수",
                                "해당 읍/면/동 내 어린이(만 12세 이하) 인구수",
                                "시설 기준 반경 300m 내 설치된 신호등 개수"
                            ]
                        })
                        html = df_def.to_html(escape=False, index=False,
                                              classes="table table-striped table-bordered", border=0)
                        return ui.HTML(f'<div class="scroll-box">{html}</div>')
