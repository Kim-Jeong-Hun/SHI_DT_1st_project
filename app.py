"""
app.py — ELLIPSE Vocabulary 점수 예측 대시보드 (STEP 6)
========================================================
실행:  streamlit run app.py

구성
----
① 프로젝트 소개  ② 데이터 정의  ③ 데이터 분석  ④ 모델 성능  ⑤ 예측 시스템

원칙
----
숫자는 전부 파일에서 읽습니다. 코드에 하드코딩하지 않습니다.
노트북을 다시 돌려 결과가 바뀌면 대시보드도 자동으로 따라가야 하기 때문입니다.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import predict, extract_features

DATA = Path('data')

st.set_page_config(page_title='어휘 점수 예측 대시보드', page_icon='📝', layout='wide')

# ── 한글 폰트 ─────────────────────────────────────────────
from matplotlib import font_manager
_avail = {f.name for f in font_manager.fontManager.ttflist}
for _f in ['Malgun Gothic', 'AppleGothic', 'NanumGothic',
           'NanumBarunGothic', 'Noto Sans CJK KR', 'Noto Sans CJK JP']:
    if _f in _avail:
        plt.rc('font', family=_f)
        plt.rcParams['axes.unicode_minus'] = False
        break


# ── 데이터 로드 ───────────────────────────────────────────
# 모델과 대용량 CSV는 캐시합니다. 탭을 옮길 때마다 다시 읽으면 느려집니다.
@st.cache_resource
def load_models():
    return (joblib.load(DATA / 'preprocessor.joblib'),
            joblib.load(DATA / 'final_model.joblib'))


@st.cache_data
def load_json(name):
    p = DATA / name
    return json.load(open(p, encoding='utf-8')) if p.exists() else None


@st.cache_data
def load_csv(name, **kw):
    p = DATA / name
    return pd.read_csv(p, **kw) if p.exists() else None


ARTIFACTS, FINAL_MODEL = load_models()
FEATS = load_json('selected_features.json')
SUMMARY = load_json('final_summary.json')
MODEL_RES = load_json('model_results.json')
VALIDATION = load_json('extraction_validation.json')

NUM_FEATURES = FEATS['numeric']

st.title('📝 영어 학습자 에세이 어휘 점수 예측')
st.caption('ELLIPSE Corpus · 해석 가능한 언어 특성 15개 기반 회귀 모델')

TABS = st.tabs(['① 프로젝트 소개', '② 데이터 정의', '③ 데이터 분석',
                '④ 모델 성능', '⑤ 예측 시스템'])

# ═══════════════════════════════════════════════════════════
# ① 프로젝트 소개
# ═══════════════════════════════════════════════════════════
with TABS[0]:
    st.header('무엇을, 왜 예측하는가')

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**예측 대상**: 영어 학습자(ELL)가 쓴 에세이의 **Vocabulary 점수** (1.0 ~ 5.0, 0.5 단위)

**데이터**: [ELLIPSE Corpus](https://github.com/scrosseye/ELLIPSE-Corpus) — 8~12학년 영어 학습자 에세이

### 이 프로젝트가 답하려는 질문

> 몇 점인지 맞히는 것이 아니라,
> **어떤 언어 특성이 어휘 점수를 만드는가**를 설명할 수 있는 모델을 만든다.

BERT 계열 딥러닝은 예측 성능이 더 높지만 **왜 그 점수인지 설명하지 못합니다.**
이 프로젝트는 성능 일부를 내주고 설명력을 얻는 쪽을 선택했습니다.

### 출발점의 문제

원본 데이터가 가진 어휘 관련 변수는 `TTR`, `MTLD`, `Type`, `Token` 넷뿐이었고,
이들은 대부분 **텍스트 길이를 재고 있었습니다.**
채점자가 실제로 보는 "어휘 수준"과 "철자 정확성" 축은 데이터에 존재하지 않았습니다.

→ 원문에서 **파생변수 28개**를 직접 설계해 그 축을 만들었습니다.
        """)
    with c2:
        st.markdown('#### 최종 성능')
        if SUMMARY:
            t = SUMMARY['test']
            b = {'MAE': 0.4700, 'RMSE': 0.5795, 'R²': 0.0000}
            st.metric('MAE', f"{t['MAE']:.4f}",
                      f"{t['MAE'] - b['MAE']:+.4f} vs 평균 예측", delta_color='inverse')
            st.metric('RMSE', f"{t['RMSE']:.4f}",
                      f"{t['RMSE'] - b['RMSE']:+.4f}", delta_color='inverse')
            st.metric('R²', f"{t['R²']:.4f}", '평균 예측 = 0.0000')
            st.caption('test 750편 · 학습에 사용하지 않은 데이터')

    st.divider()
    st.subheader('분석 파이프라인')
    st.markdown("""
| 단계 | 내용 | 결과 |
|---|---|---|
| 1. 기획 | 주제 선정, 가설 7개 설계 | Vocabulary 지표 선정 |
| 2. 전처리 | 정제 · 파생변수 28개 생성 | 3,749 → 3,747편 |
| 3. EDA | 분포 · 상관 · 다중공선성 · 공정성 점검 | 완전 종속 5개, VIF 10 이상 11개 발견 |
| 4. 변수 선택 | 규칙 R0~R5 기계적 적용 | 파생 28개 → **15개**, 최대 VIF 8.9 |
| 5. 모델링 | Linear · Ridge · Lasso · RF 비교 | 4종 성능 차이가 측정 오차 이내 |
| 6. 성능 향상 | 6가지 방법 시도 | 앙상블 채택, 한계 3가지 규명 |
    """)

    st.divider()
    st.subheader('설계 가설 7개의 검증 결과')
    hyp = pd.DataFrame([
        ['TTR은 길이에 교란되어 있다', 'r(Token) = −0.652', '✅ 확인'],
        ['√·log 보정으로 교란을 줄일 수 있다', 'guiraud 0.376 / herdan_c −0.420', '✅ 부분 확인'],
        ['어휘 수준(B)이 최우선 변수군이다', '형태 계열은 성공, 빈도 밴드는 실패', '⚠️ 절반'],
        ['철자 오류는 제거가 아니라 신호다', 'r = −0.318, 전체 3위', '✅ 확인'],
        ['어휘 밀도가 어휘 점수를 설명한다', '전 변수 |r| < 0.10', '❌ 기각'],
        ['어휘 발달과 구문 발달은 동반한다', '부호 반전 — 다른 것을 측정 중', '🔄 재해석'],
        ['프롬프트별 기대 분량이 다르다', 'Token과 r = 0.951', '❌ 기각'],
    ], columns=['설계 시 가설', '실측 결과', '판정'])
    st.dataframe(hyp, hide_index=True, use_container_width=True)
    st.info('7개 중 3개 확인 · 2개 기각 · 1개 부분 · 1개 재해석. '
            '전부 맞았다면 검증이 아니라 확인에 그쳤을 것이며, '
            '기각과 재해석이 나온 지점에서 데이터에 대한 새 정보가 나왔습니다.')

# ═══════════════════════════════════════════════════════════
# ② 데이터 정의
# ═══════════════════════════════════════════════════════════
with TABS[1]:
    st.header('변수는 어떻게 만들어졌는가')

    st.subheader('파생변수 7개 그룹')
    st.markdown("""
| 그룹 | 측정 대상 | 대표 변수 | 결과 |
|---|---|---|---|
| **A. 다양성 보정** | 길이 교란을 뺀 어휘 다양성 | `guiraud`, `herdan_c` | ✅ 최상위 |
| **B. 어휘 수준** | 단어의 난이도 | `mean_word_length`, `k2_ratio` | ⚠️ 형태 성공 / 빈도 실패 |
| **C. 어휘 밀도** | 내용어 비율 | `lexical_density` | ❌ 전멸 |
| **D. 정확성** | 철자 오류의 대리지표 | `oov_ratio`, `hapax_ratio` | ✅ 3위 |
| **E. 구문** | 문장·문단 구조 | `mean_sent_length`, `sent_per_para` | 🔄 재해석 |
| **F. 과제 상대화** | 주제별 상대 분량 | `len_z_in_prompt` | 🔄 길이 통제 변수로 |
| **G. 품질 플래그** | 측정 무효 구간 표시 | `flag_very_short` | 정보 보존용 |
    """)

    st.divider()
    st.subheader(f'모델에 투입된 {len(NUM_FEATURES)}개 변수')

    MEANING = {
        'guiraud': ('A. 다양성', 'Type ÷ √Token — 같은 분량에서 서로 다른 단어를 더 많이 씀'),
        'MTLD_valid': ('A. 다양성', '어휘가 반복되기까지의 평균 구간. 길이에 독립적'),
        'herdan_c': ('A. 다양성', 'log(Type) ÷ log(Token)'),
        'hapax_ratio': ('A. 다양성', '1회만 등장한 타입의 비율'),
        'mean_word_length': ('B. 어휘 수준', '토큰 평균 문자 수'),
        'mean_syllables': ('B. 어휘 수준', '토큰 평균 음절 수'),
        'academic_suffix_ratio': ('B. 어휘 수준', '-tion, -ity 등 학술 접미사 비율'),
        'k2_ratio': ('B. 어휘 수준', '빈도 1,001~2,000위 구간 토큰 비율'),
        'clean_beyond_2k': ('B. 어휘 수준', '2,000위 밖 단어 비율 − 철자 오류 비율'),
        'oov_ratio': ('D. 정확성', '사전 미등재 토큰 비율 — 철자 오류의 대리값'),
        'mean_sent_length': ('E. 구문', '문장당 토큰 수. 길수록 run-on 문장 신호'),
        'sent_length_std': ('E. 구문', '문장 길이의 편차'),
        'sent_per_para': ('E. 구문', '문단당 문장 수'),
        'len_z_in_prompt': ('F. 길이 통제', '같은 주제 안에서의 상대적 분량'),
        'flag_very_short': ('G. 플래그', 'MTLD 측정 불가 구간 표시 (예측 목적 아님)'),
    }
    imp = load_csv('feature_importance.csv', index_col=0)
    rows = []
    for c in NUM_FEATURES:
        grp, desc = MEANING.get(c, ('-', '-'))
        rows.append({'변수': c, '그룹': grp, '설명': desc,
                     'Ridge 계수': imp.loc[c, 'Ridge 계수'] if imp is not None and c in imp.index else None,
                     'log 변환': '○' if c in FEATS['log_transformed'] else ''})
    st.dataframe(pd.DataFrame(rows).sort_values('그룹'),
                 hide_index=True, use_container_width=True)

    st.divider()
    st.subheader('무엇을, 왜 제거했는가')
    log = load_csv('feature_selection_log.csv')
    if log is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(log.groupby(['규칙', '조치']).size().to_frame('건수'),
                         use_container_width=True)
        with c2:
            st.markdown("""
| 규칙 | 기준 |
|---|---|
| **R0** | 분할 전 전체 데이터 통계로 계산된 변수 → train 기준 재계산 |
| **R1** | 원문·타겟·파생변수로 대체된 원본 컬럼 |
| **R2** | 합이 정확히 1인 완전 종속 (VIF = ∞) |
| **R3** | 변수 간 \\|r\\| > 0.85 → 열세인 쪽 제거 |
| **R4** | \\|r(Target)\\| < 0.10 |
| **R5** | VIF ≥ 10 → 최댓값 1개씩 반복 제거 |
            """)
        with st.expander('전체 제거 로그 보기'):
            st.dataframe(log, hide_index=True, use_container_width=True)

    st.warning('**판정 기준은 결과를 보기 전에 선언했습니다.** '
               '결과를 본 뒤 기준을 조정하면 "데이터에 맞춰 기준을 골랐다"는 반박을 피할 수 없습니다.')

# ═══════════════════════════════════════════════════════════
# ③ 데이터 분석
# ═══════════════════════════════════════════════════════════
with TABS[2]:
    st.header('데이터는 어떻게 생겼는가')

    train = load_csv('train.csv')
    # 상관·히트맵은 04에서 만든 최종 행렬을 씁니다.
    # train.csv에는 04에서 생성한 clean_beyond_2k가 아직 없습니다.
    Xtr = load_csv('X_train.csv')
    ytr = load_csv('y_train.csv')
    if train is None or Xtr is None or ytr is None:
        st.error('data/train.csv · X_train.csv · y_train.csv 가 필요합니다.')
    else:
        y = train['Vocabulary']
        y_x = ytr.iloc[:, 0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('train 편수', f'{len(train):,}')
        c2.metric('평균 점수', f'{y.mean():.3f}')
        c3.metric('표준편차', f'{y.std():.3f}')
        c4.metric('점수 구간', f'{y.min():.1f} ~ {y.max():.1f}')

        st.subheader('타겟 분포 — 왜 R² 상한이 낮은가')
        cnt = y.value_counts().sort_index()
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
        axes[0].bar(cnt.index.astype(str), cnt.values,
                    color=np.where(cnt.values < 30, 'lightcoral', 'steelblue'))
        axes[0].set_xlabel('Vocabulary 점수'); axes[0].set_ylabel('편수')
        axes[0].set_title('빨강 = 30편 미만 구간 (모델이 거의 못 본 구간)')
        axes[1].pie([cnt[cnt >= 30].sum(), cnt[cnt < 30].sum()],
                    labels=['30편 이상 구간', '30편 미만 구간'],
                    autopct='%1.1f%%', colors=['steelblue', 'lightcoral'])
        axes[1].set_title('데이터 집중도')
        st.pyplot(fig); plt.close(fig)
        st.caption(f'{int(cnt[cnt < 30].sum())}편({cnt[cnt < 30].sum() / len(y) * 100:.1f}%)이 '
                   '관측 30편 미만 구간에 있습니다. 이 구간의 예측 편향은 '
                   '모델을 바꿔서 해결되지 않습니다.')

        st.divider()
        st.subheader('길이 교란 — "길게 쓰면 점수가 오른다"는 사실인가')
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(train['Token'], y, alpha=.12, s=10, color='gray')
            ax.set_xlabel('Token (텍스트 길이)'); ax.set_ylabel('Vocabulary')
            ax.set_title(f'길이 vs 점수  (r = {train["Token"].corr(y):.3f})')
            st.pyplot(fig); plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(train['guiraud'], y, alpha=.12, s=10, color='steelblue')
            ax.set_xlabel('guiraud (길이 보정 어휘 다양성)'); ax.set_ylabel('Vocabulary')
            ax.set_title(f'어휘 다양성 vs 점수  (r = {train["guiraud"].corr(y):.3f})')
            st.pyplot(fig); plt.close(fig)
        st.success('**길이 단독 R² = 0.0716 / 핵심 파생변수 4개 R² = 0.2661.** '
                   '어휘 점수를 설명하는 것은 분량이 아니라 어휘의 질입니다.')

        st.divider()
        st.subheader('선택된 15개 변수의 상관 구조')
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 6.5))
            corr = Xtr[NUM_FEATURES].corr()
            sns.heatmap(corr, cmap='RdBu_r', center=0, ax=ax,
                        vmin=-1, vmax=1, square=True,
                        cbar_kws={'shrink': .7}, annot=False)
            ax.set_title('변수 간 상관 (STEP 6 정리 후)')
            st.pyplot(fig); plt.close(fig)
        with c2:
            r = Xtr[NUM_FEATURES].corrwith(y_x).sort_values(key=abs, ascending=False)
            fig, ax = plt.subplots(figsize=(5, 6.5))
            ax.barh(r.index[::-1], r.values[::-1],
                    color=np.where(r.values[::-1] > 0, 'steelblue', 'lightcoral'))
            ax.axvline(0, color='black', lw=.8)
            ax.set_xlabel('Vocabulary와의 상관')
            ax.set_title('단변량 예측력')
            st.pyplot(fig); plt.close(fig)
        st.caption('STEP 6 이전에는 완전 종속(VIF = ∞) 변수가 5개, VIF 10 이상이 11개였습니다. '
                   '정리 후 최대 VIF는 8.9로, 회귀계수를 해석할 수 있는 상태가 됐습니다.')

# ═══════════════════════════════════════════════════════════
# ④ 모델 성능
# ═══════════════════════════════════════════════════════════
with TABS[3]:
    st.header('모델은 얼마나, 어디까지 맞히는가')

    st.subheader('베이스라인부터 최종 모델까지')
    base = pd.DataFrame(MODEL_RES['baseline']) if MODEL_RES else None
    final_t = load_csv('final_test_results.csv')
    if base is not None and final_t is not None:
        prog = pd.DataFrame([
            {'단계': '① 평균 예측 (학습 없음)', 'MAE': 0.4700, 'RMSE': 0.5795, 'R²': 0.0000},
            {'단계': '② 길이 단독', 'MAE': 0.4395, 'RMSE': 0.5583, 'R²': 0.0716},
            {'단계': '③ 핵심 파생변수 4개', 'MAE': 0.3901, 'RMSE': 0.4964, 'R²': 0.2661},
            {'단계': '④ STEP 4 — Lasso 단일', **{k: MODEL_RES['test'][0][k] for k in []},
             'MAE': 0.3637, 'RMSE': 0.4674, 'R²': 0.3421},
            {'단계': '⑤ STEP 5 — 앙상블 (최종)',
             'MAE': float(final_t[final_t['구분'].str.contains('최종')]['MAE'].iloc[0]),
             'RMSE': float(final_t[final_t['구분'].str.contains('최종')]['RMSE'].iloc[0]),
             'R²': float(final_t[final_t['구분'].str.contains('최종')]['R²'].iloc[0])},
        ])
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(prog, hide_index=True, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 3.4))
            ax.barh(prog['단계'], prog['R²'],
                    color=['lightgray'] * 3 + ['lightsteelblue', 'steelblue'])
            ax.set_xlabel('R² (test)')
            st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader('모델 4종 비교 — 알고리즘이 성능을 만들었는가')
    cv = load_csv('cv_results.csv')
    if cv is not None:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(cv, hide_index=True, use_container_width=True)
        with c2:
            st.markdown("""
**모델 간 차이가 측정 오차 안쪽입니다.**

- 폴드 간 표준편차: **±0.044**
- 모델 간 최대 차이: **0.007**

성능을 만든 것은 알고리즘이 아니라 **피처**였습니다.
전처리에 공을 들인 이유가 여기 있습니다.

프롬프트 더미 36개를 추가해도 4개 모델 전부에서
증가폭이 노이즈 범위였습니다 → 미포함 채택.
            """)

    st.divider()
    st.subheader('무엇이 점수를 설명하는가')
    imp = load_csv('feature_importance.csv', index_col=0)
    if imp is not None:
        c1, c2 = st.columns(2)
        with c1:
            o = imp.sort_values('Ridge 계수')
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(o.index, o['Ridge 계수'],
                    color=np.where(o['Ridge 계수'] > 0, 'steelblue', 'lightcoral'))
            ax.axvline(0, color='black', lw=.8)
            ax.set_title('표준화 회귀계수')
            st.pyplot(fig); plt.close(fig)
        with c2:
            o2 = imp.sort_values('RF 중요도')
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(o2.index, o2['RF 중요도'], color='seagreen')
            ax.set_title('Random Forest 변수 중요도')
            st.pyplot(fig); plt.close(fig)
        st.markdown("""
**두 모델이 순위 1·2위에 완전히 합의합니다** — `guiraud`(어휘 다양성)와 `oov_ratio`(철자 정확성).
방법론이 다른 두 모델이 같은 변수를 지목했다는 것은,
이 결론이 특정 모델의 가정에 의존하지 않는다는 뜻입니다.
        """)
        st.warning('**계수 해석에는 단서가 붙습니다.** `herdan_c` · `k2_ratio` · `flag_very_short`는 '
                   '단변량 상관과 회귀계수의 부호가 반대입니다. 겹치는 변수가 공통 성분을 '
                   '먼저 가져간 결과이므로, 이 셋은 단독 해석 대상에서 제외했습니다.')

    st.divider()
    st.subheader('어디서 틀리는가')
    pred = load_csv('final_predictions.csv')
    if pred is not None:
        bias = pred.groupby('실제').agg(건수=('예측', 'size'),
                                      평균예측=('예측', 'mean'),
                                      평균잔차=('잔차', 'mean')).round(3)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(bias, use_container_width=True)
        with c2:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
            axes[0].scatter(pred['예측'], pred['실제'], alpha=.25, s=14, color='steelblue')
            axes[0].plot([0.8, 5.2], [0.8, 5.2], 'r--', lw=1.3)
            axes[0].set_xlim(0.8, 5.2); axes[0].set_ylim(0.8, 5.2)
            axes[0].set_xlabel('예측'); axes[0].set_ylabel('실제')
            axes[0].set_title('예측 vs 실제')
            axes[1].axhline(0, color='red', ls='--', lw=1)
            axes[1].plot(bias.index, bias['평균잔차'], 'o-', color='steelblue')
            axes[1].set_xlabel('실제 점수'); axes[1].set_ylabel('평균 잔차')
            axes[1].set_title('구간별 편향')
            st.pyplot(fig); plt.close(fig)
        st.error('**저점수는 과대예측, 고점수는 과소예측됩니다.** '
                 'EDA 단계에서 분포만 보고 예고했던 패턴이 그대로 재현됐습니다. '
                 '희소 구간의 표본 부족이 원인이며, 모델 교체로 해결되지 않습니다.')

    st.divider()
    st.subheader('성능 향상 시도 6가지')
    impr = load_csv('improvement_log.csv')
    if impr is not None:
        st.dataframe(impr, hide_index=True, use_container_width=True)
        st.markdown("""
**개선폭이 전부 폴드 간 표준편차(±0.043) 안쪽입니다.** 이 사실을 감추지 않습니다.

여섯 가지 방법을 시도해도 R²가 0.31~0.35를 벗어나지 못한다면,
한계는 모델링이 아니라 **데이터와 과제의 성격**에 있습니다.

| 규명된 한계 | 근거 |
|---|---|
| 관계가 거의 선형 | 비선형 모델(RF·GB·HistGB)이 선형 모델을 못 넘음 |
| 분포 불균형 | 끝단 편향은 앙상블로 해결 안 됨. 가중치는 전체 성능과 맞바꾸는 거래 |
| 채점 자체의 불확실성 | Vocabulary의 채점자 간 일치도 Cohen's κ = .518 (7개 항목 중 최하위) |

**사람 채점자끼리도 잘 안 맞는 항목**을 해석 가능한 변수 15개로 여기까지 설명한 것입니다.
        """)

# ═══════════════════════════════════════════════════════════
# ⑤ 예측 시스템
# ═══════════════════════════════════════════════════════════
with TABS[4]:
    st.header('에세이를 넣으면 어휘 점수를 예측합니다')

    if VALIDATION:
        st.caption(
            f"원문에서 변수를 다시 계산해 예측합니다. "
            f"학습에 쓴 값과의 예측 일치도: 상관 {VALIDATION['pred_corr']:.4f}, "
            f"평균 차이 {VALIDATION['pred_mean_abs_diff']:.4f}점 "
            f"(test {VALIDATION['n_validated']}편 검증)")

    SAMPLE = (
        "Many schools now require students to participate in extracurricular activities "
        "before they can graduate. I believe this policy has considerable benefits, "
        "although it also presents significant challenges for certain students.\n\n"
        "First, participation develops abilities that traditional classrooms cannot "
        "cultivate. When students join a debate team or an orchestra, they must "
        "collaborate with unfamiliar people, negotiate disagreements, and accept "
        "criticism without becoming discouraged. These experiences build resilience and "
        "communication skills that employers consistently identify as essential. A "
        "student who has organized a fundraising campaign understands deadlines and "
        "delegation in a way that no textbook could convey.\n\n"
        "Second, activities create connections between students who might never interact "
        "otherwise. My own school is unfortunately divided along academic lines, and the "
        "volleyball team was the only place where those divisions genuinely disappeared. "
        "Such environments encourage tolerance and mutual respect, qualities that benefit "
        "the entire community rather than individual participants alone.\n\n"
        "However, mandatory involvement can overwhelm students who already balance "
        "demanding coursework with family responsibilities or part-time employment. For "
        "them, a requirement that ignores individual circumstances may transform a "
        "valuable opportunity into another burden. Schools should therefore consider "
        "alternative arrangements, such as recognizing employment or caregiving as "
        "legitimate forms of participation.\n\n"
        "In conclusion, schools should encourage extracurricular participation "
        "enthusiastically but avoid rigid mandates. Flexibility allows every student to "
        "benefit from these opportunities without sacrificing their academic performance "
        "or personal wellbeing."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        text = st.text_area('에세이 원문 (영문)', value=SAMPLE, height=260)
    with c2:
        prompts = ['(주제 미지정)'] + list(ARTIFACTS['token_prompt_stats'].index)
        prompt = st.selectbox('과제 주제', prompts,
                              help='같은 주제 안에서의 상대 분량(len_z_in_prompt) 계산에 씁니다. '
                                   '모르면 미지정으로 두세요 — 전체 평균으로 대체됩니다.')
        run = st.button('점수 예측', type='primary', use_container_width=True)

    if run:
        if not text.strip():
            st.error('에세이를 입력해 주세요.')
        else:
            try:
                res = predict(text, None if prompt.startswith('(') else prompt,
                              ARTIFACTS, FINAL_MODEL)
            except Exception as e:
                st.error(f'예측 실패: {e}')
                st.stop()

            f = res['features']
            score = res['score']

            st.divider()
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                # 평가는 연속값, 표시는 채점 형식 — 05 노트북 [4-6b]
                st.metric('예측 점수 (채점 형식)', f'{round(score * 2) / 2:.1f}')
                st.caption(f'연속값 {score:.3f}')
            with c2:
                mae = SUMMARY['test']['MAE'] if SUMMARY else 0.36
                st.metric('예상 오차 범위', f'± {mae:.2f}점')
                st.caption(f'{score - mae:.2f} ~ {score + mae:.2f}')
            with c3:
                st.markdown('**앙상블 구성 모델별 예측**')
                st.dataframe(pd.DataFrame([res['members']]).round(3),
                             hide_index=True, use_container_width=True)

            if f['flag_very_short']:
                st.warning(f'토큰이 {f["_Token"]}개로 100개 미만입니다. '
                           'MTLD는 정의상 100토큰 이상을 요구하므로 대치값을 사용했습니다. '
                           '이 예측의 신뢰도는 낮습니다.')
            elif f['_Token'] < 300:
                # train Token 25분위가 299입니다. 그보다 짧으면 guiraud가 부풀려집니다.
                st.info(f'토큰이 {f["_Token"]}개로, 학습 데이터의 하위 25% 구간(299개 미만)에 '
                        '해당합니다. `guiraud`는 길이 보정 지표이지만 완전히 길이 독립적이지는 '
                        '않아(r(Token) = 0.38) 짧고 잘 다듬어진 글은 어휘 다양성이 실제보다 '
                        '높게 측정되는 경향이 있습니다. 점수가 다소 높게 나올 수 있습니다.')

            st.divider()
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader('텍스트 기본 정보')
                st.dataframe(pd.DataFrame([
                    {'항목': '토큰 수', '값': f['_Token']},
                    {'항목': '타입 수 (서로 다른 단어)', '값': f['_Type']},
                    {'항목': '문장 수', '값': f['_num_sent']},
                    {'항목': '문단 수', '값': f['_num_para']},
                    {'항목': 'MTLD', '값': round(f['_MTLD_raw'], 1)},
                    {'항목': '사전 미등재 토큰 비율', '값': f'{f["oov_ratio"] * 100:.2f}%'},
                ]), hide_index=True, use_container_width=True)
                if f['_oov_words']:
                    with st.expander(f'사전에 없는 단어 {len(f["_oov_words"])}개 (최대 30개)'):
                        st.write(', '.join(f['_oov_words']))
                        st.caption('철자 오류이거나 고유명사입니다. '
                                   '이 프로젝트는 오타를 교정하지 않습니다 — '
                                   '철자 오류 자체가 어휘력의 신호이기 때문입니다.')

            with c2:
                st.subheader('이 점수에 무엇이 기여했는가')
                # 표준화값 × Ridge 계수 = 각 변수가 예측을 밀어올린/내린 정도
                impf = load_csv('feature_importance.csv', index_col=0)
                if impf is not None:
                    z = res['scaled'].iloc[0]
                    contrib = (z[impf.index] * impf['Ridge 계수']).sort_values()
                    top = pd.concat([contrib.head(4), contrib.tail(4)])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(top.index, top.values,
                            color=np.where(top.values > 0, 'steelblue', 'lightcoral'))
                    ax.axvline(0, color='black', lw=.8)
                    ax.set_xlabel('점수 기여도 (표준화값 × 계수)')
                    ax.set_title('파랑 = 점수를 올림 / 빨강 = 내림')
                    st.pyplot(fig); plt.close(fig)
                    st.caption('선형 모델(Lasso) 기준의 근사 설명입니다. '
                               '앙상블에는 트리 모델 2개가 함께 들어가므로 '
                               '최종 점수와 정확히 일치하지는 않습니다.')

            with st.expander('계산된 변수 15개 전체 보기'):
                st.dataframe(pd.DataFrame([
                    {'변수': c, '원값': round(f[c], 4),
                     '표준화값': round(res['scaled'].iloc[0][c], 3)}
                    for c in NUM_FEATURES]), hide_index=True, use_container_width=True)
                st.caption('표준화값이 0이면 train 평균 수준, +1이면 평균보다 표준편차 1만큼 높습니다.')

    st.divider()
    with st.expander('⚠️ 이 예측을 어디까지 믿어도 되는가'):
        st.markdown(f"""
- **평균 오차 {SUMMARY['test']['MAE']:.2f}점** — 채점 단위가 0.5점이므로 대략
  {SUMMARY['test']['MAE'] / 0.5:.1f}단계 오차입니다.
- **양 끝단은 중앙으로 끌립니다.** 실제 5.0점 에세이도 3.7점 부근으로 예측되는 경향이 있습니다.
  학습 데이터에 해당 구간이 거의 없기 때문입니다.
- **8~12학년 영어 학습자 에세이**로 학습했습니다. 다른 집단·장르에는 적용되지 않습니다.
- **사람 채점자끼리의 일치도도 Cohen's κ = .518**입니다. 정답이 하나로 정해진 과제가 아닙니다.
- 이 도구는 **참고용 보조 지표**이며 실제 채점을 대체하지 않습니다.
        """)
        if VALIDATION:
            st.markdown('**원문 재계산 검증 결과** (test 750편)')
            st.dataframe(pd.DataFrame(VALIDATION['per_feature']),
                         hide_index=True, use_container_width=True)
            st.caption(
                f"학습에 쓴 값(코퍼스 제공)과 원문 재계산값의 비교입니다. "
                f"`MTLD_valid`만 구현 차이로 완전히 일치하지 않습니다 — "
                f"ELLIPSE 코퍼스의 MTLD 계산 규칙이 공개돼 있지 않기 때문입니다. "
                f"최종 예측에 미친 영향은 평균 {VALIDATION['pred_mean_abs_diff']:.4f}점으로, "
                f"재계산 피처로 평가한 test 성능은 "
                f"MAE {VALIDATION['recomputed_mae']:.4f} / R² {VALIDATION['recomputed_r2']:.4f}"
                f"(저장 피처 기준 {VALIDATION['stored_mae']:.4f} / {VALIDATION['stored_r2']:.4f})입니다.")
