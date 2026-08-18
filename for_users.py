"""
for_users.py — 내 영어 글, 어휘 점수는 몇 점일까?
==================================================
실행:  python -m streamlit run for_users.py

app.py가 "분석 과정을 검증하는" 화면이라면,
이 앱은 "글을 쓰는 사람이 결과를 바로 이해하는" 화면입니다.

설계 원칙
--------
1. 기본 화면에는 전문 용어를 쓰지 않는다. 필요한 사람만 펼쳐 보게 한다.
2. 표보다 그래프. 정적 그래프보다 만질 수 있는 그래프.
3. 첫 화면에서 바로 무언가를 해볼 수 있어야 한다.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from feature_extraction import predict, extract_features

DATA = Path('data')

st.set_page_config(page_title='어휘 점수 진단', page_icon='✍️',
                   layout='wide', initial_sidebar_state='expanded')

# ── 스타일 ────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {padding-top: 2.2rem; max-width: 1250px;}
    [data-testid="stMetricValue"] {font-size: 2.1rem;}
    .big-score {font-size: 4.5rem; font-weight: 700; line-height: 1;
                text-align: center; margin: 0;}
    .score-label {text-align: center; color: #888; font-size: 0.9rem;
                  letter-spacing: .05em;}
    .pill {display:inline-block; padding: .18rem .7rem; border-radius: 999px;
           font-size: .78rem; margin-right: .3rem; background:#eef2f7; color:#31435c;}
    .card {border:1px solid #e6e9ef; border-radius:14px; padding:1.1rem 1.3rem;
           background:#fbfcfe;}
</style>
""", unsafe_allow_html=True)

PALETTE = ['#4C7BD9', '#E8734A', '#4FB286', '#B07FD6', '#E0B341']
BLUE, ORANGE, GREEN = PALETTE[0], PALETTE[1], PALETTE[2]


# ── 로딩 ──────────────────────────────────────────────────
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
NUM_FEATURES = FEATS['numeric']
MAE = SUMMARY['test']['MAE'] if SUMMARY else 0.36

# 사람이 읽는 이름 — 화면에는 이것만 보이게 합니다.
LABEL = {
    'guiraud': '어휘 다양성',
    'MTLD_valid': '어휘 지속력',
    'herdan_c': '어휘 다양성(로그)',
    'hapax_ratio': '한 번만 쓴 단어',
    'oov_ratio': '철자 정확성',
    'mean_word_length': '단어 길이',
    'mean_syllables': '단어 음절 수',
    'academic_suffix_ratio': '학술 어휘',
    'k2_ratio': '중급 어휘',
    'clean_beyond_2k': '고급 어휘',
    'mean_sent_length': '문장 길이',
    'sent_length_std': '문장 길이 변화',
    'sent_per_para': '문단 구성',
    'len_z_in_prompt': '글 분량',
    'flag_very_short': '짧은 글 표시',
}
# 방향: 값이 클수록 점수에 유리하면 True
HIGHER_BETTER = {
    'guiraud': True, 'MTLD_valid': True, 'hapax_ratio': True,
    'oov_ratio': False, 'mean_word_length': True, 'mean_syllables': True,
    'academic_suffix_ratio': True, 'clean_beyond_2k': True,
    'mean_sent_length': False, 'sent_length_std': False,
    'sent_per_para': True, 'len_z_in_prompt': True,
    'herdan_c': True, 'k2_ratio': True, 'flag_very_short': False,
}
TIP = {
    'guiraud': '같은 분량에서 서로 다른 단어를 얼마나 많이 썼는지',
    'MTLD_valid': '같은 단어를 다시 꺼내 쓰기까지 얼마나 오래 버티는지',
    'oov_ratio': '사전에 없는 단어(철자 오류·고유명사)의 비율',
    'clean_beyond_2k': '흔하지 않은 단어를 얼마나 썼는지 (오타는 뺀 값)',
    'mean_word_length': '쓴 단어들의 평균 글자 수',
    'mean_syllables': '쓴 단어들의 평균 음절 수',
    'academic_suffix_ratio': '-tion, -ity 처럼 학술적인 어미를 가진 단어 비율',
    'mean_sent_length': '문장 하나가 평균 몇 단어인지 (너무 길면 감점 방향)',
    'sent_length_std': '문장 길이가 들쭉날쭉한 정도',
    'sent_per_para': '문단 하나에 문장이 몇 개인지',
    'len_z_in_prompt': '같은 주제를 쓴 사람들과 비교한 분량',
    'hapax_ratio': '글 안에서 딱 한 번만 등장한 단어의 비율',
    'k2_ratio': '아주 흔하지도 아주 어렵지도 않은 중간 난이도 단어 비율',
    'herdan_c': '어휘 다양성을 로그로 보정한 값',
    'flag_very_short': '글이 100단어보다 짧으면 켜지는 표시',
}

SAMPLE = (
    "Many schools now require students to participate in extracurricular activities "
    "before they can graduate. I believe this policy has considerable benefits, "
    "although it also presents significant challenges for certain students.\n\n"
    "First, participation develops abilities that traditional classrooms cannot "
    "cultivate. When students join a debate team or an orchestra, they must collaborate "
    "with unfamiliar people, negotiate disagreements, and accept criticism without "
    "becoming discouraged. These experiences build resilience and communication skills "
    "that employers consistently identify as essential. A student who has organized a "
    "fundraising campaign understands deadlines and delegation in a way that no textbook "
    "could convey.\n\n"
    "Second, activities create connections between students who might never interact "
    "otherwise. My own school is unfortunately divided along academic lines, and the "
    "volleyball team was the only place where those divisions genuinely disappeared. "
    "Such environments encourage tolerance and mutual respect, qualities that benefit "
    "the entire community rather than individual participants alone.\n\n"
    "However, mandatory involvement can overwhelm students who already balance demanding "
    "coursework with family responsibilities or part-time employment. For them, a "
    "requirement that ignores individual circumstances may transform a valuable "
    "opportunity into another burden. Schools should therefore consider alternative "
    "arrangements, such as recognizing employment or caregiving as legitimate forms of "
    "participation.\n\n"
    "In conclusion, schools should encourage extracurricular participation "
    "enthusiastically but avoid rigid mandates. Flexibility allows every student to "
    "benefit from these opportunities without sacrificing their academic performance or "
    "personal wellbeing."
)

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('## ✍️ 어휘 점수 진단')
    st.caption('영어 에세이의 어휘 수준을 언어 특성으로 설명합니다')
    page = st.radio('메뉴', [
        '✏️  내 글 진단하기',
        '🔍  무엇이 점수를 만드나',
        '🌐  데이터 둘러보기',
        '📊  얼마나 믿을 만한가',
        'ℹ️  이 프로젝트는',
    ], label_visibility='collapsed')
    st.divider()
    st.metric('평균 오차', f'±{MAE:.2f}점')
    st.caption('1.0 ~ 5.0점 척도 · 0.5점 단위 채점')


# ── 공통 위젯 ─────────────────────────────────────────────
def gauge(score):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        number={'font': {'size': 46}, 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [1, 5], 'tickvals': [1, 2, 3, 4, 5]},
            'bar': {'color': BLUE, 'thickness': .72},
            'bgcolor': 'white',
            'steps': [
                {'range': [1, 2.5], 'color': '#fdeeea'},
                {'range': [2.5, 3.5], 'color': '#fdf6e3'},
                {'range': [3.5, 5], 'color': '#eaf5f0'},
            ],
            'threshold': {'line': {'color': '#333', 'width': 3},
                          'thickness': .8, 'value': 3.23},
        }))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=25, b=10))
    return fig


@st.cache_data
def train_frame():
    X = load_csv('X_train.csv')
    y = load_csv('y_train.csv').iloc[:, 0]
    d = X[NUM_FEATURES].copy()
    d['점수'] = y
    return d


TRAIN = train_frame()


def percentile(col, value):
    """이 값이 학습 데이터에서 상위 몇 %인지."""
    p = (TRAIN[col] < value).mean() * 100
    return p if HIGHER_BETTER.get(col, True) else 100 - p


# ═══════════════════════════════════════════════════════════
# ✏️ 내 글 진단하기
# ═══════════════════════════════════════════════════════════
if page.startswith('✏️'):
    st.title('내 영어 글, 어휘 점수는 몇 점일까?')
    st.caption('영어로 쓴 에세이를 붙여넣으면 어휘 점수를 예측하고, '
               '그 점수가 어디서 왔는지 항목별로 보여드립니다.')

    c1, c2 = st.columns([3, 1])
    with c1:
        text = st.text_area('에세이 (영문)', value=SAMPLE, height=300,
                            label_visibility='collapsed',
                            placeholder='영어 에세이를 여기에 붙여넣으세요...')
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**주제 (선택)**')
        prompts = ['모르겠어요'] + list(ARTIFACTS['token_prompt_stats'].index)
        prompt = st.selectbox('주제', prompts, label_visibility='collapsed')
        st.caption('같은 주제를 쓴 사람들과 분량을 비교할 때 씁니다.')
        st.markdown('</div>', unsafe_allow_html=True)
        st.write('')
        run = st.button('점수 예측하기', type='primary', use_container_width=True)

    if run and text.strip():
        try:
            res = predict(text, None if prompt == '모르겠어요' else prompt,
                          ARTIFACTS, FINAL_MODEL)
        except Exception as e:
            st.error(f'분석하지 못했습니다: {e}')
            st.stop()

        f, score = res['features'], res['score']
        st.divider()

        # ── 점수 ─────────────────────────────────────────
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.plotly_chart(gauge(score), use_container_width=True)
            st.markdown(f'<p class="score-label">채점 형식으로는 '
                        f'<b>{round(score * 2) / 2:.1f}점</b> · '
                        f'예상 범위 {max(1, score - MAE):.1f} ~ '
                        f'{min(5, score + MAE):.1f}</p>',
                        unsafe_allow_html=True)
        with c2:
            pct = (TRAIN['점수'] < score).mean() * 100
            st.markdown('#### 또래 글과 비교하면')
            fig = px.histogram(TRAIN, x='점수', nbins=9,
                               color_discrete_sequence=['#dfe6f2'])
            fig.add_vline(x=score, line_width=3, line_color=ORANGE,
                          annotation_text=f'내 글 {score:.2f}',
                          annotation_position='top')
            fig.add_vline(x=TRAIN['점수'].mean(), line_width=2, line_dash='dot',
                          line_color='#999', annotation_text='평균',
                          annotation_position='bottom left')
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_title='어휘 점수', yaxis_title='사람 수',
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'같은 시험을 본 학생 {len(TRAIN):,}명 중 '
                        f'**상위 {100 - pct:.0f}%** 수준입니다.')

        # 짧은 글 안내 — 과장된 점수를 그대로 믿지 않도록
        if f['flag_very_short']:
            st.warning(f'글이 {f["_Token"]}단어로 짧습니다(100단어 미만). '
                       '짧은 글은 어휘력을 제대로 재기 어려워 이 점수의 신뢰도가 낮습니다.')
        elif f['_Token'] < 300:
            st.info(f'글이 {f["_Token"]}단어입니다. 300단어보다 짧은 글은 '
                    '어휘 다양성이 실제보다 높게 측정되는 경향이 있어, '
                    '점수가 조금 후하게 나올 수 있습니다.')

        st.divider()

        # ── 항목별 프로필 (레이더) ─────────────────────────
        RADAR = ['guiraud', 'MTLD_valid', 'clean_beyond_2k',
                 'mean_word_length', 'academic_suffix_ratio', 'oov_ratio']
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('#### 어휘 프로필')
            vals = [percentile(c, f[c]) for c in RADAR]
            labels = [LABEL[c] for c in RADAR]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[50] * len(RADAR) + [50], theta=labels + [labels[0]],
                fill='toself', name='또래 평균',
                line=dict(color='#c9d3e4'), fillcolor='rgba(201,211,228,.35)'))
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]],
                fill='toself', name='내 글',
                line=dict(color=BLUE, width=3),
                fillcolor='rgba(76,123,217,.25)'))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                           ticksuffix='%')),
                height=380, margin=dict(l=60, r=60, t=30, b=30),
                legend=dict(orientation='h', y=-.08))
            st.plotly_chart(fig, use_container_width=True)
            st.caption('바깥쪽일수록 좋습니다. 회색 원이 또래 평균(50%) 선입니다.')

        with c2:
            st.markdown('#### 점수를 올린 것 / 내린 것')
            imp = load_csv('feature_importance.csv', index_col=0)
            z = res['scaled'].iloc[0]
            contrib = (z[imp.index] * imp['Ridge 계수'])
            contrib = contrib.reindex(contrib.abs().sort_values().index)
            top = contrib.tail(8)
            fig = go.Figure(go.Bar(
                x=top.values, y=[LABEL.get(i, i) for i in top.index],
                orientation='h',
                marker_color=[GREEN if v > 0 else ORANGE for v in top.values],
                hovertemplate='%{y}<br>기여 %{x:+.3f}<extra></extra>'))
            fig.add_vline(x=0, line_color='#666', line_width=1)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title='점수에 준 영향',
                              yaxis_title=None, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption('초록은 점수를 올린 요소, 주황은 내린 요소입니다.')

        # ── 강점 / 보완점 문장 ────────────────────────────
        prof = {c: percentile(c, f[c]) for c in RADAR}
        best = max(prof, key=prof.get)
        worst = min(prof, key=prof.get)
        c1, c2 = st.columns(2)
        c1.success(f'**가장 강한 부분 — {LABEL[best]}** (상위 {100 - prof[best]:.0f}%)\n\n'
                   f'{TIP[best]}')
        c2.warning(f'**보완하면 좋을 부분 — {LABEL[worst]}** (상위 {100 - prof[worst]:.0f}%)\n\n'
                   f'{TIP[worst]}')

        st.divider()

        # ── 글 정보 ──────────────────────────────────────
        st.markdown('#### 글 기본 정보')
        c = st.columns(5)
        c[0].metric('단어 수', f'{f["_Token"]:,}')
        c[1].metric('서로 다른 단어', f'{f["_Type"]:,}')
        c[2].metric('문장 수', f['_num_sent'])
        c[3].metric('문단 수', f['_num_para'])
        c[4].metric('철자 의심 단어', f'{f["oov_ratio"] * 100:.1f}%')

        if f['_oov_words']:
            with st.expander(f'사전에 없는 단어 {len(f["_oov_words"])}개 보기'):
                st.write(' · '.join(f['_oov_words']))
                st.caption('철자 오류이거나 고유명사입니다. '
                           '이 모델은 오타를 고치지 않고 그대로 봅니다 — '
                           '철자 실수 자체가 어휘 습득 정도를 알려주는 신호이기 때문입니다.')

        with st.expander('측정한 항목 15개 전부 보기'):
            rows = [{'항목': LABEL.get(c, c), '설명': TIP.get(c, ''),
                     '내 값': round(f[c], 3),
                     '또래 대비 위치': f'상위 {100 - percentile(c, f[c]):.0f}%'}
                    for c in NUM_FEATURES]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

    elif run:
        st.error('에세이를 입력해 주세요.')

# ═══════════════════════════════════════════════════════════
# 🔍 무엇이 점수를 만드나
# ═══════════════════════════════════════════════════════════
elif page.startswith('🔍'):
    st.title('무엇이 어휘 점수를 만드나')
    st.caption('직접 값을 움직여 보면서 각 요소가 점수를 어떻게 바꾸는지 확인해 보세요.')

    imp = load_csv('feature_importance.csv', index_col=0)
    o = imp.sort_values('Ridge 계수')
    fig = go.Figure(go.Bar(
        x=o['Ridge 계수'], y=[LABEL.get(i, i) for i in o.index], orientation='h',
        marker_color=[GREEN if v > 0 else ORANGE for v in o['Ridge 계수']],
        hovertemplate='%{y}<br>영향력 %{x:+.3f}<extra></extra>'))
    fig.add_vline(x=0, line_color='#666')
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title='점수에 주는 영향 (초록 = 올림 / 주황 = 내림)',
                      title='항목별 영향력')
    st.plotly_chart(fig, use_container_width=True)

    st.info('**어휘 다양성**과 **철자 정확성**이 압도적인 1·2위입니다. '
            '얼마나 길게 썼는지보다, 어떤 단어를 얼마나 정확하게 썼는지가 중요합니다.')

    st.divider()
    st.subheader('직접 움직여 보기')
    st.caption('평균적인 글에서 항목 하나를 바꾸면 점수가 어떻게 달라질까요?')

    SLIDERS = ['guiraud', 'MTLD_valid', 'clean_beyond_2k',
               'mean_word_length', 'oov_ratio', 'len_z_in_prompt']
    med = TRAIN[NUM_FEATURES].median()

    cols = st.columns(3)
    vals = med.copy()
    for i, c in enumerate(SLIDERS):
        lo, hi = TRAIN[c].quantile(.02), TRAIN[c].quantile(.98)
        with cols[i % 3]:
            vals[c] = st.slider(LABEL[c], float(lo), float(hi), float(med[c]),
                                (float(hi) - float(lo)) / 100, help=TIP.get(c))

    # 학습 때와 동일한 변환 → 앙상블 예측
    raw = pd.DataFrame([vals[NUM_FEATURES]])
    logged = raw.copy()
    for c in ARTIFACTS['log_cols']:
        logged[c] = np.log1p(logged[c])
    scaled = pd.DataFrame(ARTIFACTS['scaler'].transform(logged),
                          columns=NUM_FEATURES)
    m = FINAL_MODEL['members']
    sim = float(np.clip(np.mean([
        m['lasso'].predict(scaled[FINAL_MODEL['lasso_features']])[0],
        m['rf'].predict(raw[FINAL_MODEL['tree_features']])[0],
        m['gb'].predict(raw[FINAL_MODEL['tree_features']])[0]]),
        *FINAL_MODEL['clip_range']))

    base_raw = pd.DataFrame([med[NUM_FEATURES]])
    base_log = base_raw.copy()
    for c in ARTIFACTS['log_cols']:
        base_log[c] = np.log1p(base_log[c])
    base_scaled = pd.DataFrame(ARTIFACTS['scaler'].transform(base_log),
                               columns=NUM_FEATURES)
    base = float(np.clip(np.mean([
        m['lasso'].predict(base_scaled[FINAL_MODEL['lasso_features']])[0],
        m['rf'].predict(base_raw[FINAL_MODEL['tree_features']])[0],
        m['gb'].predict(base_raw[FINAL_MODEL['tree_features']])[0]]),
        *FINAL_MODEL['clip_range']))

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(gauge(sim), use_container_width=True)
        st.metric('평균적인 글 대비', f'{sim:.2f}점', f'{sim - base:+.2f}')
    with c2:
        st.markdown('#### 지금 설정한 글의 프로필')
        labels = [LABEL[c] for c in SLIDERS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[50] * len(SLIDERS) + [50], theta=labels + [labels[0]],
            fill='toself', name='평균', line=dict(color='#c9d3e4')))
        pv = [percentile(c, vals[c]) for c in SLIDERS]
        fig.add_trace(go.Scatterpolar(
            r=pv + [pv[0]], theta=labels + [labels[0]], fill='toself',
            name='설정값', line=dict(color=BLUE, width=3)))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix='%')),
                          height=330, margin=dict(l=60, r=60, t=20, b=20),
                          legend=dict(orientation='h', y=-.1))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# 🌐 데이터 둘러보기
# ═══════════════════════════════════════════════════════════
elif page.startswith('🌐'):
    st.title('데이터 둘러보기')
    st.caption(f'영어 학습자 에세이 {len(TRAIN):,}편을 항목별로 살펴봅니다. '
               '그래프는 마우스로 돌리고 확대할 수 있습니다.')

    st.subheader('3차원으로 보기')
    c1, c2, c3 = st.columns(3)
    opts = [c for c in NUM_FEATURES if c != 'flag_very_short']
    fx = c1.selectbox('가로축', opts, index=opts.index('guiraud'),
                      format_func=lambda c: LABEL[c])
    fy = c2.selectbox('세로축', opts, index=opts.index('oov_ratio'),
                      format_func=lambda c: LABEL[c])
    fz = c3.selectbox('높이축', opts, index=opts.index('mean_word_length'),
                      format_func=lambda c: LABEL[c])

    samp = TRAIN.sample(min(1200, len(TRAIN)), random_state=42)
    fig = px.scatter_3d(samp, x=fx, y=fy, z=fz, color='점수',
                        color_continuous_scale='RdYlBu', opacity=.75,
                        labels={fx: LABEL[fx], fy: LABEL[fy], fz: LABEL[fz]})
    fig.update_traces(marker=dict(size=3.4))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title=LABEL[fx], yaxis_title=LABEL[fy],
                                 zaxis_title=LABEL[fz]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption('파란 점이 높은 점수, 빨간 점이 낮은 점수입니다. '
               '드래그로 회전, 휠로 확대할 수 있습니다.')

    st.divider()
    st.subheader('길게 쓰면 점수가 오를까?')
    train_raw = load_csv('train.csv')
    if train_raw is not None:
        d = pd.DataFrame({'분량(단어 수)': train_raw['Token'],
                          '어휘 다양성': train_raw['guiraud'],
                          '점수': train_raw['Vocabulary']})
        c1, c2 = st.columns(2)
        for col, (x, title) in zip(
                [c1, c2],
                [('분량(단어 수)', '분량과 점수'), ('어휘 다양성', '어휘 다양성과 점수')]):
            r = d[x].corr(d['점수'])
            fig = px.scatter(d, x=x, y='점수', opacity=.25,
                             color_discrete_sequence=['#a8b8d4'])
            # 추세선을 직접 계산합니다 (외부 패키지 없이).
            b, a = np.polyfit(d[x], d['점수'], 1)
            xs = np.linspace(d[x].min(), d[x].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode='lines',
                                     line=dict(color=ORANGE, width=3),
                                     name='추세선', hoverinfo='skip'))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10),
                              title=f'{title} (상관 {r:+.2f})', showlegend=False)
            col.plotly_chart(fig, use_container_width=True)
        st.success('**분량만으로는 점수의 7%밖에 설명되지 않습니다.** '
                   '어휘 다양성 같은 질적 지표 4개만 써도 27%까지 올라갑니다. '
                   '길게 쓰는 것보다 어떤 단어를 쓰는지가 훨씬 중요합니다.')

    st.divider()
    st.subheader('항목별 분포 비교')
    pick = st.multiselect('보고 싶은 항목', opts,
                          default=['guiraud', 'oov_ratio', 'mean_word_length'],
                          format_func=lambda c: LABEL[c])
    if pick:
        band = pd.cut(TRAIN['점수'], [0, 2.5, 3.25, 5],
                      labels=['낮음 (~2.5)', '중간 (3.0)', '높음 (3.5~)'])
        long = TRAIN[pick].copy()
        long['점수대'] = band
        long = long.melt(id_vars='점수대', var_name='항목', value_name='값')
        long['항목'] = long['항목'].map(LABEL)
        fig = px.box(long, x='항목', y='값', color='점수대',
                     color_discrete_sequence=[ORANGE, '#E0B341', GREEN],
                     facet_col='항목', facet_col_wrap=3)
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(visible=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig.update_layout(height=330 * ((len(pick) + 2) // 3),
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption('점수대가 올라갈수록 상자가 위로(또는 아래로) 이동하면 '
                   '그 항목이 점수와 관련 있다는 뜻입니다.')

# ═══════════════════════════════════════════════════════════
# 📊 얼마나 믿을 만한가
# ═══════════════════════════════════════════════════════════
elif page.startswith('📊'):
    st.title('이 예측, 얼마나 믿을 만한가')

    c = st.columns(3)
    c[0].metric('평균 오차', f'{MAE:.2f}점', '채점 단위 0.5점 기준 약 0.7단계')
    c[1].metric('설명한 비율', f"{SUMMARY['test']['R²'] * 100:.0f}%",
                '아무것도 안 하면 0%')
    c[2].metric('검증 편수', '750편', '학습에 쓰지 않은 글')

    st.divider()
    st.subheader('아무 정보도 없이 찍는 것보다 얼마나 나은가')
    prog = pd.DataFrame([
        ['평균만 답하기', 0.4700, 0.0],
        ['글 길이만 보기', 0.4395, 7.2],
        ['핵심 항목 4개', 0.3901, 26.6],
        ['이 모델 (항목 15개)', SUMMARY['test']['MAE'], SUMMARY['test']['R²'] * 100],
    ], columns=['방법', '평균 오차', '설명한 비율(%)'])
    c1, c2 = st.columns(2)
    for col, (y, title, color) in zip(
            [c1, c2],
            [('평균 오차', '평균 오차 — 낮을수록 좋음', ORANGE),
             ('설명한 비율(%)', '설명력 — 높을수록 좋음', BLUE)]):
        fig = px.bar(prog, x=y, y='방법', orientation='h',
                     color_discrete_sequence=[color], text_auto='.3g')
        fig.update_layout(height=310, margin=dict(l=10, r=10, t=45, b=10),
                          title=title, yaxis_title=None)
        col.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('실제 점수 vs 예측 점수')
    pred = load_csv('final_predictions.csv')
    if pred is not None:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = px.scatter(pred, x='예측', y='실제', opacity=.3,
                             color_discrete_sequence=[BLUE])
            fig.add_shape(type='line', x0=1, y0=1, x1=5, y1=5,
                          line=dict(color=ORANGE, dash='dash', width=2))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10),
                              title='점이 주황 선에 가까울수록 정확한 예측',
                              xaxis_range=[1, 5], yaxis_range=[1, 5])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            bias = pred.groupby('실제')['잔차'].mean().reset_index()
            fig = px.bar(bias, x='실제', y='잔차',
                         color=bias['잔차'] > 0,
                         color_discrete_map={True: GREEN, False: ORANGE})
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10),
                              title='점수대별 예측 치우침', showlegend=False,
                              xaxis_title='실제 점수', yaxis_title='실제 − 예측')
            st.plotly_chart(fig, use_container_width=True)

        st.warning("""
**이 모델의 약점은 명확합니다.**

아주 잘 쓴 글은 실제보다 **낮게**, 아주 못 쓴 글은 실제보다 **높게** 예측합니다.
학습에 쓴 글이 대부분 중간 점수대(3.0~3.5점)에 몰려 있어서,
모델이 양 끝단을 거의 본 적이 없기 때문입니다.

이건 모델을 바꿔서 해결되는 문제가 아니라 데이터의 한계입니다.
실제로 6가지 개선 방법을 시도했지만 이 치우침은 그대로였습니다.
        """)

    st.divider()
    st.subheader('그래서, 어디까지 믿으면 되나')
    st.markdown(f"""
| 상황 | 신뢰도 |
|---|---|
| 300단어 이상, 중간 점수대(2.5~4.0) 글 | 비교적 정확 (±{MAE:.2f}점) |
| 100~300단어의 짧은 글 | 점수가 다소 후하게 나옴 |
| 100단어 미만 | 신뢰도 낮음 — 참고만 |
| 아주 잘 쓴 글 / 아주 못 쓴 글 | 중간 쪽으로 끌려감 |

**참고로**, 사람 채점자끼리도 이 항목의 일치도가 높지 않습니다
(Cohen's κ = .518로, 7개 채점 항목 중 가장 낮음).
정답이 하나로 정해지는 과제가 아니라는 뜻입니다.

이 도구는 **글을 고칠 방향을 찾는 참고 지표**이지, 실제 채점을 대신하지 않습니다.
    """)

# ═══════════════════════════════════════════════════════════
# ℹ️ 이 프로젝트는
# ═══════════════════════════════════════════════════════════
else:
    st.title('이 프로젝트는')

    st.markdown("""
### 하려던 것

영어 학습자가 쓴 에세이의 **어휘 점수를 예측**하되,
"몇 점"만 알려주는 것이 아니라 **왜 그 점수인지 설명**할 수 있는 모델을 만드는 것.
    """)

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><b>데이터</b><br>ELLIPSE Corpus<br>'
                '8~12학년 영어 학습자 에세이 3,747편</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><b>예측 대상</b><br>Vocabulary 점수<br>'
                '1.0 ~ 5.0점 · 0.5점 단위</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><b>방법</b><br>원문에서 언어 특성 28개 설계<br>'
                '→ 15개 선별 → 3개 모델 결합</div>', unsafe_allow_html=True)

    st.write('')
    st.divider()
    st.subheader('가장 어려웠던 부분')
    st.markdown("""
원본 데이터에 있던 어휘 관련 정보는 네 개뿐이었고,
그마저도 대부분 **"얼마나 길게 썼는가"를 재고 있었습니다.**

채점자가 실제로 보는 **어휘의 수준**과 **철자 정확성**은 데이터에 아예 없었습니다.
그래서 에세이 원문을 다시 읽어 그 축을 직접 만들어야 했습니다.
    """)

    st.info("""
**예를 들면 이런 것입니다.**

흔히 텍스트 분석에서는 오타를 먼저 교정합니다.
그런데 이 프로젝트에서는 **오타를 그대로 뒀습니다.**

우리가 예측하려는 것이 "어휘를 얼마나 익혔는가"이고,
철자 오류는 그 단어를 아직 확실히 익히지 못했다는 **관찰 가능한 증거**이기 때문입니다.
교정했다면 가장 중요한 신호 하나를 스스로 지우는 셈이 됐을 겁니다.

실제로 철자 정확성은 전체 항목 중 **영향력 2위**로 나왔습니다.
    """)

    st.divider()
    st.subheader('만든 과정')
    steps = pd.DataFrame([
        ['1', '기획', '주제 선정 · 가설 7개 설계'],
        ['2', '전처리', '정제 · 언어 특성 28개 생성'],
        ['3', '탐색', '분포 · 상관 · 편향 점검'],
        ['4', '변수 선택', '28개 → 15개 (겹치는 것 정리)'],
        ['5', '모델링', '4가지 모델 비교'],
        ['6', '성능 향상', '6가지 방법 시도 → 3개 모델 결합'],
    ], columns=['단계', '이름', '내용'])
    st.dataframe(steps, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader('알게 된 것 세 가지')
    c1, c2, c3 = st.columns(3)
    c1.markdown('#### 1️⃣\n**길이는 답이 아니다**\n\n'
                '길게 쓰는 것만으로 설명되는 몫은 7%뿐이었습니다. '
                '어떤 단어를 쓰는지가 훨씬 중요합니다.')
    c2.markdown('#### 2️⃣\n**모델보다 특성이 중요하다**\n\n'
                '4가지 모델의 성능 차이는 측정 오차 수준이었습니다. '
                '성능을 만든 것은 알고리즘이 아니라 어떤 특성을 넣었는가였습니다.')
    c3.markdown('#### 3️⃣\n**한계를 아는 것도 결과다**\n\n'
                '6가지 방법으로 개선해도 벽이 있었습니다. '
                '그 벽이 어디서 오는지 밝힌 것이 이 프로젝트의 성과 중 하나입니다.')

    st.divider()
    st.caption('데이터 출처: ELLIPSE Corpus (https://github.com/scrosseye/ELLIPSE-Corpus) · '
               '이 도구는 학습·연구 목적이며 실제 채점을 대체하지 않습니다.')
