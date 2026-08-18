"""
for_users.py — 어휘 판독기 (Lexis Reader)
========================================
실행:  python -m streamlit run for_users.py

디자인 방향
-----------
주제가 '어휘'이므로 사전과 교정지의 세계에서 형태를 가져왔습니다.
잉크빛 남색, 교정 표시의 적갈색, 뉴스프린트 지면.
화면의 주인공은 점수 숫자가 아니라 **표시가 들어간 원고 자체**입니다.
"""

import html
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from feature_extraction import predict, _load_bands, TOKEN_RE

DATA = Path('data')

st.set_page_config(page_title='어휘 판독기', page_icon='✒️',
                   layout='wide', initial_sidebar_state='expanded')

# ══════════════════════════════════════════════════════════
#  디자인 토큰
# ══════════════════════════════════════════════════════════
INK = '#1F2E4D'        # 잉크 남색 — 제목, 주요 요소
INK_SOFT = '#5F6E8C'   # 흐린 잉크 — 보조 텍스트
OXBLOOD = '#93343B'    # 교정 표시 — 부정 방향
SAGE = '#43705F'       # 검수 완료 — 긍정 방향
PAPER = '#F5F5F1'      # 지면
PAPER_HI = '#FCFCFA'   # 밝은 지면 (카드)
RULE = '#D9D9D1'       # 괘선
GOLD = '#A8823C'       # 리본 — 딱 한 곳에만

# 웹폰트 링크는 CSS와 반드시 따로 넣습니다.
# <link>로 시작하면 마크다운이 이를 '빈 줄에서 끝나는 HTML 블록'으로 보기 때문에,
# 뒤따르는 CSS가 첫 빈 줄부터 본문 텍스트로 화면에 찍혀 버립니다.
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,700'
    '&family=Gowun+Batang:wght@400;700'
    '&family=IBM+Plex+Sans+KR:wght@300;400;500;600'
    '&family=IBM+Plex+Mono:wght@400;500&display=swap">',
    unsafe_allow_html=True)

# <style>로 시작하면 </style>까지 하나의 HTML 블록으로 처리되지만,
# 빈 줄을 미리 걷어내 어떤 마크다운 구현에서도 깨지지 않게 합니다.
_CSS = f"""<style>
:root {{
  --ink:{INK}; --ink-soft:{INK_SOFT}; --oxblood:{OXBLOOD}; --sage:{SAGE};
  --paper:{PAPER}; --paper-hi:{PAPER_HI}; --rule:{RULE}; --gold:{GOLD};
  --serif:'Fraunces','Gowun Batang',Georgia,serif;
  --sans:'IBM Plex Sans KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif;
  --mono:'IBM Plex Mono','D2Coding',monospace;
}}

/* ── 지면 ───────────────────────────────────────────── */
.stApp {{ background:var(--paper); }}
[data-testid="stHeader"] {{ background:transparent; height:0; }}
.block-container {{ padding:2.6rem 3rem 5rem; max-width:1180px; }}
html, body, [class*="css"], .stMarkdown, p, li, label, div[data-baseweb] {{
  font-family:var(--sans); color:var(--ink);
}}
p, li {{ line-height:1.75; }}

/* ── 서체 위계 ───────────────────────────────────────── */
h1 {{ font-family:var(--serif); font-weight:500; font-size:2.7rem;
     letter-spacing:-.02em; line-height:1.15; color:var(--ink);
     margin:0 0 .35rem; }}
h2 {{ font-family:var(--serif); font-weight:500; font-size:1.5rem;
     letter-spacing:-.01em; color:var(--ink); margin:2.4rem 0 .9rem; }}
h3 {{ font-family:var(--sans); font-weight:600; font-size:1.02rem;
     color:var(--ink); margin:1.7rem 0 .6rem; }}
h4 {{ font-family:var(--sans); font-weight:600; font-size:.94rem;
     color:var(--ink); margin:1.2rem 0 .5rem; }}

/* 표제 위 작은 라벨 — 사전의 분류 표시에서 가져왔습니다 */
.eyebrow {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.22em;
  text-transform:uppercase; color:var(--ink-soft); margin-bottom:.8rem;
  display:flex; align-items:center; gap:.75rem; }}
.eyebrow::after {{ content:''; flex:1; height:1px; background:var(--rule); }}
.lede {{ font-size:1.02rem; color:var(--ink-soft); max-width:62ch;
  line-height:1.72; margin:.5rem 0 .3rem; }}

/* ── 좌측 색인 ───────────────────────────────────────── */
[data-testid="stSidebar"] {{ background:var(--ink); border:none; }}
[data-testid="stSidebar"] * {{ color:#DDE3EE; }}
[data-testid="stSidebar"] .block-container {{ padding:2.2rem 1.4rem; }}
.brand {{ font-family:var(--serif); font-size:1.55rem; font-weight:500;
  color:#FFFFFF; line-height:1.15; margin-bottom:.3rem; }}
.brand-sub {{ font-family:var(--mono); font-size:.65rem; letter-spacing:.2em;
  text-transform:uppercase; color:#8FA0BE; }}
.rail-rule {{ height:1px; background:#33456B; margin:1.5rem 0 1.15rem; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:.15rem; }}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
  padding:.5rem .1rem .5rem 1rem; border-left:2px solid transparent;
  transition:border-color .15s ease, background .15s ease; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
  border-left-color:#546C9B; background:rgba(255,255,255,.03); }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
  border-left-color:var(--gold); background:rgba(255,255,255,.05); }}
[data-testid="stSidebar"] [role="radiogroup"] p {{
  font-size:.9rem; color:#C3CEE2; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{
  color:#FFFFFF; font-weight:500; }}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] {{
  margin-left:-.4rem; }}
.rail-note {{ font-size:.74rem; color:#8FA0BE; line-height:1.6; }}
.rail-stat {{ font-family:var(--mono); font-size:1.5rem; color:#FFFFFF;
  line-height:1.3; }}

/* ── 사전 표제부 ─────────────────────────────────────── */
.entry {{ border-top:2px solid var(--ink); border-bottom:1px solid var(--rule);
  padding:1.5rem 0 1.35rem; }}
.entry-head {{ font-family:var(--serif); font-size:1.3rem; color:var(--ink);
  letter-spacing:.01em; }}
.entry-pos {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.16em;
  color:var(--ink-soft); text-transform:uppercase; margin-left:.6rem; }}
.entry-score {{ font-family:var(--serif); font-size:5.2rem; font-weight:300;
  line-height:.95; letter-spacing:-.035em; color:var(--ink); }}
.entry-of {{ font-family:var(--mono); font-size:.95rem; color:var(--ink-soft);
  margin-left:.35rem; }}
.entry-gloss {{ font-size:.94rem; color:var(--ink-soft); line-height:1.72;
  margin-top:.6rem; }}
.entry-gloss b {{ color:var(--ink); font-weight:500; }}

/* ── 원고 (핵심 요소) ────────────────────────────────── */
.ms {{ background:var(--paper-hi); border:1px solid var(--rule); border-radius:3px;
  padding:2.2rem 2.5rem; max-height:560px; overflow-y:auto;
  box-shadow:0 1px 0 rgba(31,46,77,.04); }}
.ms p {{ margin:0 0 1.15rem; font-family:var(--serif); font-size:1.02rem;
  color:#22252B; line-height:2.05; }}
.ms p:last-child {{ margin-bottom:0; }}
.mk-oov {{ color:var(--oxblood); text-decoration:underline wavy var(--oxblood);
  text-underline-offset:5px; text-decoration-thickness:1px; }}
.mk-adv {{ color:var(--ink); border-bottom:2px solid rgba(31,46,77,.3);
  padding-bottom:1px; }}
.legend {{ display:flex; gap:1.8rem; flex-wrap:wrap; font-size:.81rem;
  color:var(--ink-soft); margin-top:1rem; line-height:1.6; }}
.legend b {{ font-family:var(--mono); font-weight:500; color:var(--ink); }}

/* ── 카드 ────────────────────────────────────────────── */
.card {{ background:var(--paper-hi); border:1px solid var(--rule);
  border-radius:3px; padding:1.15rem 1.3rem; height:100%; }}
.card-k {{ font-family:var(--mono); font-size:.65rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-soft); }}
.card-v {{ font-family:var(--serif); font-size:1.9rem; font-weight:400;
  color:var(--ink); line-height:1.2; margin-top:.3rem; }}
.card-n {{ font-size:.78rem; color:var(--ink-soft); margin-top:.25rem;
  line-height:1.55; }}

/* 강점 / 보완 — 색으로만 구분하고 장식은 넣지 않습니다 */
.verdict {{ border-left:3px solid var(--rule); padding:.2rem 0 .2rem 1.15rem; }}
.verdict.good {{ border-left-color:var(--sage); }}
.verdict.weak {{ border-left-color:var(--oxblood); }}
.verdict-t {{ font-weight:600; font-size:.95rem; color:var(--ink); }}
.verdict-t .pct {{ font-family:var(--mono); font-weight:400; font-size:.83rem;
  color:var(--ink-soft); margin-left:.45rem; }}
.verdict-d {{ font-size:.87rem; color:var(--ink-soft); margin-top:.3rem;
  line-height:1.65; }}

/* 주의 문구 — Streamlit 기본 알림 대신 씁니다 */
.note {{ border-left:3px solid var(--gold); background:rgba(168,130,60,.07);
  padding:.9rem 1.15rem; font-size:.89rem; line-height:1.75;
  color:var(--ink); border-radius:0 3px 3px 0; }}
.note.warn {{ border-left-color:var(--oxblood);
  background:rgba(147,52,59,.055); }}
.note b {{ font-weight:600; }}

/* ── 폼 요소 ─────────────────────────────────────────── */
.stTextArea textarea {{ font-family:var(--serif); font-size:1rem; line-height:1.85;
  background:var(--paper-hi); border:1px solid var(--rule); border-radius:3px;
  color:#22252B; padding:1.3rem 1.5rem; }}
.stTextArea textarea:focus {{ border-color:var(--ink);
  box-shadow:0 0 0 2px rgba(31,46,77,.12); }}
.stButton button {{ background:var(--ink); color:#FFFFFF; border:1px solid var(--ink);
  border-radius:3px; font-family:var(--sans); font-weight:500; font-size:.92rem;
  letter-spacing:.01em; padding:.62rem 1.1rem; transition:background .15s ease; }}
.stButton button:hover {{ background:#16223B; border-color:#16223B; color:#FFFFFF; }}
.stButton button:focus-visible {{ outline:2px solid var(--gold);
  outline-offset:2px; }}
div[data-baseweb="select"] > div {{ background:var(--paper-hi);
  border-color:var(--rule); border-radius:3px; }}
[data-testid="stWidgetLabel"] p {{ font-size:.83rem; font-weight:500;
  color:var(--ink-soft); }}

/* ── 표 ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{ border:1px solid var(--rule); border-radius:3px; }}
[data-testid="stExpander"] {{ border:1px solid var(--rule); border-radius:3px;
  background:var(--paper-hi); }}
[data-testid="stExpander"] summary p {{ font-size:.88rem; font-weight:500; }}
.stMarkdown table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
.stMarkdown th {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-soft); text-align:left;
  border-bottom:1px solid var(--ink); padding:.55rem .8rem .5rem 0; font-weight:400; }}
.stMarkdown td {{ border-bottom:1px solid var(--rule); padding:.65rem .8rem .65rem 0;
  vertical-align:top; }}

hr {{ border:none; border-top:1px solid var(--rule); margin:2.5rem 0 1.8rem; }}

@media (prefers-reduced-motion: reduce) {{
  * {{ transition:none !important; animation:none !important; }}
}}
@media (max-width: 780px) {{
  .block-container {{ padding:1.6rem 1.1rem 3rem; }}
  h1 {{ font-size:2rem; }}
  .entry-score {{ font-size:3.8rem; }}
  .ms {{ padding:1.3rem 1.4rem; }}
  .ms p {{ font-size:.97rem; line-height:1.9; }}
}}
</style>"""
st.markdown(re.sub(r'\n\s*\n', '\n', _CSS), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  자료 적재
# ══════════════════════════════════════════════════════════
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

LABEL = {
    'guiraud': '어휘 다양성', 'MTLD_valid': '어휘 지속력',
    'herdan_c': '어휘 다양성(로그)', 'hapax_ratio': '한 번만 쓴 단어',
    'oov_ratio': '철자 정확성', 'mean_word_length': '단어 길이',
    'mean_syllables': '단어 음절 수', 'academic_suffix_ratio': '학술 어휘',
    'k2_ratio': '중급 어휘', 'clean_beyond_2k': '고급 어휘',
    'mean_sent_length': '문장 길이', 'sent_length_std': '문장 길이 변화',
    'sent_per_para': '문단 구성', 'len_z_in_prompt': '글 분량',
    'flag_very_short': '짧은 글 표시',
}
HIGHER_BETTER = {
    'guiraud': True, 'MTLD_valid': True, 'hapax_ratio': True, 'oov_ratio': False,
    'mean_word_length': True, 'mean_syllables': True, 'academic_suffix_ratio': True,
    'clean_beyond_2k': True, 'mean_sent_length': False, 'sent_length_std': False,
    'sent_per_para': True, 'len_z_in_prompt': True, 'herdan_c': True,
    'k2_ratio': True, 'flag_very_short': False,
}
TIP = {
    'guiraud': '같은 분량에서 서로 다른 단어를 얼마나 많이 썼는지',
    'MTLD_valid': '같은 단어를 다시 꺼내 쓰기까지 얼마나 오래 버티는지',
    'oov_ratio': '사전에 없는 단어, 즉 철자 오류나 고유명사의 비율',
    'clean_beyond_2k': '흔하지 않은 단어를 얼마나 썼는지 (오타는 뺀 값)',
    'mean_word_length': '쓴 단어들의 평균 글자 수',
    'mean_syllables': '쓴 단어들의 평균 음절 수',
    'academic_suffix_ratio': '-tion, -ity 처럼 학술적인 어미를 가진 단어 비율',
    'mean_sent_length': '문장 하나가 평균 몇 단어인지. 너무 길면 감점 방향',
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


# ══════════════════════════════════════════════════════════
#  공통 요소
# ══════════════════════════════════════════════════════════
def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def lede(text):
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def note(text, kind=''):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def stat_card(key, value, sub=''):
    st.markdown(f'<div class="card"><div class="card-k">{key}</div>'
                f'<div class="card-v">{value}</div>'
                f'<div class="card-n">{sub}</div></div>', unsafe_allow_html=True)


def style_fig(fig, height=340, legend=False):
    """모든 그래프가 같은 지면 위에 있는 것처럼 보이게 합니다."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans KR, sans-serif', size=12, color=INK),
        title=dict(font=dict(family='Fraunces, Gowun Batang, serif',
                             size=15, color=INK), x=0, xanchor='left'),
        showlegend=legend,
        legend=dict(orientation='h', y=-.16, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=PAPER_HI, bordercolor=RULE,
                        font=dict(family='IBM Plex Sans KR', color=INK)),
    )
    fig.update_xaxes(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE,
                     tickfont=dict(size=11, color=INK_SOFT),
                     title_font=dict(size=11, color=INK_SOFT))
    fig.update_yaxes(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE,
                     tickfont=dict(size=11, color=INK_SOFT),
                     title_font=dict(size=11, color=INK_SOFT))
    return fig


@st.cache_data
def train_frame():
    X = load_csv('X_train.csv')
    y = load_csv('y_train.csv').iloc[:, 0]
    d = X[NUM_FEATURES].copy()
    d['점수'] = y
    return d


TRAIN = train_frame()
NO_BAR = {'displayModeBar': False}


def percentile(col, value):
    p = (TRAIN[col] < value).mean() * 100
    return p if HIGHER_BETTER.get(col, True) else 100 - p


def render_manuscript(text, limit=7000):
    """
    원고에 교정 표시를 넣습니다.
      · 사전에 없는 단어  → 적갈색 물결 밑줄
      · 저빈도(고급) 단어  → 잉크색 실선 밑줄

    이 화면의 주인공입니다. 무엇이 점수를 만들었는지를 숫자가 아니라
    사용자 자신의 문장 위에서 바로 보게 하는 것이 목적입니다.
    """
    bands = _load_bands()
    K1, K2, KNOWN = bands['K1'], bands['K2'], bands['KNOWN']
    src = text[:limit]
    out, cursor = [], 0
    n_oov = n_adv = 0
    for m in TOKEN_RE.finditer(src.lower()):
        s, e = m.span()
        w = m.group()
        cls = None
        if w not in KNOWN:
            cls, n_oov = 'mk-oov', n_oov + 1
        elif w not in K1 and w not in K2:
            cls, n_adv = 'mk-adv', n_adv + 1
        if cls:
            out.append(html.escape(src[cursor:s]))
            out.append(f'<span class="{cls}">{html.escape(src[s:e])}</span>')
            cursor = e
    out.append(html.escape(src[cursor:]))
    body = ''.join(out)
    paras = ''.join(f'<p>{p}</p>' for p in re.split(r'\n\s*\n|\n', body) if p.strip())
    return paras, n_oov, n_adv, len(text) > limit


# ══════════════════════════════════════════════════════════
#  좌측 색인
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="brand">어휘 판독기</div>'
                '<div class="brand-sub">Lexis Reader</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="rail-rule"></div>', unsafe_allow_html=True)
    page = st.radio('메뉴', [
        '원고 판독',
        '점수를 만드는 것',
        '자료 살펴보기',
        '정확도와 한계',
        '이 도구에 대하여',
    ], label_visibility='collapsed')
    st.markdown('<div class="rail-rule"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rail-note">평균 오차</div>'
                f'<div class="rail-stat">±{MAE:.2f}</div>'
                f'<div class="rail-note" style="margin-top:.55rem">'
                f'1.0–5.0점 척도 · 0.5점 단위 채점<br>'
                f'학습에 쓰지 않은 글 750편으로 확인</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  원고 판독
# ══════════════════════════════════════════════════════════
if page == '원고 판독':
    eyebrow('원고 판독')
    st.markdown('# 당신의 문장에서<br>어휘를 읽습니다', unsafe_allow_html=True)
    lede('영어로 쓴 글을 붙여넣으면 어휘 점수를 매기고, 그 점수가 '
         '어느 단어에서 왔는지 원고 위에 그대로 표시합니다.')

    st.write('')
    c1, c2 = st.columns([3, 1], gap='large')
    with c1:
        text = st.text_area('에세이', value=SAMPLE, height=290,
                            label_visibility='collapsed',
                            placeholder='영어로 쓴 글을 여기에 붙여넣으세요')
    with c2:
        prompts = ['모르겠어요'] + list(ARTIFACTS['token_prompt_stats'].index)
        prompt = st.selectbox('과제 주제', prompts)
        st.markdown('<div class="card-n">같은 주제를 쓴 다른 학생들과 분량을 '
                    '비교할 때 씁니다. 몰라도 괜찮습니다.</div>',
                    unsafe_allow_html=True)
        st.write('')
        run = st.button('판독하기', type='primary', use_container_width=True)

    if run and text.strip():
        try:
            res = predict(text, None if prompt == '모르겠어요' else prompt,
                          ARTIFACTS, FINAL_MODEL)
        except Exception as e:
            note(f'글을 읽지 못했습니다. 영문 텍스트인지 확인해 주세요. ({e})', 'warn')
            st.stop()

        f, score = res['features'], res['score']
        pct_below = (TRAIN['점수'] < score).mean() * 100

        # ── 사전 표제부 ──────────────────────────────────
        st.markdown('<hr>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.3], gap='large')
        with c1:
            st.markdown(f"""
<div class="entry">
  <div class="entry-head">어휘 <i>vo·cab·u·lar·y</i><span class="entry-pos">판독 결과</span></div>
  <div style="margin-top:.75rem">
    <span class="entry-score">{score:.2f}</span><span class="entry-of">/ 5.00</span>
  </div>
  <div class="entry-gloss">
    채점 형식으로는 <b>{round(score * 2) / 2:.1f}점</b>.
    같은 시험을 본 학생 {len(TRAIN):,}명 가운데 <b>상위 {100 - pct_below:.0f}%</b>입니다.<br>
    실제 점수는 <b>{max(1, score - MAE):.1f}점에서 {min(5, score + MAE):.1f}점</b> 사이일
    가능성이 높습니다.
  </div>
</div>""", unsafe_allow_html=True)

            if f['flag_very_short']:
                st.write('')
                note(f'이 글은 {f["_Token"]}단어로, 어휘력을 재기에 너무 짧습니다'
                     f'(100단어 미만). 점수를 참고만 하세요.', 'warn')
            elif f['_Token'] < 300:
                st.write('')
                note(f'이 글은 {f["_Token"]}단어입니다. 300단어보다 짧은 글은 어휘 '
                     f'다양성이 실제보다 높게 잡히는 편이라, 점수가 조금 후하게 '
                     f'나왔을 수 있습니다.')

        with c2:
            fig = px.histogram(TRAIN, x='점수', nbins=9,
                               color_discrete_sequence=['#DEE1E7'])
            fig.add_vline(x=score, line_width=2.5, line_color=OXBLOOD)
            fig.add_annotation(x=score, y=1, yref='paper', text='이 글',
                               showarrow=False, yanchor='bottom',
                               font=dict(color=OXBLOOD, size=12))
            fig.add_vline(x=TRAIN['점수'].mean(), line_width=1,
                          line_dash='dot', line_color=INK_SOFT)
            fig.add_annotation(x=TRAIN['점수'].mean(), y=1, yref='paper',
                               text='평균', showarrow=False, yanchor='bottom',
                               font=dict(color=INK_SOFT, size=11))
            fig.update_layout(title='학생 3,000명의 점수 분포 위에서',
                              xaxis_title=None, yaxis_title=None, bargap=.12)
            st.plotly_chart(style_fig(fig, 305), use_container_width=True,
                            config=NO_BAR)

        # ── 원고 (핵심 요소) ─────────────────────────────
        st.markdown('<hr>', unsafe_allow_html=True)
        eyebrow('표시된 원고')
        st.markdown('## 어느 단어가 점수를 만들었나')
        lede('점수를 끌어올린 저빈도 어휘와, 사전에 없는 단어를 원고 위에 표시했습니다.')
        st.write('')

        paras, n_oov, n_adv, cut = render_manuscript(text)
        c1, c2 = st.columns([2.4, 1], gap='large')
        with c1:
            st.markdown(f'<div class="ms">{paras}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="legend">'
                f'<span><span class="mk-adv">저빈도 어휘</span> <b>{n_adv}</b>개'
                ' &nbsp;— 흔히 쓰이는 2,000단어 밖. 점수를 올립니다</span>'
                f'<span><span class="mk-oov">사전에 없는 단어</span> <b>{n_oov}</b>개'
                ' &nbsp;— 철자 오류이거나 고유명사입니다</span>'
                '</div>', unsafe_allow_html=True)
            if cut:
                st.markdown('<div class="card-n">글이 길어 앞부분만 표시했습니다. '
                            '점수는 전체 글로 계산했습니다.</div>',
                            unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card-k" style="margin-bottom:.75rem">'
                        '원고 정보</div>', unsafe_allow_html=True)
            for k, v, s in [
                    ('단어 수', f'{f["_Token"]:,}', '학생 평균은 432단어'),
                    ('서로 다른 단어', f'{f["_Type"]:,}',
                     f'전체의 {f["_Type"] / f["_Token"] * 100:.0f}%'),
                    ('문장 / 문단', f'{f["_num_sent"]} / {f["_num_para"]}',
                     f'문장당 평균 {f["mean_sent_length"]:.0f}단어'),
                    ('철자 의심 비율', f'{f["oov_ratio"] * 100:.1f}%',
                     f'{n_oov}개 단어'),
            ]:
                stat_card(k, v, s)
                st.write('')

        if f['_oov_words']:
            with st.expander(f'사전에 없는 단어 {len(f["_oov_words"])}개 따로 보기'):
                st.markdown('<div style="font-family:IBM Plex Mono, monospace;'
                            'font-size:.9rem;line-height:2;color:' + OXBLOOD + '">'
                            + ' · '.join(f['_oov_words']) + '</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="card-n" style="margin-top:.8rem">'
                            '이 도구는 오타를 고치지 않고 그대로 봅니다. '
                            '철자 실수 자체가 그 단어를 아직 확실히 익히지 못했다는 '
                            '신호이기 때문입니다.</div>', unsafe_allow_html=True)

        # ── 항목별 진단 ──────────────────────────────────
        st.markdown('<hr>', unsafe_allow_html=True)
        eyebrow('항목별 진단')
        st.markdown('## 여섯 갈래로 나눠 보면')

        RADAR = ['guiraud', 'MTLD_valid', 'clean_beyond_2k',
                 'mean_word_length', 'academic_suffix_ratio', 'oov_ratio']
        c1, c2 = st.columns([1, 1.1], gap='large')
        with c1:
            vals = [percentile(c, f[c]) for c in RADAR]
            labels = [LABEL[c] for c in RADAR]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[50] * len(RADAR) + [50], theta=labels + [labels[0]],
                fill='toself', name='또래 평균', line=dict(color=RULE, width=1),
                fillcolor='rgba(217,217,209,.45)', hoverinfo='skip'))
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]], fill='toself',
                name='이 글', line=dict(color=INK, width=2.5),
                fillcolor='rgba(31,46,77,.13)',
                customdata=[100 - v for v in vals] + [100 - vals[0]],
                hovertemplate='%{theta}<br>상위 %{customdata:.0f}%<extra></extra>'))
            fig.update_layout(
                polar=dict(bgcolor='rgba(0,0,0,0)',
                           radialaxis=dict(visible=True, range=[0, 100],
                                           ticksuffix='%', gridcolor=RULE,
                                           tickfont=dict(size=10, color=INK_SOFT)),
                           angularaxis=dict(gridcolor=RULE,
                                            tickfont=dict(size=11, color=INK))),
                title='또래 대비 위치 — 바깥쪽일수록 좋습니다')
            st.plotly_chart(style_fig(fig, 405, legend=True),
                            use_container_width=True, config=NO_BAR)

        with c2:
            imp = load_csv('feature_importance.csv', index_col=0)
            z = res['scaled'].iloc[0]
            contrib = (z[imp.index] * imp['Ridge 계수'])
            contrib = contrib.reindex(contrib.abs().sort_values().index).tail(8)
            fig = go.Figure(go.Bar(
                x=contrib.values, y=[LABEL.get(i, i) for i in contrib.index],
                orientation='h',
                marker_color=[SAGE if v > 0 else OXBLOOD for v in contrib.values],
                hovertemplate='%{y}<br>%{x:+.3f}<extra></extra>'))
            fig.add_vline(x=0, line_color=INK_SOFT, line_width=1)
            fig.update_layout(title='점수를 올린 것과 내린 것',
                              xaxis_title=None, yaxis_title=None)
            st.plotly_chart(style_fig(fig, 405), use_container_width=True,
                            config=NO_BAR)

        prof = {c: percentile(c, f[c]) for c in RADAR}
        best, worst = max(prof, key=prof.get), min(prof, key=prof.get)
        st.write('')
        c1, c2 = st.columns(2, gap='large')
        c1.markdown(
            f'<div class="verdict good"><div class="verdict-t">'
            f'가장 강한 부분 — {LABEL[best]}'
            f'<span class="pct">상위 {100 - prof[best]:.0f}%</span></div>'
            f'<div class="verdict-d">{TIP[best]}</div></div>',
            unsafe_allow_html=True)
        c2.markdown(
            f'<div class="verdict weak"><div class="verdict-t">'
            f'보완하면 좋을 부분 — {LABEL[worst]}'
            f'<span class="pct">상위 {100 - prof[worst]:.0f}%</span></div>'
            f'<div class="verdict-d">{TIP[worst]}</div></div>',
            unsafe_allow_html=True)

        st.write('')
        with st.expander('측정한 항목 15개 전부 보기'):
            st.dataframe(pd.DataFrame([
                {'항목': LABEL.get(c, c), '무엇을 재는가': TIP.get(c, ''),
                 '이 글의 값': round(f[c], 3),
                 '또래 대비': f'상위 {100 - percentile(c, f[c]):.0f}%'}
                for c in NUM_FEATURES]), hide_index=True,
                use_container_width=True)

    elif run:
        note('판독할 글이 없습니다. 영어로 쓴 글을 붙여넣고 다시 눌러 주세요.', 'warn')

# ══════════════════════════════════════════════════════════
#  점수를 만드는 것
# ══════════════════════════════════════════════════════════
elif page == '점수를 만드는 것':
    eyebrow('영향력')
    st.markdown('# 무엇이 어휘 점수를<br>만드는가', unsafe_allow_html=True)
    lede('열다섯 개 항목이 점수에 각각 얼마나 관여하는지, 그리고 값을 바꾸면 '
         '점수가 어떻게 움직이는지 직접 확인할 수 있습니다.')

    st.write('')
    imp = load_csv('feature_importance.csv', index_col=0)
    o = imp.sort_values('Ridge 계수')
    fig = go.Figure(go.Bar(
        x=o['Ridge 계수'], y=[LABEL.get(i, i) for i in o.index], orientation='h',
        marker_color=[SAGE if v > 0 else OXBLOOD for v in o['Ridge 계수']],
        hovertemplate='%{y}<br>%{x:+.3f}<extra></extra>'))
    fig.add_vline(x=0, line_color=INK_SOFT, line_width=1)
    fig.update_layout(title='항목별 영향력 — 초록은 올리는 방향, 적갈색은 내리는 방향',
                      xaxis_title=None, yaxis_title=None)
    st.plotly_chart(style_fig(fig, 465), use_container_width=True, config=NO_BAR)

    note('<b>어휘 다양성</b>과 <b>철자 정확성</b>이 나머지를 크게 앞섭니다. '
         '얼마나 길게 썼는지보다, 어떤 단어를 얼마나 정확하게 썼는지가 점수를 만듭니다.')

    st.markdown('<hr>', unsafe_allow_html=True)
    eyebrow('실험')
    st.markdown('## 값을 움직여 보기')
    lede('평균적인 글에서 항목 하나를 조정하면 점수가 어떻게 달라질까요. '
         '슬라이더를 옮기면 실제 모델이 다시 계산합니다.')
    st.write('')

    SLIDERS = ['guiraud', 'MTLD_valid', 'clean_beyond_2k',
               'mean_word_length', 'oov_ratio', 'len_z_in_prompt']
    med = TRAIN[NUM_FEATURES].median()
    cols = st.columns(3, gap='large')
    vals = med.copy()
    for i, c in enumerate(SLIDERS):
        lo, hi = TRAIN[c].quantile(.02), TRAIN[c].quantile(.98)
        with cols[i % 3]:
            vals[c] = st.slider(LABEL[c], float(lo), float(hi), float(med[c]),
                                (float(hi) - float(lo)) / 100, help=TIP.get(c))

    def ensemble(v):
        raw = pd.DataFrame([v[NUM_FEATURES]])
        logged = raw.copy()
        for c in ARTIFACTS['log_cols']:
            logged[c] = np.log1p(logged[c])
        sc = pd.DataFrame(ARTIFACTS['scaler'].transform(logged),
                          columns=NUM_FEATURES)
        m = FINAL_MODEL['members']
        return float(np.clip(np.mean([
            m['lasso'].predict(sc[FINAL_MODEL['lasso_features']])[0],
            m['rf'].predict(raw[FINAL_MODEL['tree_features']])[0],
            m['gb'].predict(raw[FINAL_MODEL['tree_features']])[0]]),
            *FINAL_MODEL['clip_range']))

    sim, base = ensemble(vals), ensemble(med)
    st.write('')
    c1, c2 = st.columns([1, 1.35], gap='large')
    with c1:
        delta = sim - base
        arrow = '↑' if delta > 0.005 else ('↓' if delta < -0.005 else '—')
        color = SAGE if delta > 0.005 else (OXBLOOD if delta < -0.005 else INK_SOFT)
        st.markdown(f"""
<div class="entry">
  <div class="entry-head">예상 점수<span class="entry-pos">실험값</span></div>
  <div style="margin-top:.75rem">
    <span class="entry-score">{sim:.2f}</span><span class="entry-of">/ 5.00</span>
  </div>
  <div class="entry-gloss">평균적인 글({base:.2f}점) 대비
    <b style="color:{color}">{arrow} {abs(delta):.2f}점</b></div>
</div>""", unsafe_allow_html=True)
    with c2:
        labels = [LABEL[c] for c in SLIDERS]
        pv = [percentile(c, vals[c]) for c in SLIDERS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[50] * len(SLIDERS) + [50], theta=labels + [labels[0]],
            fill='toself', name='평균', line=dict(color=RULE, width=1),
            fillcolor='rgba(217,217,209,.45)', hoverinfo='skip'))
        fig.add_trace(go.Scatterpolar(
            r=pv + [pv[0]], theta=labels + [labels[0]], fill='toself',
            name='설정값', line=dict(color=INK, width=2.5),
            fillcolor='rgba(31,46,77,.13)'))
        fig.update_layout(polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(range=[0, 100], ticksuffix='%', gridcolor=RULE,
                            tickfont=dict(size=10, color=INK_SOFT)),
            angularaxis=dict(gridcolor=RULE,
                             tickfont=dict(size=11, color=INK))))
        st.plotly_chart(style_fig(fig, 355, legend=True),
                        use_container_width=True, config=NO_BAR)

# ══════════════════════════════════════════════════════════
#  자료 살펴보기
# ══════════════════════════════════════════════════════════
elif page == '자료 살펴보기':
    eyebrow('자료')
    st.markdown('# 학생 3,000명의<br>글은 어떻게 생겼나', unsafe_allow_html=True)
    lede('이 도구가 배운 자료를 직접 살펴볼 수 있습니다. '
         '그래프는 마우스로 돌리고 확대할 수 있습니다.')

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 세 항목을 동시에 보기')
    opts = [c for c in NUM_FEATURES if c != 'flag_very_short']
    c1, c2, c3 = st.columns(3, gap='large')
    fx = c1.selectbox('가로', opts, index=opts.index('guiraud'),
                      format_func=lambda c: LABEL[c])
    fy = c2.selectbox('세로', opts, index=opts.index('oov_ratio'),
                      format_func=lambda c: LABEL[c])
    fz = c3.selectbox('높이', opts, index=opts.index('mean_word_length'),
                      format_func=lambda c: LABEL[c])

    SCALE = [[0, OXBLOOD], [.5, '#E2E0D8'], [1, INK]]
    samp = TRAIN.sample(min(1300, len(TRAIN)), random_state=42)
    fig = px.scatter_3d(samp, x=fx, y=fy, z=fz, color='점수',
                        color_continuous_scale=SCALE, opacity=.8)
    fig.update_traces(marker=dict(size=3.2, line=dict(width=0)))
    axis = dict(backgroundcolor=PAPER, gridcolor=RULE, zerolinecolor=RULE,
                showbackground=True, tickfont=dict(size=10, color=INK_SOFT))
    fig.update_layout(
        height=580, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans KR', color=INK),
        scene=dict(xaxis=dict(title=LABEL[fx], **axis),
                   yaxis=dict(title=LABEL[fy], **axis),
                   zaxis=dict(title=LABEL[fz], **axis)),
        coloraxis_colorbar=dict(title='점수', thickness=12, len=.6))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="card-n">점 하나가 학생 한 명의 글입니다. '
                '남색이 높은 점수, 적갈색이 낮은 점수. 드래그로 회전, 휠로 확대.</div>',
                unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 길게 쓰면 점수가 오를까')
    train_raw = load_csv('train.csv')
    if train_raw is not None:
        d = pd.DataFrame({'분량(단어 수)': train_raw['Token'],
                          '어휘 다양성': train_raw['guiraud'],
                          '점수': train_raw['Vocabulary']})
        c1, c2 = st.columns(2, gap='large')
        for col, (x, title) in zip([c1, c2], [
                ('분량(단어 수)', '분량과 점수'),
                ('어휘 다양성', '어휘 다양성과 점수')]):
            r = d[x].corr(d['점수'])
            fig = px.scatter(d, x=x, y='점수', opacity=.2,
                             color_discrete_sequence=['#B6BCC9'])
            fig.update_traces(marker=dict(size=5, line=dict(width=0)))
            b, a = np.polyfit(d[x], d['점수'], 1)
            xs = np.linspace(d[x].min(), d[x].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode='lines',
                                     line=dict(color=OXBLOOD, width=2.5),
                                     hoverinfo='skip'))
            fig.update_layout(title=f'{title} · 관련도 {r:+.2f}')
            col.plotly_chart(style_fig(fig, 355), use_container_width=True,
                             config=NO_BAR)
        note('<b>분량만으로는 점수의 7%밖에 설명되지 않습니다.</b> '
             '어휘 다양성 같은 질적 지표 네 개만 써도 27%까지 올라갑니다. '
             '길게 쓰는 것보다 어떤 단어를 쓰는지가 훨씬 중요합니다.')

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 점수대별로 항목이 어떻게 다른가')
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
        fig = px.box(long, x='점수대', y='값', color='점수대',
                     color_discrete_sequence=[OXBLOOD, '#C09A52', SAGE],
                     facet_col='항목', facet_col_wrap=3)
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(title=None)
        fig.for_each_annotation(lambda a: a.update(
            text=a.text.split('=')[-1],
            font=dict(family='Fraunces, Gowun Batang, serif', size=13, color=INK)))
        fig.update_traces(marker=dict(size=3, opacity=.45), line=dict(width=1.5))
        st.plotly_chart(style_fig(fig, 320 * ((len(pick) + 2) // 3), legend=True),
                        use_container_width=True, config=NO_BAR)
        st.markdown('<div class="card-n">점수대가 올라갈수록 상자가 한 방향으로 '
                    '움직이면, 그 항목이 점수와 관련이 있다는 뜻입니다.</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  정확도와 한계
# ══════════════════════════════════════════════════════════
elif page == '정확도와 한계':
    eyebrow('검증')
    st.markdown('# 이 판독을<br>어디까지 믿을 수 있나', unsafe_allow_html=True)
    lede('학습에 쓰지 않은 글 750편으로 확인한 결과입니다. '
         '잘 맞는 부분과 그렇지 않은 부분을 함께 적었습니다.')

    st.write('')
    c = st.columns(3, gap='large')
    with c[0]:
        stat_card('평균 오차', f'±{MAE:.2f}점', '채점 단위 0.5점 기준 약 0.7단계')
    with c[1]:
        stat_card('설명한 비율', f"{SUMMARY['test']['R²'] * 100:.0f}%",
                  '아무 정보도 안 쓰면 0%')
    with c[2]:
        stat_card('검증에 쓴 글', '750편', '학습에 한 번도 쓰지 않은 글')

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 그냥 찍는 것보다 얼마나 나은가')
    prog = pd.DataFrame([
        ['평균만 답하기', 0.4700, 0.0],
        ['글 길이만 보기', 0.4395, 7.2],
        ['핵심 항목 네 개', 0.3901, 26.6],
        ['이 도구 (항목 15개)', SUMMARY['test']['MAE'], SUMMARY['test']['R²'] * 100],
    ], columns=['방법', '평균 오차', '설명한 비율'])
    c1, c2 = st.columns(2, gap='large')
    for col, (y, title, color) in zip([c1, c2], [
            ('평균 오차', '평균 오차 — 낮을수록 좋습니다', OXBLOOD),
            ('설명한 비율', '설명력 — 높을수록 좋습니다', INK)]):
        fig = px.bar(prog, x=y, y='방법', orientation='h', text_auto='.3g',
                     color_discrete_sequence=[color])
        fig.update_traces(textfont=dict(family='IBM Plex Mono', size=11,
                                        color=INK_SOFT),
                          textposition='outside', cliponaxis=False)
        fig.update_layout(title=title, yaxis_title=None, xaxis_title=None)
        col.plotly_chart(style_fig(fig, 325), use_container_width=True,
                         config=NO_BAR)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 어디서 틀리는가')
    pred = load_csv('final_predictions.csv')
    if pred is not None:
        c1, c2 = st.columns([1.2, 1], gap='large')
        with c1:
            fig = px.scatter(pred, x='예측', y='실제', opacity=.25,
                             color_discrete_sequence=[INK])
            fig.add_shape(type='line', x0=1, y0=1, x1=5, y1=5,
                          line=dict(color=OXBLOOD, dash='dash', width=2))
            fig.update_traces(marker=dict(size=6, line=dict(width=0)))
            fig.update_layout(title='점이 대각선에 가까울수록 정확한 판독',
                              xaxis_range=[1, 5], yaxis_range=[1, 5])
            st.plotly_chart(style_fig(fig, 405), use_container_width=True,
                            config=NO_BAR)
        with c2:
            bias = pred.groupby('실제')['잔차'].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=bias['실제'], y=bias['잔차'],
                marker_color=[SAGE if v > 0 else OXBLOOD for v in bias['잔차']],
                hovertemplate='실제 %{x}점<br>%{y:+.2f}<extra></extra>'))
            fig.add_hline(y=0, line_color=INK_SOFT, line_width=1)
            fig.update_layout(title='점수대별 치우침',
                              xaxis_title='실제 점수', yaxis_title='실제 − 판독')
            st.plotly_chart(style_fig(fig, 405), use_container_width=True,
                            config=NO_BAR)

        note('<b>이 도구의 약점은 분명합니다.</b><br>'
             '아주 잘 쓴 글은 실제보다 낮게, 아주 못 쓴 글은 실제보다 높게 판독합니다. '
             '배운 글이 대부분 중간 점수대(3.0–3.5점)에 몰려 있어 양 끝을 거의 본 적이 '
             '없기 때문입니다. 모델을 바꿔서 해결되는 문제가 아니라 자료의 한계입니다. '
             '여섯 가지 개선 방법을 시도했지만 이 치우침은 그대로였습니다.', 'warn')

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 상황별 신뢰도')
    st.markdown(f"""
| 어떤 글인가 | 얼마나 믿을 수 있나 |
|---|---|
| 300단어 이상, 중간 점수대(2.5–4.0) | 비교적 정확합니다 (±{MAE:.2f}점) |
| 100–300단어의 짧은 글 | 점수가 다소 후하게 나옵니다 |
| 100단어 미만 | 신뢰도가 낮습니다. 참고만 하세요 |
| 아주 잘 썼거나 아주 못 쓴 글 | 중간 쪽으로 끌려갑니다 |
    """)
    st.write('')
    note('참고로, 사람 채점자끼리도 이 항목의 일치도가 높지 않습니다. '
         '일곱 개 채점 항목 가운데 가장 낮았습니다(Cohen\'s κ = .518). '
         '정답이 하나로 정해지는 과제가 아니라는 뜻입니다. 이 도구는 글을 고칠 방향을 '
         '찾는 참고 지표이지, 실제 채점을 대신하지 않습니다.')

# ══════════════════════════════════════════════════════════
#  이 도구에 대하여
# ══════════════════════════════════════════════════════════
else:
    eyebrow('배경')
    st.markdown('# 사전에 없는 단어를<br>고치지 않은 이유', unsafe_allow_html=True)
    lede('이 도구는 영어 학습자가 쓴 글의 어휘 점수를 예측합니다. '
         '점수만 내놓는 것이 아니라, 왜 그 점수인지 설명하는 것이 목표였습니다.')

    st.write('')
    c = st.columns(3, gap='large')
    with c[0]:
        stat_card('자료', 'ELLIPSE', '8–12학년 영어 학습자 에세이 3,747편')
    with c[1]:
        stat_card('예측 대상', 'Vocabulary', '1.0–5.0점 · 0.5점 단위 채점')
    with c[2]:
        stat_card('방법', '항목 15개', '원문에서 28개를 설계해 15개로 추림')

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 가장 어려웠던 부분')
    st.markdown("""
원본 자료에 있던 어휘 관련 정보는 네 개뿐이었고, 그마저도 대부분
**"얼마나 길게 썼는가"를 재고 있었습니다.**

채점자가 실제로 보는 **어휘의 수준**과 **철자 정확성**은 자료에 아예 없었습니다.
그래서 에세이 원문을 다시 읽어 그 축을 직접 만들어야 했습니다.
    """)
    st.write('')
    note('보통 텍스트 분석에서는 오타를 먼저 교정합니다. 이 프로젝트에서는 '
         '<b>오타를 그대로 뒀습니다.</b><br><br>'
         '예측하려는 것이 "어휘를 얼마나 익혔는가"이고, 철자 오류는 그 단어를 아직 '
         '확실히 익히지 못했다는 <b>관찰 가능한 증거</b>이기 때문입니다. 교정했다면 '
         '가장 중요한 신호 하나를 스스로 지우는 셈이 됐을 겁니다.<br><br>'
         '실제로 철자 정확성은 전체 항목 가운데 <b>영향력 2위</b>로 나왔습니다.')

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 알게 된 것 세 가지')
    st.write('')
    c1, c2, c3 = st.columns(3, gap='large')
    for col, (title, body) in zip([c1, c2, c3], [
        ('길이는 답이 아니다',
         '길게 쓰는 것만으로 설명되는 몫은 7%뿐이었습니다. '
         '어떤 단어를 쓰는지가 훨씬 중요합니다.'),
        ('모델보다 항목이 중요하다',
         '네 가지 모델의 성능 차이는 측정 오차 수준이었습니다. 성능을 만든 것은 '
         '알고리즘이 아니라 어떤 항목을 넣었는가였습니다.'),
        ('한계를 아는 것도 결과다',
         '여섯 가지 방법으로 개선해도 벽이 있었습니다. 그 벽이 어디서 오는지 밝힌 것이 '
         '이 프로젝트의 성과 중 하나입니다.'),
    ]):
        col.markdown(f'<div class="card"><div class="card-v" '
                     f'style="font-size:1.12rem;line-height:1.4">{title}</div>'
                     f'<div class="card-n" style="margin-top:.65rem;'
                     f'font-size:.85rem">{body}</div></div>',
                     unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('## 만든 과정')
    st.markdown("""
| 단계 | 한 일 |
|---|---|
| 기획 | 주제 선정, 가설 일곱 개 설계 |
| 전처리 | 정제, 언어 항목 28개 생성 |
| 탐색 | 분포·관련도·치우침 점검 |
| 항목 선택 | 28개에서 15개로 (겹치는 것 정리) |
| 모델링 | 네 가지 모델 비교 |
| 성능 향상 | 여섯 가지 방법 시도 후 세 모델 결합 |
    """)

    st.write('')
    st.markdown('<div class="card-n">자료 출처 — ELLIPSE Corpus '
                '(github.com/scrosseye/ELLIPSE-Corpus) · '
                '학습과 연구 목적으로 만들었으며 실제 채점을 대체하지 않습니다.</div>',
                unsafe_allow_html=True)
