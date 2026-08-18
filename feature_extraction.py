"""
feature_extraction.py
=====================
원문 에세이 하나에서 모델 입력 15개 변수를 계산합니다.

이 모듈이 필요한 이유
---------------------
학습에 쓴 `train.csv`는 ELLIPSE 코퍼스가 미리 계산해 둔 컬럼
(`Token`, `Type`, `MTLD`, `num_sent`, `num_para`)을 재료로 만들어졌습니다.
그러나 대시보드 사용자는 **원문만** 입력합니다.
따라서 그 원본 컬럼들을 직접 계산해야 합니다.

여기서 계산한 값이 코퍼스 원본과 다르면 예측이 학습 시점과 어긋납니다.
`validate_extraction.py`가 그 일치도를 측정하며, 결과는 대시보드에도 표시됩니다.

02_preprocessing.ipynb의 계산 규칙을 그대로 옮겼습니다.
바꾼 것은 "코퍼스 컬럼을 읽는 대신 직접 센다"는 부분뿐입니다.
"""

import re
from collections import Counter

import numpy as np

# ── 02 노트북과 동일한 토큰화 규칙 ────────────────────────────
# 영문자와 내부 아포스트로피만 토큰으로 인정합니다 (don't → don't).
# 숫자와 기호는 어휘력 측정 대상이 아니므로 제외합니다.
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# 02 [4-B]와 동일한 학술 접미사 목록
ACADEMIC_SUFFIXES = ('tion', 'sion', 'ment', 'ity', 'ance', 'ence',
                     'ism', 'ize', 'ise', 'ate', 'ous', 'ive', 'ical', 'able')

MTLD_MIN_TOKEN = 100      # 02 [2-3]: MTLD 정의상 최소 요구 토큰 수
# McCarthy & Jarvis(2010)의 표준 임계값은 0.72입니다.
# 그러나 ELLIPSE 코퍼스의 MTLD 컬럼이 어떤 구현으로 계산됐는지는 공개돼 있지 않고,
# 0.72로 재계산하면 값이 체계적으로 커집니다(평균 53.5 vs 코퍼스 45.1).
# train 600편으로 임계값을 훑어 본 결과 0.75에서 척도가 일치했으므로(비율 1.02)
# 학습 데이터와 같은 척도를 맞추기 위해 0.75를 사용합니다.
# validate_extraction.py가 이 선택의 결과를 실측으로 보고합니다.
MTLD_TTR_THRESHOLD = 0.75

_BANDS = {}


def _load_bands():
    """K1/K2 빈도 밴드와 사전을 준비합니다 (최초 1회만)."""
    if _BANDS:
        return _BANDS
    try:
        from wordfreq import top_n_list
        k1 = set(top_n_list('en', 1000))
        k2 = set(top_n_list('en', 2000)) - k1
        known = set(top_n_list('en', 50000))
        _BANDS.update(K1=k1, K2=k2, KNOWN=known, source='wordfreq')
    except ImportError:
        # 02 노트북은 wordfreq를 사용했습니다. 없으면 학습 시점과 기준이 달라져
        # 예측이 어긋나므로, 조용히 대체하지 않고 명시적으로 실패시킵니다.
        raise ImportError(
            'wordfreq 패키지가 필요합니다. 학습 시 사용한 빈도 기준과 '
            '동일해야 예측이 유효합니다.\n  pip install wordfreq')
    return _BANDS


def tokenize(text):
    """02 [4-0]과 동일. 소문자 토큰 리스트를 반환합니다."""
    return TOKEN_RE.findall(str(text).replace('__PII__', ' ').lower())


def count_syllables(word):
    """
    02 [4-B]와 동일한 음절 추정 규칙.
    모음군 하나를 1음절로 세고, 발음되지 않는 어말 e를 보정합니다.
    """
    w = word.lower().strip("'")
    if not w:
        return 0
    n = len(re.findall(r'[aeiouy]+', w))
    if w.endswith('e') and n > 1 and not w.endswith(('le', 'ee', 'ye')):
        n -= 1
    return max(n, 1)


def _mtld_one_direction(tokens, threshold=MTLD_TTR_THRESHOLD):
    """
    한 방향 MTLD factor count.

    토큰을 앞에서부터 훑으며 누적 TTR을 계산하고,
    TTR이 임계값 아래로 떨어지면 factor 하나를 세고 초기화합니다.
    남은 구간은 (1 - TTR) / (1 - threshold)로 부분 factor를 인정합니다.
    """
    factors = 0.0
    types = set()
    n = 0
    for t in tokens:
        n += 1
        types.add(t)
        ttr = len(types) / n
        if ttr <= threshold:
            factors += 1
            types, n = set(), 0
    if n > 0:
        ttr = len(types) / n
        remainder = (1 - ttr) / (1 - threshold) if threshold < 1 else 0.0
        factors += max(min(remainder, 1.0), 0.0)
    return len(tokens) / factors if factors > 0 else float(len(tokens))


def compute_mtld(tokens):
    """정방향·역방향 평균 (McCarthy & Jarvis 2010의 표준 구현)."""
    if len(tokens) < 2:
        return 0.0
    return (_mtld_one_direction(tokens)
            + _mtld_one_direction(list(reversed(tokens)))) / 2


def split_sentences(text):
    """마침표·물음표·느낌표 기준. 02 [4-E] sentence_variation과 동일한 규칙."""
    return [s for s in re.split(r'[.!?]+', str(text)) if s.strip()]


def split_paragraphs(text):
    """빈 줄 또는 줄바꿈 기준. 내용이 있는 덩어리만 셉니다."""
    parts = [p for p in re.split(r'\n\s*\n|\n', str(text)) if p.strip()]
    return parts if parts else [str(text)]


def extract_features(text, prompt=None, artifacts=None):
    """
    원문 → 모델 입력 변수 dict.

    Parameters
    ----------
    text : str
        에세이 원문
    prompt : str or None
        과제 주제. `len_z_in_prompt` 계산에 사용합니다.
        None이거나 학습 시 없던 주제면 전체 통계로 대체합니다.
    artifacts : dict or None
        `preprocessor.joblib`에서 로드한 객체.
        `token_prompt_stats`(프롬프트별 Token 평균·표준편차)와
        `token_global`(전체 평균·표준편차)이 필요합니다.

    Returns
    -------
    dict : 변수명 → 값. 중간 계산값(`Token`, `Type`, `MTLD` 등)도 함께 담아
           대시보드에서 사용자에게 근거로 보여줄 수 있게 합니다.
    """
    bands = _load_bands()
    K1, K2, KNOWN = bands['K1'], bands['K2'], bands['KNOWN']

    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        raise ValueError('토큰이 하나도 추출되지 않았습니다. 영문 텍스트를 입력하세요.')

    counter = Counter(tokens)
    n_type = len(counter)

    # ── 원본 코퍼스가 제공하던 컬럼들을 직접 계산 ──────────────
    sents = split_sentences(text)
    paras = split_paragraphs(text)
    num_sent = max(len(sents), 1)
    num_para = max(len(paras), 1)
    mtld = compute_mtld(tokens)

    # ── A. 어휘 다양성 (02 [4-A]) ─────────────────────────────
    eps = 1e-9
    guiraud = n_type / np.sqrt(n + eps)
    herdan_c = np.log(n_type + eps) / np.log(n + eps)

    # MTLD는 100토큰 미만에서 정의되지 않습니다 (02 [2-3]).
    # 학습 시에는 무효값을 유효 구간의 최솟값으로 보수적 대치했으므로
    # 여기서도 같은 값을 사용합니다.
    flag_very_short = int(n < MTLD_MIN_TOKEN)
    if flag_very_short and artifacts and 'mtld_fill_value' in artifacts:
        mtld_valid = artifacts['mtld_fill_value']
    else:
        mtld_valid = mtld

    # ── B. 어휘 수준 (02 [4-B]) ───────────────────────────────
    k1 = sum(t in K1 for t in tokens)
    k2 = sum(t in K2 for t in tokens)
    lengths = [len(t) for t in tokens]
    sylls = [count_syllables(t) for t in tokens]

    k2_ratio = k2 / n
    beyond_2k_ratio = (n - k1 - k2) / n
    mean_word_length = float(np.mean(lengths))
    mean_syllables = float(np.mean(sylls))
    academic_suffix_ratio = sum(t.endswith(ACADEMIC_SUFFIXES) for t in tokens) / n

    # ── D. 정확성 (02 [4-D]) ──────────────────────────────────
    oov_ratio = sum(t not in KNOWN for t in tokens) / n
    hapax_ratio = sum(1 for v in counter.values() if v == 1) / n_type

    # 04 [6-1]: 고급 어휘 성분에서 철자 오류 성분을 분리
    clean_beyond_2k = beyond_2k_ratio - oov_ratio

    # ── E. 구문 (02 [4-E]) ────────────────────────────────────
    mean_sent_length = n / num_sent
    sent_per_para = num_sent / num_para
    if len(sents) < 2:
        sent_length_std = 0.0
    else:
        sent_length_std = float(np.std([len(TOKEN_RE.findall(s.lower())) for s in sents]))

    # ── F. 과제 상대화 (04 [6-3], train 기준 재계산본) ────────
    len_z = 0.0
    if artifacts is not None:
        gm, gs = artifacts.get('token_global', (n, 1.0))
        stats = artifacts.get('token_prompt_stats')
        m, s = gm, gs
        if prompt is not None and stats is not None and prompt in stats.index:
            m = stats.loc[prompt, 'mean']
            s = stats.loc[prompt, 'std']
            if not np.isfinite(s) or s == 0:
                s = gs
        len_z = (n - m) / s if s else 0.0

    return {
        # 모델 입력 15개
        'MTLD_valid': float(mtld_valid),
        'guiraud': float(guiraud),
        'herdan_c': float(herdan_c),
        'k2_ratio': float(k2_ratio),
        'mean_word_length': mean_word_length,
        'mean_syllables': mean_syllables,
        'academic_suffix_ratio': float(academic_suffix_ratio),
        'oov_ratio': float(oov_ratio),
        'hapax_ratio': float(hapax_ratio),
        'mean_sent_length': float(mean_sent_length),
        'sent_per_para': float(sent_per_para),
        'sent_length_std': sent_length_std,
        'flag_very_short': flag_very_short,
        'clean_beyond_2k': float(clean_beyond_2k),
        'len_z_in_prompt': float(len_z),
        # 참고용 중간값 — 사용자에게 근거로 보여줍니다
        '_Token': n,
        '_Type': n_type,
        '_MTLD_raw': float(mtld),
        '_num_sent': num_sent,
        '_num_para': num_para,
        '_beyond_2k_ratio': float(beyond_2k_ratio),
        '_oov_words': sorted({t for t in counter if t not in KNOWN})[:30],
    }


def transform_for_models(feats, artifacts, final_model):
    """
    extract_features 결과를 두 모델 계열의 입력 형태로 변환합니다.

    선형(Lasso) : log1p 변환 → StandardScaler → 학습 시 컬럼 순서
    트리(RF/GB) : 원값 그대로

    변환기는 반드시 학습 시점에 저장한 객체를 씁니다.
    코드를 다시 짜면 미세한 차이가 생기고, 그 차이가 예측을 어긋나게 합니다.
    """
    import pandas as pd

    num_features = artifacts['selected_features']
    log_cols = artifacts['log_cols']
    scaler = artifacts['scaler']

    raw = pd.DataFrame([{c: feats[c] for c in num_features}])

    logged = raw.copy()
    for c in log_cols:
        logged[c] = np.log1p(logged[c])
    scaled = pd.DataFrame(scaler.transform(logged),
                          columns=num_features)

    return {
        'lasso': scaled[final_model['lasso_features']],
        'tree': raw[final_model['tree_features']],
        'raw': raw,
        'scaled': scaled,
    }


def predict(text, prompt, artifacts, final_model):
    """원문 → 예측 점수. 앙상블 3개 모델의 개별 예측도 함께 반환합니다."""
    feats = extract_features(text, prompt, artifacts)
    X = transform_for_models(feats, artifacts, final_model)

    members = final_model['members']
    parts = {
        'Lasso': float(members['lasso'].predict(X['lasso'])[0]),
        'RandomForest': float(members['rf'].predict(X['tree'])[0]),
        'GradientBoosting': float(members['gb'].predict(X['tree'])[0]),
    }
    lo, hi = final_model['clip_range']
    score = float(np.clip(np.mean(list(parts.values())), lo, hi))

    return {'score': score, 'members': parts, 'features': feats,
            'scaled': X['scaled']}
