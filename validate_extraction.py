"""
validate_extraction.py
======================
`feature_extraction.py`가 계산한 값이 학습에 쓴 값과 일치하는지 검증합니다.

왜 필요한가
-----------
학습 데이터의 `Token`, `Type`, `MTLD`, `num_sent`, `num_para`는
ELLIPSE 코퍼스가 제공한 값입니다. 그 계산 규칙은 공개돼 있지 않습니다.
대시보드는 원문에서 이 값들을 직접 계산하므로, 규칙이 다르면
사용자에게 보여주는 예측이 학습 시점과 다른 척도 위에서 나옵니다.

이 스크립트는 test 750편에 대해 재계산값과 저장값을 비교하고,
결과를 `data/extraction_validation.json`에 남깁니다.
대시보드는 그 값을 읽어 사용자에게 신뢰 구간을 함께 표시합니다.
"""

import json
import joblib
import numpy as np
import pandas as pd

from feature_extraction import extract_features

TARGET = 'Vocabulary'
NUM = json.load(open('data/selected_features.json', encoding='utf-8'))['numeric']
artifacts = joblib.load('data/preprocessor.joblib')

raw = pd.read_csv('data/test.csv')
stored = pd.read_csv('data/X_test.csv')

print(f'검증 대상: test {len(raw)}편')
rows = []
for i, r in raw.iterrows():
    try:
        rows.append(extract_features(r['full_text'], r['prompt'], artifacts))
    except Exception as e:
        rows.append({c: np.nan for c in NUM})
recomputed = pd.DataFrame(rows)

report = []
for c in NUM:
    a, b = stored[c], recomputed[c]
    ok = a.notna() & b.notna()
    corr = a[ok].corr(b[ok])
    # 표준편차 단위 평균 절대 오차 — 변수마다 스케일이 달라 그대로 비교할 수 없습니다
    nmae = (a[ok] - b[ok]).abs().mean() / a[ok].std() if a[ok].std() else np.nan
    report.append({'변수': c, '상관': round(corr, 4),
                   '정규화 MAE': round(nmae, 4),
                   '판정': '일치' if corr > 0.99 and nmae < 0.1 else
                          ('유사' if corr > 0.95 else '불일치')})

rep = pd.DataFrame(report).sort_values('상관')
print(rep.to_string(index=False))

# 최종 성능 영향 — 재계산 피처로 예측하면 성능이 얼마나 떨어지는가
from sklearn.metrics import mean_absolute_error, r2_score
fm = joblib.load('data/final_model.joblib')
scaler, log_cols = artifacts['scaler'], artifacts['log_cols']

def to_scaled(df):
    d = df[NUM].copy()
    for c in log_cols:
        d[c] = np.log1p(d[c])
    return pd.DataFrame(scaler.transform(d), columns=NUM)

y = pd.read_csv('data/y_test.csv').iloc[:, 0]


def ensemble_pred(df_raw):
    s = to_scaled(df_raw)
    p = (fm['members']['lasso'].predict(s[fm['lasso_features']])
         + fm['members']['rf'].predict(df_raw[fm['tree_features']])
         + fm['members']['gb'].predict(df_raw[fm['tree_features']])) / 3
    return np.clip(p, *fm['clip_range'])


p_stored = ensemble_pred(stored)
valid = recomputed[NUM].notna().all(axis=1)
p_recomp = ensemble_pred(recomputed[valid])

print(f'\n저장 피처   : MAE {mean_absolute_error(y, p_stored):.4f} / '
      f'R² {r2_score(y, p_stored):.4f}')
print(f'재계산 피처 : MAE {mean_absolute_error(y[valid], p_recomp):.4f} / '
      f'R² {r2_score(y[valid], p_recomp):.4f}')
print(f'예측값 상관 : {np.corrcoef(p_stored[valid.values], p_recomp)[0, 1]:.4f}')
print(f'예측값 평균 절대차 : {np.abs(p_stored[valid.values] - p_recomp).mean():.4f}점')

out = {
    'n_validated': int(valid.sum()),
    'per_feature': rep.to_dict('records'),
    'stored_mae': float(mean_absolute_error(y, p_stored)),
    'stored_r2': float(r2_score(y, p_stored)),
    'recomputed_mae': float(mean_absolute_error(y[valid], p_recomp)),
    'recomputed_r2': float(r2_score(y[valid], p_recomp)),
    'pred_corr': float(np.corrcoef(p_stored[valid.values], p_recomp)[0, 1]),
    'pred_mean_abs_diff': float(np.abs(p_stored[valid.values] - p_recomp).mean()),
}
with open('data/extraction_validation.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('\n저장: data/extraction_validation.json')
