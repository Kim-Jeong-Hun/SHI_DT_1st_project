# 대시보드 실행 가이드 (STEP 6)

## 1. 폴더 구조

```
프로젝트/
├─ app.py                     ← 평가·검증용 대시보드 (분석 과정 전체)
├─ for_users.py               ← 사용자용 대시보드 (인터랙티브 · 용어 최소화)
├─ feature_extraction.py      ← 원문 → 변수 15개 계산
├─ validate_extraction.py     ← 재계산값이 학습값과 맞는지 검증
├─ requirements.txt
├─ 01_eda.ipynb ~ 06_model_improvement.ipynb
└─ data/
   ├─ train.csv, test.csv
   ├─ X_train.csv, X_test.csv, X_train_scaled.csv, X_test_scaled.csv
   ├─ y_train.csv, y_test.csv
   ├─ preprocessor.joblib          ← scaler · encoder · 프롬프트 그룹 통계
   ├─ final_model.joblib           ← Lasso + RF + GB 앙상블
   ├─ selected_features.json, feature_selection_log.csv
   ├─ cv_results.csv, test_results.csv, model_results.json
   ├─ feature_importance.csv, final_predictions.csv
   ├─ final_test_results.csv, improvement_log.csv, final_summary.json
   └─ extraction_validation.json   ← validate_extraction.py 실행 결과
```

`app.py`와 `feature_extraction.py`는 **같은 폴더**에, `data/`는 그 **하위**에 두세요.

## 2. 설치

```bash
pip install -r requirements.txt
```

`wordfreq`는 필수입니다. 학습 시 K1/K2 빈도 밴드와 사전(상위 5만 단어)을
이 패키지로 만들었기 때문에, 없으면 기준이 달라져 예측이 어긋납니다.
`feature_extraction.py`는 이 경우 조용히 대체하지 않고 명시적으로 실패합니다.

## 3. 실행

```bash
python -m streamlit run app.py        # 평가·검증용
python -m streamlit run for_users.py  # 사용자용
```

브라우저에서 `http://localhost:8501`이 열립니다.
포트가 겹치면 `--server.port 8502`를 뒤에 붙이세요.

> Windows 관리 PC에서 `streamlit.exe`가 정책으로 차단되면
> 위처럼 `python -m streamlit` 형태로 실행하세요. `pip`도 `python -m pip`으로 씁니다.

### 두 앱의 차이

| | `app.py` | `for_users.py` |
|---|---|---|
| 대상 | 평가자 · 검토자 | 글을 쓰는 사람 |
| 내용 | 판정 기준, 제거 로그, 누수 검증, 부호 반전 등 과정 전체 | 점수와 그 이유, 고칠 방향 |
| 용어 | `guiraud`, VIF, R² 등 그대로 | 전부 한국어 이름 (`어휘 다양성` 등) |
| 그래프 | matplotlib 정적 | plotly 인터랙티브 (3D 산점도, 게이지, 레이더) |
| 특징 | 근거 추적 가능 | 슬라이더로 "이 항목을 올리면?" 실험 가능 |

발표 시연은 `for_users.py`, 질의응답 대비는 `app.py`를 띄워 두시면 편합니다.

## 4. 검증 스크립트 (선택)

```bash
python validate_extraction.py
```

test 750편에 대해 "원문에서 다시 계산한 값"과 "학습에 쓴 값"을 비교합니다.
`data/extraction_validation.json`을 갱신하며, 대시보드 ⑤번 탭 하단에 표시됩니다.

---

## 탭 구성

| 탭 | 내용 |
|---|---|
| ① 프로젝트 소개 | 주제, 파이프라인, 설계 가설 7개의 검증 결과 |
| ② 데이터 정의 | 파생변수 7개 그룹, 선택된 15개, 제거 규칙 R0~R5 |
| ③ 데이터 분석 | 타겟 분포, 길이 교란, 상관 히트맵 |
| ④ 모델 성능 | 베이스라인→최종 진행, 모델 4종 비교, 계수·중요도, 잔차, 개선 로그 |
| ⑤ 예측 시스템 | 에세이 입력 → 점수 + 기여도 분해 + 신뢰 범위 |

---

## 발표 시 짚을 지점

**⑤번 탭에서 예측을 시연한 뒤 ④번 탭으로 돌아가는 순서**를 권합니다.
"맞힌다"를 먼저 보여주고, "어떻게·어디까지 맞히는가"로 넘어가는 흐름입니다.

1. **③번 탭 — 길이 교란 산점도 2개 나란히**
   길이 단독 R² 0.0716 vs 핵심 파생변수 4개 R² 0.2661.
   "길게 쓰면 점수가 오른다"는 통념이 이 데이터에서는 성립하지 않습니다.

2. **⑤번 탭 — 기여도 분해 그래프**
   딥러닝이 못 하는 것이 이것입니다. 어떤 언어 특성이 점수를 밀어올렸는지 변수 단위로 보여줍니다.

3. **④번 탭 — 모델 4종 비교표**
   폴드 간 표준편차 ±0.044, 모델 간 최대 차이 0.007.
   성능을 만든 것은 알고리즘이 아니라 피처였습니다.

4. **④번 탭 — 구간별 편향 그래프**
   EDA 단계에서 분포만 보고 예고한 실패 양상이 그대로 재현됐습니다.
   실패를 감추지 않고 원인까지 규명한 지점입니다.

---

## 알려진 한계 (질문 대비)

**Q. R² 0.35면 낮은 것 아닌가?**
평균 예측 대비 RMSE 19.9% 감소, 길이 단독 대비 R² 4.9배입니다.
`Vocabulary`는 사람 채점자끼리의 일치도도 Cohen's κ = .518로 7개 항목 중 최하위입니다.
선행 연구의 다차원 모델이 보고한 Vocabulary 차원 Pearson .599(R² 환산 약 .36)와 근접합니다.

**Q. 왜 딥러닝을 안 썼나?**
BERT 계열은 성능이 더 높지만 어떤 언어 특성이 점수를 만들었는지 설명하지 못합니다.
이 프로젝트는 성능 일부를 내주고 설명력을 얻는 쪽을 선택했고,
④번 탭의 계수표가 그 대가로 얻은 것입니다.

**Q. 원문에서 다시 계산한 변수가 학습 때와 같은가?**
15개 중 14개는 상관 0.99 이상으로 일치합니다.
`MTLD_valid`만 상관 0.90입니다 — ELLIPSE 코퍼스의 MTLD 계산 규칙이 공개돼 있지 않아
표준 알고리즘으로 재구현했기 때문입니다.
최종 예측에 미친 영향은 평균 0.019점이며, 재계산 피처로 평가한 test 성능은
저장 피처 기준과 사실상 동일합니다. ⑤번 탭 하단에 실측값을 그대로 공개해 두었습니다.

**Q. 짧은 글을 넣으면 점수가 높게 나온다?**
맞습니다. `guiraud`는 길이 보정 지표이지만 완전히 길이 독립적이지는 않습니다(r(Token) = 0.38).
300토큰 미만 입력에는 앱이 경고를 표시합니다.
100토큰 미만은 MTLD가 정의되지 않아 대치값을 쓰므로 신뢰도가 더 낮습니다.
