# Programming Plan (v0.6.4)

## 1. Overview

본 문서는 `algorithm_spec.md`(v0.6.4)의 설계를 코드로 구현하기 위한 종합 계획이다. 본 버전의 핵심 목표는 다음과 같다.

1. **블록 단위 수렴 기반 학습 종료 도입**: 고정 epoch(50) 폐지. 학습을 50 epoch 블록으로 반복 수행하며, 블록 종료 시마다 수렴 판독을 수행하고 **기준 1(수렴 추세)과 기준 2(진동 진폭)가 동시에 만족되면 학습을 종료**한다. 안전 상한은 **총 1000 epoch**이다. (`algorithm_spec.md` Section 2.2.1)
2. **수렴 판독 지표(Section 3.4) 구현**: 블록별 판독 레코드(만족/불만족 지표, 기준 3 하한 이탈 로그, 기준 1 상승 이탈 이벤트 로그)를 `data/log/convergence_report.txt`에 누적 기록하고 종료 시 요약을 덧붙인다.
3. **시간 지표 갱신 (Section 3.2)**: 기존 "10 epoch 간격 블록 시간 기록"을 유지하고, 종료 시 **총 학습 소요 시간 및 1 epoch당 평균 시간**을 추가 기록한다.

### 1.1 미뤄진 항목 (v0.6.5 방향성 예정)

- **데이터 수집 및 전처리 알고리즘 수정** (원본 데이터 양 부족·웹캠/사진 도메인 간극 대응)
- **Train/Validation 분할 및 Validation Loss 곡선 지표**
- **과적합 판정(F) 지표 및 최적 epoch 가중치 저장 정책**

위 항목은 이번 버전 구현에서 다루지 않는다. 데이터 규격(30+30장), 6배 증강 구성 그대로 유지하며 가중치 저장은 **학습 종료 시점(현행 방식)** 이다.

## 2. Repository Structure

```text
.
├── algorithm_spec.md
├── ai_rule.md
├── programming_plan.md
├── README.md
├── CHANGELOG.md
├── report.md
├── requirements.txt
├── data/                    # .gitignore
│   ├── target/              # 특정 인물 사진 (Label: 1.0) — 30장 (변경 없음)
│   ├── non_target/          # 타인/비타겟 사진 (Label: 0.0) — 30장 (변경 없음)
│   ├── weights.pth          # 학습 종료 시점 가중치
│   └── log/                 # 지표 산출물 저장소
│       ├── train_time_log.txt
│       ├── loss_curve.png
│       └── convergence_report.txt   # 신규 (Section 3.4)
├── model.py
├── face_detector.py
├── pipeline.py
├── trainer.py
├── convergence.py           # 신규: 수렴 판독/판정 모듈
├── train_dataset.py
└── main.py
```

## 3. Dependencies & Libraries

**변경 없음.** (torch, torchvision, mediapipe, opencv-python, numpy, matplotlib 유지. 신규 의존성 없음 — 수렴 판독 통계는 numpy 기본 산술로 구현)

## 4. Module & Class Specifications

### 4.1 `face_detector.py` — `FaceDetector`
**변경 없음.** (v0.6.3과 동일)

### 4.2 `model.py` — `SimpleCNN`
**변경 없음.**

### 4.3 `pipeline.py` — `FaceRecognitionPipeline`
**변경 없음.**

### 4.4 `trainer.py` — `ModelTrainer`
**변경 없음.** Optimizer: SGD(lr=1e-3, momentum=0), Loss: `nn.BCELoss()`

### 4.5 신규 `convergence.py` — 수렴 판독/판정 모듈

`algorithm_spec.md` Section 3.4의 판독 절차를 **순수 함수(입력→출력, 상태 없음)** 로 구현한다. 종료 통제(반복/멈춤)는 `train_dataset.py`의 루프에서 수행하므로 본 모듈은 판독·판정 연산만 담당한다.

**상수 (Section 3.4 문턱값):**

```python
THETA_R = 0.05      # 기준 1 상대 기울기 문턱값 (θ₁)
THETA_PI = 0.10     # 기준 2 상대 진폭 문턱값 (θ₂)
```
   - 블록 길이(50)는 학습 루프의 소속 상수(BLOCK_EPOCHS)이므로 본 모듈에 두지 않는다. 판독은 창(리스트) 입력만으로 수행된다.

**함수 명세:**

1. `block_stats(losses: list[float]) -> dict`
   - 입력: 블록(직전 50 epoch)의 epoch별 학습 Loss 값
   - 산출: `m_first`(전반 25 epoch 평균), `m_second`(후반 25 epoch 평균), `m`(전체 평균), `std`(σ), `max`, `min`
2. `judge_block(k: int, losses: list[float]) -> dict`
   - 블록 순번 $k$와 분석 창 $W_k$를 받아 판독 레코드 1건을 산출한다. 필드:
   - 표기 규칙: 구현상 $k$는 0-based 인덱스로 취급하며, 레코드 출력·블록 번호는 1-based($k+1$)로 변환한다 (스펙의 "k번째 블록" 표기와 일치시킴).
     - `r = (m_first - m_second) / m_first`
     - `c1 = |r| <= THETA_R` (만족/불만족)
     - `rising_event = (r < -THETA_R)` — 기준 1 상승 이탈 이벤트 (불만족 판정과 동시에 부가 로깅)
     - `pi = (max - min) / m`
     - `c2 = pi <= THETA_PI` (만족/불만족)
     - `dip_threshold = m - 3*std`
     - `dips = [(전역 epoch 번호, Loss 값), ...]` — `L(e) < m - 3σ` 를 만족하는 epoch 전부 (기준 3, 로그 전용)
     - `stop = c1 and c2` — 종료 조건 (기준 3 미포함)
     - `epoch_range = (k*len(losses)+1, (k+1)*len(losses))` — 레코드 표기용 전역 epoch 범위 (창 길이로부터 산출)
3. `format_record(record: dict) -> str`
   - 레코드를 텍스트 1줄(또는 소수 줄)로 직렬화. 사람이 읽는 텍스트 형식.

**방어 처리:** 분모 0 오류 방지 — $m_{\text{first}} = 0$이면 $r = 0$으로, $m = 0$이면 $\Pi = 0$으로 처리한다(실제 BCE 학습에서는 발생하지 않음). 문턱값 경계 비교는 부동소수 마진 `EPS = 1e-9`를 적용한다($|r| \le \theta_1 + \epsilon$, $\Pi \le \theta_2 + \epsilon$, $r < -\theta_1 - \epsilon$; 스펙의 부등호 시맨틱은 보존됨).

### 4.6 `train_dataset.py` — 주요 변경

**변경 항목:**

1. **상수 재정의 (Section 2.2.1 반영)**:

```python
BATCH_SIZE = 9            # Section 2.2.1: Train Batch
BLOCK_EPOCHS = 50         # Section 2.2.1: 50 epoch 단위 블록
SAFETY_CAP = 1000         # Section 2.2.1: 안전 상한 (총 학습 epoch)
LEARNING_RATE = 1e-3      # Section 2.2.1
```

   - 기존 `EPOCHS = 50` 상수 폐지. BLOCK_EPOCHS·SAFETY_CAP은 `train_dataset.py`에 직접 정의하되, 문턱값(THETA_R, THETA_PI)은 `convergence.py`에서 정의한 것을 import하여 사용.

2. **학습 루프 재구성 (고정 횟수 → while 반복형)**:

```python
while True:
    # 블록 수행: 50 epoch 학습 (기존 미니배치 루프/ModelTrainer 호출 로직 그대로)
    # 매 epoch의 학습 Loss를 block_losses와 loss_history(전역 누적)에 저장
    # 블록 판독: result = judge_block(k, block_losses)
    # 리포트 레코드 누적: convergence_report 텍스트 및 콘솔 출력
    if result["stop"]:
        종료 사유 = "수렴 조건 만족 (기준 1 & 2)"
        break
    if (k+1) * BLOCK_EPOCHS >= SAFETY_CAP:   # k = 완료한 블록 수 - 1
        종료 사유 = "안전 상한 도달"
        break
```

   - 데이터 로드("data/target"·"data/non_target"), 6배 증강, DataLoader(batch=9, shuffle), `ModelTrainer.train_step()` 사용은 **변경 없음**.
   - 1 epoch = 40 iterations (360 ÷ 9, 변경 없음).
   - Console 표기: 매 epoch `Epoch {n}, Loss: ...` 유지, 블록 말미마다 판독 요약 1줄 출력.
   - 안전 상한 도달 판정: 1000 = 20블록 정수 배수이므로 블록 경계에서 정확히 발생 (부분 블록 없음).

3. **시간 측정 (Section 3.2)**:

   - 매 10 epoch 구간마다 소요 시간 기록(기존 방식 유지; `LOG_INTERVAL = 10`).
   - 학습 종료 후 동일 파일에 선택 항목 추가:
     - `Total time: xxx.xx s` (총 학습 진행 과정의 소요 시간)
     - `Average time per epoch: x.xx s` (총 epoch에 대한 평균)
   - 최종 블록이 10의 배수 경계에서 끝나므로(50 단위) 누락 구간은 발생하지 않는다. 다만 종료 epoch가 10의 배수가 아닌 구조로 바뀌는 예외 상황을 대비해, 마지막 부분 구간도 총 시간에 포함하도록 구현한다.

4. **수렴 리포트 (Section 3.4)**:
   - `data/log/convergence_report.txt`에 매 블록 판독 레코드 누적, 학습 종료 시 종료 요약 덧붙여 저장.
   - 레코드 내용(블록 순번/epoch 범위, r, C1 만족·불만족, 상승 이탈 이벤트(블록/r), Π, C2 만족·불만족, C3 발생 epoch·Loss 목록 또는 '없음', 계속 학습/종료 판정, 종료 요약(총 학습 epoch, 종료 사유)).
   - 기존 `train_time_log.txt`와 동일하게 **실행 시 덮어쓰기**.
   - 포맷 예시:
     ```
     [block 1 | epoch 1-50]   r=+0.1234  C1: 불만족   Π=0.0312  C2: 불만족   C3: 없음                  → 계속 학습
     [block 2 | epoch 51-100] r=-0.0721  C1: 불만족   Π=0.0288  C2: 만족     C3: 발생 (epoch 87, 0.4123) → 계속 학습
     ```
     (실제 포맷 공학은 구현 시 조정 가능; 내용 필드는 위와 같이 포함)
   - 종료 요약(추가 줄):
     ```
     [Termination] total_epochs=350, reason=수렴 조건 만족 (기준 1 & 2)
     ```

5. **Loss 곡선 (Section 3.3)**:
   - 기존 방식 유지: 전체 epoch의 학습 Loss를 수집하고, **학습 종료 시 1회** 생성하여 `data/log/loss_curve.png` 저장.
   - x축=epoch(1..N), y축=BCE Loss. Train 곡선만 포함(Validation 곡선은 v0.6.5 예정 항목).
   - `matplotlib.use("Agg")` 백앤드 유지.

6. **가중치 저장**:
   - 학습 종료 시 `torch.save(model.state_dict(), "data/weights.pth")` (현행 유지; 최적 epoch 저장은 v0.6.5 재논의).

### 4.7 `main.py`
**변경 없음.** (v0.6.3에서 경로 수정 반영 완료: `weights_path="data/weights.pth"`)

## 5. Step 4 검증 계획 (안내)

구현 완료 후 Step 4에서 다음을 검증한다.

1. **판독 로직 단위 검증 (합성 데이터)**: 새 의존성 없이 짧은 검증 코드로 다음 사례를 점검한다.
   - 평탄·소진폭 합성 곡선 → 기준 1·2 만족 → 종료 판정
   - 지속 하강 합성 곡선 → 기준 1 불만족 → 계속 학습
   - 후반부 5% 이상 상승 곡선 → 기준 1 불만족 + 상승 이탈 이벤트 로깅
   - 진동 폭 큰 곡선 → 기준 2 불만족
   - 급락 이탈 포함 곡선 → 기준 3 이벤트 (epoch 번호·Loss 값) 로깅
2. **실제 학습 실행**: 실제 데이터(30+30장)로 실행하여 블록 반복 동작, 종료 경로(수렴 만족 종료), 리포트·시간 로그·그래프 산출물 확인.
3. **안전 상한 경로 검증**: `SAFETY_CAP`을 일시적으로 낮춘 값(예: 150)으로 실행하여 "안전 상한 도달" 종료 경로를 확인 후 원래 값(1000)으로 복구.
4. **추론 재확인**: 학습 완료된 `data/weights.pth`로 `main.py` 실시간 추론 정상 동작 확인.

## 6. Edge Cases Handling

- **디렉터리 부재**: `data/`, `data/log/`가 없을 때 자동 생성 (기존 유지)
- **얼굴 미검출** (기존): Stage 2 연산 생략, 확률 0.0 반환 (추론 측)
- **matplotlib 백엔드**: `Agg` 강제 유지
- **수렴 미도달 상태의 상한 종료**: "안전 상한 도달"을 종료 사유로 명시 기록 (수렴 실패 자체를 관찰 결과로 남김)
- **$m_{\text{first}} = 0$ 방어**: 발생하지 않지만 방어 코드 포함
- **산출물 overwrite**: 3개 산출물 모두 덮어쓰기 (v0.6.3 train_time_log와 동일 정책)

## 7. Implementation Notes

- **스펙-코드 상수 동기화**: Batch 9 / 블록 50 / 상한 1000 / θ₁=0.05 / θ₂=0.10은 `algorithm_spec.md` Section 2.2.1, 3.4와 반드시 일치. 지시 변경 시 양쪽 동시 수정.
- **난수 시드**: 고정하지 않음 (결정: 미도입). 실행별 결과 차이는 관찰 대상으로 유지.
- **판독 모듈 특성**: 본 모듈(`convergence.py`)은 상태 없는 순수 함수이므로, 테스트 가능성이 높다. 종료 통제는 학습 루프에만 존재.
- **Section 3.1 파라미터 수**: 산출 자체는 개발자가 1회 계산 후 README "프로그램 개요" 절에 수동 기입 (Step 5 처리; v0.6.3 값과 동일할 예정 — 모델 구조 변경 없음).

## 8. Step 5 이관 사항 (참고)

- README.md 프로그램 개요 갱신 (지표 체계에 수렴 판독 추가 반영)
- CHANGELOG.md v0.6.4 업데이트 내용 작성 및 "v0.6.5 예정 항목"(데이터 수집·전처리, Train/Val 지표, 과적합 판정) 명기
   - 개발 일지 영역은 Architect 직접 작성
