# Report: v0.6.3 구현 보고서

## 1. 개요

본 보고서는 v0.6.3 개발의 Step 4(코드 구현 및 테스트) 결과를 정리한 문서이다. 본 버전은 `algorithm_spec.md` Section 3에 명시된 **3개 지표(파라미터 수, epoch당 소요 시간, BCE 손실 곡선)** 를 산출하는 체계를 도입하고, 하이퍼파라미터를 `algorithm_spec.md`에 직접 명시하는 것을 목표로 한다. 내부 알고리즘(SimpleCNN, MediaPipe, SGD)은 v0.6.2와 동일하게 유지한다.

## 2. 구현 범위

`programming_plan.md`에 명시된 다음 사항을 구현하였다.

- `train_dataset.py`: Batch=9, Epoch=50, LR=1e-3 반영 / `data/weights.pth` 저장 / `data/log/`에 지표 산출물 저장 / matplotlib Loss 곡선 생성
- `main.py`: `weights_path="data/weights.pth"`로 경로 수정
- `requirements.txt`: `matplotlib` 추가
- `.gitignore`: `weights.pth` 라인 제거 (기존 루트 가중치 파일 정리 반영)
- 변경 없음: `face_detector.py`, `model.py`, `pipeline.py`, `trainer.py`

## 3. `algorithm_spec.md`와의 일치성

### 3.1 Program Specification (Section 1)
- 파이프라인 흐름 변화 없음. `pipeline.py`/`main.py` 구조 유지.

### 3.2 Stage 2 CNN Specification (Section 2.1)
- 네트워크 구조 변화 없음 (`model.py` 그대로).

### 3.3 Learning (Section 2.2)
- Loss: BCE (`trainer.py`의 `nn.BCELoss()`)
- Batch: 9 (`train_dataset.py`의 `BATCH_SIZE = 9`)
- Epoch: 50 (`EPOCHS = 50`)
- Learning Rate: 1e-3 (`LEARNING_RATE = 1e-3`)
- Optimizer: SGD, momentum=0 (`trainer.py`의 `optim.SGD(model.parameters(), lr=lr)`)

위 네 하이퍼파라미터는 모두 `algorithm_spec.md` Section 2.2.1의 값과 일치한다.

### 3.4 Initialization (Section 2.3)
- He Normal (Conv), Xavier Normal (FC), Zero (Bias) — 변경 없음.

### 3.5 Dataset Preprocessing & Augmentation (Section 2.4)
- 6배 증강 로직 동일. 가중치 저장 위치만 `./data/weights.pth`로 변경.

### 3.6 Indicator (Section 3)
- **3.1 파라미터 수**: `model.py` 구조로부터 산출. (→ Section 5.1 참조)
- **3.2 epoch당 소요 시간**: `train_dataset.py`가 10 epoch 블록마다 측정, `data/log/train_time_log.txt`에 저장.
- **3.3 Loss 곡선**: `train_dataset.py`가 매 epoch loss 누적, matplotlib으로 `data/log/loss_curve.png` 생성.

## 4. 주요 변경 사항

- `data/weights.pth` 경로로 이관 (기존 루트 `weights.pth`는 v0.6.3 배포 시 제거)
- `data/log/` 디렉터리 자동 생성 및 지표 산출물 저장
- `matplotlib` 도입 (`Agg` 백엔드로 헤드리스 호환)
- 학습률/Batch/Epoch 하이퍼파라미터를 `algorithm_spec.md` Section 2.2.1 기준으로 통일
- `requirements.txt`에 `matplotlib` 추가
- `.gitignore`에서 `weights.pth` 제거

## 5. 테스트 결과

### 5.1 파라미터 수 (Section 3.1 지표)

`model.py`의 `SimpleCNN` 구조로부터 산출:

| Layer | 파라미터 수 | 산출식 |
|---|---|---|
| Conv1 | 448 | 16×3×3×3 + 16 |
| Conv2 | 4,640 | 32×16×3×3 + 32 |
| FC1 | 5,017,800 | 25,088×200 + 200 |
| FC2 | 201 | 200×1 + 1 |
| **합계** | **5,023,089 (≈ 5.02M)** | |

> Step 5(README 갱신) 시 이 값을 "프로그램 개요" 절에 기재 예정.

### 5.2 학습 실행 (Section 3.2 & 3.3 지표)

- **실행 명령**: `python train_dataset.py`
- **Batch**: 9 / **Epoch**: 50 / **LR**: 1e-3 / **Optimizer**: SGD
- **1 epoch당 iteration 수**: 40회 (360 ÷ 9, partial batch 없음)

**최종 BCE 손실**: **0.3118**

**`data/log/train_time_log.txt` 산출 내용:**

```
[10 epoch] elapsed: 22.84s
[20 epoch] elapsed: 22.49s
[30 epoch] elapsed: 22.55s
[40 epoch] elapsed: 22.24s
[50 epoch] elapsed: 22.93s
Average time per epoch: 2.26s
```

**관찰**:
- 10 epoch 블록별 소요 시간이 **22.2~22.9초 범위**로 매우 안정적 → 학습 시간의 변동성이 낮음
- 블록별 편차는 ±0.4초 이내로, 지표로서 신뢰도가 높음

**`data/log/loss_curve.png` 산출 내용(관찰 요약)**:
- Loss 곡선은 전반적으로 **매끄러운 감소 형태**
- 직전 epoch 대비 손실이 **증가한 경우는 총 3회** 발생
- 기울기는 점차 낮아지는 추세이나, **수렴했다고 보기에는 기울기가 남아 있음**
- 즉, **50 epoch 시점에서도 최적화가 완전히 종료되지 않은 상태**로 판단됨

**산출물 생성 확인**:
- `data/weights.pth` 정상 생성
- `data/log/train_time_log.txt` 정상 생성
- `data/log/loss_curve.png` 정상 생성

### 5.3 추론 (main.py)

- **실행 명령**: `python main.py`
- 웹캠 기반 실시간 추론 정상 작동 확인
- 얼굴 검출 시 bounding box + 확률(%) 표시 확인
- 얼굴 미검출 시 `No face detected` 표시 확인
- **v0.6.2 대비 큰 체감 변화는 없음**
- **타겟 인물을 실제 카메라 앞에 두는 상황의 추가 테스트가 필요**하다고 판단됨 (v0.6.2 개발일지에서 제기된 "실제 인물 노출 시 50% 미만" 문제의 재확인 필요)

### 5.4 지표 도입을 통해 드러난 사항

본 버전의 목적(지표 도입을 통한 후속 방향성 판단)과 관련하여, 지표에서 관찰된 사실을 정리한다.

1. **파라미터 수 (5.02M)**: 전체 파라미터의 **99.9%가 FC1(5,017,800개)** 에 집중되어 있음. Feature Extractor(Conv1+Conv2 = 5,088개)는 전체의 **0.1%** 에 불과. 이것은 이 모델의 표현력이 사실상 FC1에 편중되어 있음을 시사한다.
2. **epoch당 소요 시간 (2.26s)**: 50 epoch 총 소요 시간이 **약 113초**로, 반복 실험에 부담이 없는 수준. 하이퍼파라미터 튜닝 실험에 적합.
3. **Loss 곡선**: 최종 BCE 0.3118이지만 **수렴 미완료**로 판단됨. 50 epoch은 이 학습 설정에서 충분하지 않을 수 있음.

## 6. 결론

`algorithm_spec.md` Section 3의 3개 지표가 모두 자동 산출되도록 구현되었으며, 하이퍼파라미터가 `algorithm_spec.md`와 코드 양쪽에서 일치함을 확인하였다. 학습·추론 모두 정상 작동하며, 산출물이 예상 경로에 생성됨을 확인하였다.

본 버전의 핵심 성과는 **"지표를 통해 후속 버전의 방향성을 판단할 근거가 마련되었다"** 는 점이다. 지표에서 드러난 사항(Section 5.4)은 v0.6.4 이후의 방향 설정에 활용한다.