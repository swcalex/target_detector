# Programming Plan

## 1. Overview

본 문서는 `algorithm_spec.md`(v0.6.3)의 설계를 코드로 구현하기 위한 종합 계획이다. 본 버전의 핵심 목표는 세 가지이다.

1. **지표(Indicator) 체계 도입**: `algorithm_spec.md` Section 3에 명시된 3개 지표(파라미터 수, epoch당 소요 시간, BCE 손실 곡선)를 산출한다.
2. **하이퍼파라미터 명세화**: 학습률 1e-3, Batch 9, Epoch 50, Loss BCE를 코드가 아닌 `algorithm_spec.md`에서 관리한다.
3. **가중치 경로 변경**: `./data/weights.pth`로 이관.

내부 알고리즘(SimpleCNN 구조, MediaPipe, SGD 학습)은 v0.6.2와 동일하게 유지한다.

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
│   ├── target/              # 특정 인물 사진 (Label: 1.0)
│   ├── non_target/          # 타인/비타겟 사진 (Label: 0.0)
│   ├── weights.pth          # 학습된 가중치 (v0.6.3부터 이 위치)
│   └── log/                 # 지표 산출물 저장소
│       ├── train_time_log.txt
│       └── loss_curve.png
├── model.py
├── face_detector.py
├── pipeline.py
├── trainer.py
├── train_dataset.py
└── main.py
```

## 3. Dependencies & Libraries

| 구분 | 패키지/라이브러리 | 용도 |
|---|---|---|
| 딥러닝 | `torch`, `torchvision` | SimpleCNN 구현, 텐서 연산, 역전파 |
| 얼굴 검출 | `mediapipe` | Stage 1 MediaPipe Face Detection |
| 영상 처리 | `opencv-python` | 웹캠 캡처, 이미지 로드/크롭/리사이즈 |
| 수치 연산 | `numpy` | 배열 연산, 텐서 변환 |
| **시각화** | **`matplotlib`** | **Loss 곡선 그래프 생성 (Section 3.3)** |

### 3.1 Dependency Management (`requirements.txt`)
- `matplotlib` 추가. 버전은 개발 환경의 `pip freeze` 결과를 반영.

## 4. Module & Class Specifications

### 4.1 `face_detector.py` — `FaceDetector`
**변경 없음.** (v0.6.2와 동일)

### 4.2 `model.py` — `SimpleCNN`
**변경 없음.** (v0.6.2와 동일)

### 4.3 `pipeline.py` — `FaceRecognitionPipeline`
**변경 없음.** (`weights_path` 인자 기본값은 유지)

### 4.4 `trainer.py` — `ModelTrainer`
**변경 없음.**
- Optimizer: `optim.SGD(model.parameters(), lr=1e-3)`
  - momentum 미지정 (PyTorch 기본값 0) → A-1 결정과 일치
- 손실: `nn.BCELoss()`

### 4.5 `train_dataset.py` — 주요 변경

**변경 항목:**

1. **가중치 경로 변경**: `torch.save(model.state_dict(), "data/weights.pth")`
2. **디렉터리 보장**: 학습 시작 시 `data/`, `data/log/`가 없으면 `os.makedirs(..., exist_ok=True)`로 생성
3. **시간 측정 (Section 3.2)**:
   - 매 10 epoch 블록마다 소요 시간을 측정하여 `data/log/train_time_log.txt`에 기록
   - 학습 종료 시 전체 epoch 평균 시간을 계산하여 동일 파일에 기록
   - 로그 포맷(예시):
     ```
     [10 epoch] elapsed: 12.34s
     [20 epoch] elapsed: 11.98s
     [30 epoch] elapsed: 12.10s
     [40 epoch] elapsed: 11.87s
     [50 epoch] elapsed: 12.05s
     Average time per epoch: 1.21s
     ```
4. **Loss 곡선 (Section 3.3)**:
   - 매 epoch의 평균 BCE loss를 리스트에 저장
   - 학습 종료 후 `matplotlib`으로 x=epoch, y=loss 그래프 생성
   - `data/log/loss_curve.png`로 저장
   - 헤드리스 환경 호환을 위해 백엔드 `Agg` 사용 (`matplotlib.use("Agg")`)

**변경 없는 부분**: 데이터 로드, 6배 증강, 미니배치 학습 루프, `ModelTrainer` 호출

### 4.6 `main.py`
**변경 항목**: `FaceRecognitionPipeline(weights_path="data/weights.pth")`로 경로 수정

### 4.7 신규 모듈
**없음.** 지표 산출 로직은 `train_dataset.py` 내부에 통합 (별도 모듈화는 지표가 3개뿐이라 과함)

## 5. Edge Cases Handling

- **디렉터리 부재**: `data/`, `data/log/`가 없을 때 자동 생성
- **얼굴 미검출** (v0.6.2와 동일): Stage 2 연산 생략, 확률 0.0 반환
- **matplotlib 백엔드**: GUI 없는 환경(예: 서버)에서도 동작하도록 `Agg` 강제
- **기존 루트 `weights.pth`**: v0.6.3 배포 시 삭제 (또는 개발자가 수동 제거). 신규 학습 시 `data/weights.pth`로 재생성됨

## 6. Implementation Notes

- **`algorithm_spec.md` 명칭 동기화**: Section 1 = "Program Specification", Section 2 = "Stage 2: CNN(in Program) Specification" 명칭을 참조 시 사용
- **하이퍼파라미터 위치**: 학습률(1e-3)·Batch(9)·Epoch(50)은 `train_dataset.py`에 하드코딩하되, `algorithm_spec.md` Section 2.2.1의 값과 반드시 일치시킴 (지시 변경 시 양쪽 동기 수정)
- **Section 3.1 파라미터 수**: 산출 자체는 개발자가 1회 계산 후 `README.md`의 "프로그램 개요" 절에 수동 기입 (Step 5에서 처리)

## 7. Step 5 이관 사항 (참고)

다음 항목은 본 Step 3이 아닌 Step 5(README/CHANGELOG 갱신)에서 처리한다.
- README.md "프로그램 개요" 절에 **파라미터 수** 기재 (B-4 결정)
- CHANGELOG.md에 v0.6.3 업데이트 내용 및 개발 일지 추가