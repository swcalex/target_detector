# Report: v0.6.2 구현 보고서

## 1. 개요

본 보고서는 v0.6.2 개발의 Step 4(코드 구현 및 테스트) 결과를 정리한 문서이다. 본 버전은 프로젝트 정체성을 "특정 인물일 확률을 실시간으로 촬영하며 알려주는 탐지기 프로그램"으로 재정의하고, 신규 `ai_rule.md` 정책에 맞춰 문서 체계를 재설계하는 것을 목표로 한다. 내부 알고리즘(SimpleCNN, MediaPipe)은 유지하며, `algorithm_spec.md` Section 1~5의 설계를 코드로 구현하였다.

## 2. 구현 범위

`programming_plan.md`에 명시된 다음 모듈을 구현하였다.

- `face_detector.py`: MediaPipe Face Detection 기반 얼굴 검출 및 전처리
- `model.py`: SimpleCNN 이진 분류 네트워크
- `pipeline.py`: Stage 1 + Stage 2 통합 추론 파이프라인
- `trainer.py`: BCE 손실 기반 학습 관리
- `train_dataset.py`: 데이터 로드, 6배 증강, 학습 루프, `weights.pth` 저장
- `main.py`: 웹캠 2fps 실시간 추론 및 화면 표시
- `requirements.txt`: 의존성 패키지 목록

## 3. `algorithm_spec.md`와의 일치성

### 3.1 Program Definition (Section 1)

- 입력: 240p(320×240), 4:3 웹캠 영상
- 출력: 특정 인물일 확률 $y \in [0.0, 1.0]$
- 파이프라인: [영상 촬영] → [Stage 1] → [Crop] → [Resize] → [Transpose & Normalize] → [Stage 2] → [확률]
- 위 흐름이 `pipeline.py`와 `main.py`에 그대로 반영됨

### 3.2 Stage 2 CNN Specification (Section 2)

- Conv1: 3→16, 3×3, stride 1, padding 1, ReLU
- MaxPool1: 2×2, stride 2
- Conv2: 16→32, 3×3, stride 1, padding 1, ReLU
- MaxPool2: 2×2, stride 2
- Flatten: 32×28×28 = 25,088
- FC1: 25,088→200, Sigmoid
- FC2: 200→1, Sigmoid
- `model.py`의 `SimpleCNN`에 정확히 구현됨

### 3.3 Learning (Section 3)

- BCE 손실 함수 사용, MAE 삭제 반영
- `trainer.py`의 `compute_loss`가 BCE만 계산
- `algorithm_spec.md` Section 3.2~3.6에 기술된 수동 역전파 수식(출력층 그레디언트 $y - t$, MaxUnpool, Full Convolution 등)은 **PyTorch autograd가 내부적으로 수행**하며, 코드에서는 `loss.backward()`로 위임됨. 즉 Section 3는 autograd의 수학적 동작을 문서화한 것이며, 별도의 수동 구현은 하지 않음

### 3.4 Initialization (Section 4)

- Conv 레이어: He Normal
- FC 레이어: Xavier Normal
- 편향: Zero
- `model.py`의 `_init_weights`에 반영됨

### 3.5 Dataset Preprocessing & Augmentation (Section 5)

- `data/target/`, `data/non_target/` 로드
- `FaceDetector.detect_and_crop`으로 얼굴 필터링
- 112×112 Bilinear Resize, [0,1] Float32 정규화
- 6배 증강(원본, 수평 반전, 밝기 증가, 밝기 감소, 대비 증가, 복합 증강) 적용
- `train_dataset.py`에 구현됨

## 4. 주요 변경 사항

- `algorithm_spec.md` Section 6을 `programming_plan.md`로 이관하고, `algorithm_spec.md`에서는 삭제
- MAE 지표 제거, BCE만 사용
- `requirements.txt` 추가
- 디렉터리 루트를 `.`으로 표시 (GitHub repo name: `target_detector`)
- 시각화 목적의 메서드 추가: `FaceDetector.detect_bbox`, `FaceRecognitionPipeline.predict_with_bbox`
  (관련 명세는 `programming_plan.md` 4.1, 4.3에 반영됨)

## 5. 테스트 결과

### 5.1 학습 (train_dataset.py)

- 50 에폭 학습 완료
- 최종 손실: **BCE ≈ 0.4632** (SGD, lr=1e-3 기준)
- `weights.pth` 정상 생성 확인
- **학습 속도 향상의 실제 원인**: 옵티마이저가 v0.6.1의 Adam(lr=0.01)에서 v0.6.2의 SGD(lr=1e-3)로 변경된 것. SGD는 Adam 대비 step당 연산량이 2~3배 적어 속도가 향상됨. 부차적으로 MAE 지표 및 관련 `.item()` 동기화 호출이 제거됨
- 두 버전 모두 PyTorch autograd(`loss.backward()`)를 사용하므로, autograd 전환에 의한 속도 향상은 아님
- 학습률이 0.01에서 1e-3으로 낮아져 수렴이 보수적으로 진행되었으며, BCE 0.4632는 이 설정에 기인할 수 있음

### 5.2 추론 (main.py)

- 웹캠 기반 실시간 추론 정상 작동
- 얼굴 검출 시 bounding box 시각화 및 확률(%) 표시
- 얼굴 미검출 시 `No face detected` 표시

### 5.3 발견 및 수정된 버그

**UI 버그 (수정 완료)**: 초기 구현에서 `main.py`의 확률 표시가 `--`로 깜빡이는 문제 발생.

- **원인**: `prob` 변수가 2fps 주기가 아닌 프레임에서 `None`으로 덮어써져, 2fps로 갱신된 확률이 1프레임만 표시되고 즉시 사라짐. 또한 이전 버전에 있던 얼굴 검출 영역 시각화가 누락됨
- **수정**: `last_prob`, `last_bbox` 캐시 변수를 루프 외부에 두고 2fps 주기로만 갱신, 매 프레임은 마지막 상태를 유지하여 표시. `detect_bbox`, `predict_with_bbox` 메서드를 추가하여 MediaPipe 호출을 1회로 통합
- **결과**: UI 정상 작동 확인

### 5.4 무해한 경고 메시지

학습 및 추론 시 다음 경고가 출력되나 동작에 영향 없음 (MediaPipe 내부 TFLite/absl 로깅).
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 ... inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference.

## 6. 결론

v0.6.2의 코드 구현은 `algorithm_spec.md`의 설계와 일치하며, `programming_plan.md`의 계획을 모두 반영하였다. 내부 알고리즘 변경 없이 문서 체계 재설계 및 프로젝트 정체성 재정의라는 목표를 달성하였다. 테스트 과정에서 발견된 UI 버그는 수정 완료되었으며, 학습 및 추론이 정상 작동함을 확인하였다. 또한 v0.6.1은 스펙(`algorithm_spec.md` 3.2)에 명시된 SGD 대신 Adam을 사용하고 있었음이 확인되었다. v0.6.2에서 SGD로 정정하여 스펙과의 일치성을 회복하였다.