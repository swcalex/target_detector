# Programming Plan

## 1. Overview

본 문서는 `algorithm_spec.md`의 설계 내용을 실제 코드로 구현하기 위한 종합적인 계획을 정의한다. 본 버전(v0.6.2)은 프로젝트 정체성을 "특정 인물일 확률을 실시간으로 촬영하며 알려주는 탐지기 프로그램"으로 재정의하고, 신규 `ai_rule.md` 정책에 맞춰 문서 체계를 재설계하는 것을 목표로 한다. 내부 알고리즘(SimpleCNN, MediaPipe)은 유지하며, 구현 계획은 `algorithm_spec.md` Section 1~5에 명시된 내용만을 대상으로 한다.

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
├── data/
│   ├── target/             # 특정 인물 사진 (Label: 1.0)
│   └── non_target/         # 타인/비타겟 사진 (Label: 0.0)
├── model.py
├── face_detector.py
├── pipeline.py
├── trainer.py
├── train_dataset.py
├── weights.pth
└── main.py
```

## 3. Dependencies & Libraries

| 구분 | 패키지/라이브러리 | 용도 |
|---|---|---|
| 딥러닝 | `torch`, `torchvision` | SimpleCNN 구현, 텐서 연산, 역전파 |
| 얼굴 검출 | `mediapipe` | Stage 1 MediaPipe Face Detection |
| 영상 처리 | `opencv-python` (`cv2`) | 웹캠 캡처, 이미지 로드/크롭/리사이즈 |
| 수치 연산 | `numpy` | 배열 연산, 텐서 변환 |
| 데이터 증강 | (자체 구현) | `algorithm_spec.md` 5.3의 6배 증강 |

> **Note**: 별도의 테스트 프레임워크(`pytest` 등)는 정책에 따라 도입하지 않는다.

### 3.1 Dependency Management (`requirements.txt`)

프로젝트의 원활한 환경 구성을 위해 `requirements.txt` 파일을 사용한다. 이 파일에는 위 표에 명시된 패키지들과 그 버전이 기록된다.

- **파일 생성**: 개발 환경에서 `pip freeze > requirements.txt` 명령어로 생성한다.
- **일괄 설치**: 새로운 환경에서는 `pip install -r requirements.txt` 명령어로 모든 의존성을 한 번에 설치한다.

## 4. Module & Class Specifications

### 4.1 `face_detector.py` — `FaceDetector`

- **역할**: 웹캠 프레임 입력($240 \times 320 \times 3$, uint8, BGR)을 받아 얼굴을 검출하고 Stage 2 텐서 형태로 전처리한다.
- **외부 모듈**: `mediapipe.solutions.face_detection`, `opencv-python`
- **주요 메서드**:
  - `detect_and_crop(frame: np.ndarray) -> Optional[np.ndarray]`
    - BGR → RGB 변환 후 MediaPipe로 Bounding Box 검출
    - 검출된 영역 크롭, 얼굴 없으면 `None` 반환
  - `preprocess(cropped_face: np.ndarray) -> torch.Tensor`
    - $112 \times 112$ Bilinear Resize
    - $[0, 1]$ 범위 Float32 정규화
    - Transpose ($H \times W \times C \rightarrow C \times H \times W$)
    - 차원 확장 ($1 \times 3 \times 112 \times 112$)
  - `detect_bbox(frame) -> Optional[tuple]`: 시각화용 bounding box 반환

### 4.2 `model.py` — `SimpleCNN`

- **역할**: Stage 2 이진 분류 네트워크 구조 정의 및 가중치 초기화
- **프레임워크**: PyTorch `nn.Module`
- **주요 메서드**:
  - `__init__()`: Conv1, Conv2, MaxPool, FC1, FC2 레이어 및 활성화 함수(ReLU, Sigmoid) 정의
  - `_init_weights()`: Conv 레이어에 He Normal, FC 레이어에 Xavier Normal, 편향에 Zero 초기화 적용
  - `forward(x: torch.Tensor) -> torch.Tensor`: 순전파 ($1 \times 3 \times 112 \times 112 \rightarrow 1 \times 1$)

### 4.3 `pipeline.py` — `FaceRecognitionPipeline`

- **역할**: `FaceDetector`와 `SimpleCNN`을 하나의 추론 단위로 결합하고 예외 상황을 안전하게 처리
- **주요 메서드**:
  - `predict(frame: np.ndarray) -> float`:
    1. `FaceDetector.detect_and_crop` 호출
    2. 반환값이 `None`일 경우, Stage 2를 실행하지 않고 즉시 확률 $0.0$ 반환
    3. cropped 이미지가 존재할 경우 전처리 후 `SimpleCNN.forward` 실행하여 $y \in [0.0, 1.0]$ 반환
  - `predict_with_bbox(frame) -> tuple[float, Optional[tuple]]`: 확률과 bbox를 함께 반환

### 4.4 `trainer.py` — `ModelTrainer`

- **역할**: Stage 2 CNN 모델의 역전파, 손실 계산 및 파라미터 업데이트 관리
- **옵티마이저**: **SGD**, 기본 학습률 `lr=1e-3` (`algorithm_spec.md` Section 3.2 "Parameter Update via SGD" 명세에 따름)
- **주요 메서드**:
  - `compute_loss(y_pred, target) -> torch.Tensor`: BCE 손실 계산
  - `train_step(x, target) -> dict`: Gradient 초기화, Forward, Loss 계산, Backward, Optimizer Step 수행 및 loss dict 반환

### 4.5 `train_dataset.py`

- **역할**: `data/` 디렉터리의 이미지를 Stage 1 전처리 및 6배 증강 후 미니배치 학습 진행, `weights.pth` 파일 생성
- **세부 계획**:
  - `data/target/`, `data/non_target/` 이미지 로드 (OpenCV `cv2.imread`)
  - `FaceDetector.detect_and_crop`으로 얼굴 영역 추출 (미검출 이미지 제외)
  - `algorithm_spec.md` 5.3의 6가지 증강 기법 적용
  - 미니배치 학습 루프 실행, `ModelTrainer` 활용
  - 학습 완료 후 `state_dict`를 `weights.pth`로 저장

### 4.6 `main.py`

- **역할**: 프로그램 진입점. OpenCV 웹캠 캡처를 2fps 주기로 구동하며 파이프라인 연동 및 결과를 화면에 시각화
- **세부 계획**:
  - `FaceRecognitionPipeline` 인스턴스 생성
  - 웹캠 프레임 캡처 (2fps)
  - `predict` 호출 후 확률을 화면에 오버레이
  - 종료 조건 처리 (ESC 키 등)

## 5. Edge Cases Handling

- **얼굴 미검출 (No Face Detected)**: Stage 1에서 MediaPipe가 얼굴을 감지하지 못하거나 bounding box 영역이 프레임 범위를 벗어날 경우, Stage 2 연산을 생략하고 예측 확률을 $0.0$으로 처리하여 불필요한 연산을 방지한다.
- **입력 채널 및 정규화 오류**: OpenCV의 BGR 색상 공간을 MediaPipe 및 PyTorch 입력 기준인 RGB로 반드시 변환하고, uint8($0\sim255$)을 float32($0.0\sim1.0$)로 정확히 정규화한다.

## 6. Implementation Notes

- **`weights.pth`**: `algorithm_spec.md`의 알고리즘 변경이 없으므로 기존 `weights.pth`를 그대로 사용한다. 단, 디렉터리 구조 변경으로 인한 경로 차이가 발생할 경우 로딩 경로만 수정한다.
- **코드 자산**: v0.6.1의 `model.py`, `face_detector.py`, `pipeline.py`, `trainer.py`, `train_dataset.py`, `main.py`를 기반으로 하되, `algorithm_spec.md`와의 일치성을 우선하여 필요한 부분만 수정한다.
- **문서 동기화**: `algorithm_spec.md` Section 6은 본 문서로 이관되었으므로, `algorithm_spec.md`에서는 해당 섹션을 완전히 삭제한다.