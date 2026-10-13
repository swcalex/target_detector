# Target Detector

특정 인물일 확률을 실시간으로 촬영하며 알려주는 탐지기 프로그램.

## 1. Project Overview

본 프로그램은 웹캠 영상에서 얼굴을 검출하고, 검출된 얼굴이 **특정 인물(target)일 확률**을 실시간으로 화면에 표시하는 2단계 파이프라인 기반 탐지기이다.

### 파이프라인 구조

```
[영상 촬영] → [Stage 1: MediaPipe Face Detection] → [Crop] → [Resize]
           → [Transpose & Normalize] → [Stage 2: SimpleCNN] → [확률]
```

- **Stage 1**: Google MediaPipe의 BlazeFace 모델로 웹캠 프레임에서 얼굴 영역을 검출
- **Stage 2**: 직접 설계한 경량 CNN(SimpleCNN)이 검출된 얼굴이 특정 인물인지 이진 분류하여 확률 $y \in [0.0, 1.0]$ 을 출력

### 주요 특징

- **실시간 추론**: 2fps 주기로 추론하며 웹캠 화면에 결과를 오버레이
- **얼굴 미검출 처리**: 얼굴이 검출되지 않으면 Stage 2 연산을 생략하고 0.0% 로 처리
- **경량 CNN**: 3→16→32 채널 구조의 소형 CNN으로 CPU 환경에서도 구동 가능
- **데이터 증강**: 소규모 데이터셋(60장)을 6배 증강(360장)하여 과적합 방지

### 상세 명세

- 알고리즘 설계: [`algorithm_spec.md`](algorithm_spec.md)
- 구현 계획: [`programming_plan.md`](programming_plan.md)
- 테스트 보고서: [`report.md`](report.md)
- 개발 협업 규칙: [`ai_rule.md`](ai_rule.md)

## 2. Usage

### 2.1 요구 사항

- Python 3.9 이상
- 웹캠 (240p, 4:3 비율 지원)
- CPU 또는 CUDA 지원 GPU

### 2.2 설치

```bash
git clone https://github.com/swcalex/target_detector.git
cd target_detector
pip install -r requirements.txt
```

`requirements.txt`에 명시된 패키지:

| 패키지 | 용도 |
|---|---|
| `torch`, `torchvision` | SimpleCNN 구현, 텐서 연산, 역전파 |
| `mediapipe` | Stage 1 얼굴 검출 |
| `opencv-python` | 웹캠 캡처, 이미지 처리 |
| `numpy` | 배열 연산 |

### 2.3 학습 (weights.pth 생성)

1. `data/target/` 디렉터리에 특정 인물 사진을, `data/non_target/` 디렉터리에 타인 사진을 배치합니다.
2. 다음 명령으로 학습을 실행합니다.

```bash
python train_dataset.py
```

학습이 완료되면 `weights.pth` 파일이 생성됩니다.

### 2.4 실시간 추론

```bash
python main.py
```

- 웹캠 창이 열리고, 얼굴이 검출되면 bounding box와 함께 특정 인물일 확률(%)이 표시됩니다.
- 얼굴이 검출되지 않으면 `No face detected` 가 표시됩니다.
- **ESC** 키를 눌러 종료합니다.