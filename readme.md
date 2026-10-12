# Project AGI

## 1. 개요 (Overview)

본 프로젝트의 주 목표는 범용 인공지능 모델(AGI)을 개발하는 것이다. 여기서 AGI란 임의로 생각해 낸, 대부분의 사람이 수행 가능한 작업을 똑같이(혹은 더 월등히) 수행할 수 있는 인공지능 모델을 의미하며, 통제 가능한 재귀적 발전 또한 가능해야 한다. 추가로 인간과의 관계에 대한 윤리적인 조건을 항상 고려해야 한다.

보조 목표는 다음과 같다. 현재의 전문성으로는 인공지능 발전의 최전선에 직접 기여하기 어렵기 때문에, 관련 주제에 관하여 깊이 추론하고 지식을 습득하기 위한 학습의 장으로서 이 프로젝트를 진행한다.

## 2. 현재 버전의 개발 현황: v0.6.0 (SimpleCNN Architecture & Pipeline Specification)

* **26-07-19**: research branch 생성 후 테스트용 커밋, v0.6.0 개발 시작

* **26-07-23**:
    * **카메라 ~ CNN 입력 데이터 파이프라인 설계 완료**: 원본 프레임 추출($240 \times 320 \times 3$, uint8) $\rightarrow$ MediaPipe Face Detector (바운딩 박스 추출) $\rightarrow$ Crop & Bilinear Interpolation Resize ($112 \times 112 \times 3$) $\rightarrow$ PyTorch Tensor 변환 및 정규화 ($3 \times 112 \times 112$, float32) 구조 확립.
    * **CNN 기본 모델 순전파 구현**: $5 \times 5$ 크기의 2차원 데이터를 입력받아 Feature Extractor(합성곱층) $\rightarrow$ Flatten(1차원 평탄화) $\rightarrow$ Classifier(MLP) 과정을 거쳐 확률을 출력하는 구조의 기본 CNN 모델 학습 및 구현. 순전파 알고리즘 동작 테스트 완료.

* **26-08-17 (v0.6.0 설계 및 명세 확립)**:
    * **Stage 2 CNN 학습 및 역전파 알고리즘 수학적 정립 (`algorithm_spec.md`)**:
        - 출력층(FC + Sigmoid)의 BCE 손실 기준 오차 신호 유도 ($\delta^{\text{out}} = y - t$) 및 가중치/편향 그레디언트 유도.
        - Fully Connected 은닉층 및 Flatten 3D 텐서 복원(Reshape) 명세 작성.
        - Max Pooling, ReLU 마스킹, Conv Layer 필터 가중치/편향 미분(교차 상관 연산) 및 패딩/회전 커널을 이용한 오차 전파 수식 구체화.
    * **이원화 초기화 전략 (Initialization Strategy) 정립**:
        - Conv Layers (ReLU): **He (Kaiming) Normal Initialization** 적용으로 음수 영역 분산 소실 방지.
        - Fully Connected Layers (Sigmoid): **Xavier (Glorot) Normal Initialization** 적용으로 입출력 분산 보존.
        - 모든 레이어 편향: **Zero Initialization** 적용.
    * **Stage 1 & Stage 2 모듈 구현 및 단위 테스트 완료**:
        - `FaceDetector` (`face_detector.py`): MediaPipe 기반 실시간 얼굴 검출 및 $112 \times 112$ 텐서 전처리 구현.
        - `SimpleCNN` (`model.py`): 2계층 Conv-Pool 구조 및 이원화 가중치 초기화 적용.
        - `FaceRecognitionPipeline` (`pipeline.py`): 얼굴 미검출 시 즉시 $0.0$ 반환 엣지 케이스 처리 및 Stage 1+2 결합.
        - `ModelTrainer` (`trainer.py`): BCE Loss 및 MAE 관측 지표 기반의 안정적 역전파/옵티마이저 스텝 관리.
    * **오프라인 데이터 증강 및 배치 학습 파이프라인 구축 (`train_dataset.py`)**:
        - `data/target/` (30장) 및 `data/non_target/` (30장) 정적 데이터셋 구축.
        - 수평 반전, 밝기 증/감, 대비 조절, 복합 변환을 결합한 **6배 데이터 증강(총 360장)** 적용.
        - Mini-batch 학습을 통해 MAE 수렴 확인 후 최적 파라미터 직렬화(`weights.pth`).
    * **실시간 메인 시스템 연동 (`main.py`)**:
        - 2fps 추론 주기 제어 및 `weights.pth` 자동 로드.
        - 타겟 매칭 상태 오버레이 UI 및 우측 하단 크롭 얼굴 실시간 PiP(Picture-in-Picture) 시각화 완성.
        - v0.6.0 개발 완료
     
* **26-09-28**:
   - repository name changed (agi -> model_architecture_study)
