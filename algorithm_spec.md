# Algorithm Specification

## 1. Program Specification

본 프로그램은 240p ($320 \times 240$), 4:3 영상 촬영을 지원하는 웹캠의 영상 정보를 받아, 프레임 내 특정 인물이 존재하는지에 대한 확률 $y \in [0.0, 1.0]$ 을 출력하는 파이프라인 구조이다.

### 1.1 Architecture

[영상 촬영 시작] - [stage 1: MediaPipe Face Detection] - [Crop] - [Resize] - [Transpose & Normalize] - [stage 2: CNN] - [확률 $y \in [0.0, 1.0]$]

### 1.2 Data Pipeline

* **[영상 촬영 시작]**: 모델 추론 주파수를 2fps로 설정해 0.5초마다 원형 프레임($240 \times 320 \times 3$, uint8) 추출.
* **[stage 1: MediaPipe Face Detection]**: MediaPipe Face Detection을 사용하여 원형 프레임에서 얼굴을 검출.
  - **입력 (Input)**: 원형 프레임 ($240 \times 320 \times 3$, uint8, BGR)
  - **처리 (Processing)**: MediaPipe Face Detection 모델을 통한 얼굴 영역 검출
  - **출력 (Output)**: bounding box ($x_{min}, y_{min}, w, h$) 또는 `None`/`[]` (얼굴 미검출 시)
* **[Crop]**: 원형 프레임에서 bounding box 영역에 따라 절단. 절단 이미지($h_{face} \times w_{face} \times 3$, uint8) 생성.
* **[Resize]**: 절단 이미지 $\rightarrow$ 쌍선형 보간법(Bilinear Interpolation)을 적용하여 $112 \times 112 \times 3$ 크기의 이미지 생성.
* **[Transpose & Normalize]**: 이미지를 PyTorch 텐서 규격으로 수정 및 정규화하여 `image_pytorch_tensor` ($3 \times 112 \times 112$, float32) 생성.
  $$
  x_{norm} = \frac{x_{pixel}}{255.0}
  $$
* **[stage 2: CNN]**: PyTorch를 사용하여 가볍고 강력하며 호환성 높은 직접 설계 CNN 모델을 적용.
  - **입력 (Input)**: `[Transpose & Normalize]` 단계에서 생성된 `image_pytorch_tensor` ($3 \times 112 \times 112$, float32)
  - **처리 (Processing)**: 특징 추출기와 분류기로 구성된 CNN 순전파 연산
  - **출력 (Output)**: 특정 인물일 확률 $y \in [0.0, 1.0]$ (스칼라, float32)

## 2. Stage 2: CNN(in Program) Specification

Stage 2 모듈은 특징 추출기(Feature Extractor)와 분류기(Classifier)가 결합된 형태의 이진 분류 CNN 구조이다.

### 2.1 Architecture & Inference(Forward Propagation)

1. **Input**: $\mathbf{X} \in \mathbb{R}^{3 \times 112 \times 112}$ (RGB 3채널 이미지, 자료형: float32)
2. **Conv Layer 1**: 
   - Input Channels: $3$, Output Channels: $16$
   - Kernel Size: $3 \times 3$ (※ 공간 커널 크기는 $3 \times 3$이나, 각 필터는 3개의 입력 채널(RGB) 전체를 동시에 커버하므로 실제 필터 텐서 형태는 $16 \times 3 \times 3 \times 3$ 임)
   - Stride: $1$, Padding: $1$
   - Output Feature Map: $\mathbf{C}_1 \in \mathbb{R}^{16 \times 112 \times 112}$ (16개의 서로 다른 채널/특징 지도 생성)
   - Activation Function: ReLU ($\text{ReLU}(z) = \max(0, z)$)
3. **Max Pooling 1**:
   - Kernel Size: $2 \times 2$, Stride: $2$
   - Output Feature Map: $\mathbf{P}_1 \in \mathbb{R}^{16 \times 56 \times 56}$
4. **Conv Layer 2**:
   - Input Channels: $16$, Output Channels: $32$
   - Kernel Size: $3 \times 3$ (※ 실제 필터 텐서 형태는 $32 \times 16 \times 3 \times 3$ 임)
   - Stride: $1$, Padding: $1$
   - Output Feature Map: $\mathbf{C}_2 \in \mathbb{R}^{32 \times 56 \times 56}$ (32개의 채널/특징 지도 생성)
   - Activation Function: ReLU
5. **Max Pooling 2**:
   - Kernel Size: $2 \times 2$, Stride: $2$
   - Output Feature Map: $\mathbf{P}_2 \in \mathbb{R}^{32 \times 28 \times 28}$
6. **Flatten & Fully Connected Layer**:
   - Flatten Dimension: $32 \times 28 \times 28 = 25,088$
   - Hidden Layer 1 (Wide Strategy): $25,088 \rightarrow 200$ (기존 MLP 은닉층 구조와의 호환 및 비교 용이)
   - Activation Function: Sigmoid
7. **Output Layer**:
   - Output Dimension: $200 \rightarrow 1$
   - Activation Function: Sigmoid
   - Output: $y \in [0.0, 1.0]$ (특정 인물일 확률)

### 2.2 Learning

Model training method: Backpropagation
   - 출력값 $y \in [0.0, 1.0]$과 정답 레이블 $t \in \{0.0, 1.0\}$ 간의 차이를 계산하여 역전파(Backpropagation)를 통해 모델의 학습 최적화를 진행한다.

#### 2.2.1 Hyperparameter Setting

1. Loss function: Binary Cross Entropy(BCE)
   - CNN 모델의 이진 분류 문제이므로 Binary Classification에 최적화 된 BCE를 선정하였다.
   - 출력층의 Sigmoid 활성화 함수와 결합할 때 발생하는 기울기 소실(Gradient Vanishing) 문제를 수학적으로 상쇄한다.
   - 학습 과정에서 $L_{BCE}$ 의 값이 더 작을 수록 최소한 학습 데이터에서는 분류를 더 잘한다고 판단할 수 있다.

2. Train Batch: 9

3. Train Epoch
   - 고정값을 사용하지 않는다. 학습은 50 epoch 단위의 블록(1 블록 = 50 epoch)으로 반복하여 수행한다.
   - 블록이 종료될 때마다 Section 3.4의 판독 절차를 수행하고, 직전 블록에서 기준 1(수렴 추세)과 기준 2(진동 진폭)가 동시에 만족되면 학습을 종료한다. 만족하지 않으면 다음 블록을 이어서 학습한다.
   - 안전 상한: 총 학습 epoch가 1000에 도달하면 판독 결과와 무관하게 학습을 종료한다.

4. Learning Rate: 1e-3

#### 2.2.2 Backpropagation & Output Layer Gradients

출력층(Fully Connected Layer + Sigmoid) 단계에서 BCE 손실 함수를 최소화하기 위해 Chain Rule을 적용하여 그레디언트를 구하고 에러를 전파한다.

1. **출력층 에러 및 그레디언트 ($\delta^{\text{out}}$)**:
   - Sigmoid 활성화 함수 적용 전의 선형 출력값을 $z$, 최종 예측 확률을 $y = \sigma(z)$라 할 때, BCE 손실 함수와 Sigmoid의 도함수가 결합되어 오차 신호 $\delta^{\text{out}}$는 다음과 같이 매우 간결하게 유도된다.
   
   $$
   \delta^{\text{out}} = \frac{\partial L}{\partial z} = y - t
   $$
   
   (여기서 $t$는 특정 인물의 존재 여부를 나타내는 타겟 레이블(Target Value, $t \in \{0.0, 1.0\}$)이다.)

2. **분류기 가중치 및 편향 그레디언트**:
   - 출력층 가중치 $\mathbf{W}_{\text{out}}$과 편향 $b_{\text{out}}$에 대한 손실 함수의 기울기는 다음과 같다.
   
   $$
   \frac{\partial L}{\partial \mathbf{W}_{\text{out}}} = \delta^{\text{out}} \mathbf{h}^{\top}, \quad \frac{\partial L}{\partial b_{\text{out}}} = \delta^{\text{out}}
   $$
   
   (여기서 $\mathbf{h}$는 이전 은닉층에서 전달된 활성화 값이다.)

3. **파라미터 업데이트 (Parameter Update via SGD)**:
   - 계산된 그레디언트를 바탕으로 경사하강법(Gradient Descent)을 적용하여 모델의 파라미터를 갱신한다. ($\eta$는 학습률(Learning Rate))
   
   $$
   \begin{aligned}
   \mathbf{W}_{\text{out}} &\leftarrow \mathbf{W}_{\text{out}} - \eta \frac{\partial L}{\partial \mathbf{W}_{\text{out}}} \\
   b_{\text{out}} &\leftarrow b_{\text{out}} - \eta \frac{\partial L}{\partial b_{\text{out}}}
   \end{aligned}
   $$

4. **이전 레이어로의 오차 전파 (Chain Rule)**:
   - 출력층에서 계산된 오차 신호 $\delta^{\text{out}}$는 가중치 전치 행렬과의 곱을 통해 은닉층 및 합성곱층(Conv/Pooling)으로 계속해서 역전파된다.

#### 2.2.3 Fully Connected Hidden Layer 및 Flatten 단계 역전파

출력층에서 계산된 오차 신호 $\delta^{\text{out}}$가 Fully Connected 은닉층(Hidden Layer)과 Flatten 단계를 거쳐 2D 특징 지도(Feature Map) 영역으로 전파되는 과정이다.

1. **Fully Connected 은닉층 오차 전파 ($\delta^{\text{fc}}$)**:
   - 출력층의 가중치 $\mathbf{W}_{\text{out}}$과 오차 $\delta^{\text{out}}$, 그리고 은닉층의 활성화 함수(Sigmoid) 도함수를 결합하여 은닉층의 오차를 계산한다.
   
   $$
   \delta^{\text{fc}} = (\mathbf{W}_{\text{out}}^{\top} \delta^{\text{out}}) \odot \sigma'(\mathbf{z}^{\text{fc}})
   $$
   
   (여기서 $\odot$은 Hadamard Product(원소별 곱), $\sigma'(z) = \sigma(z)(1 - \sigma(z))$이다.)

2. **은닉층 파라미터 업데이트**:
   - 은닉층의 가중치 $\mathbf{W}_{\text{fc}}$와 편향 $b_{\text{fc}}$에 대한 그레디언트를 구하고, 동일하게 경사하강법으로 갱신한다.
   
   $$
   \frac{\partial L}{\partial \mathbf{W}_{\text{fc}}} = \delta^{\text{fc}} \mathbf{A}_{\text{prev}}^{\top}, \quad \frac{\partial L}{\partial b_{\text{fc}}} = \delta^{\text{fc}}
   $$
   
   (여기서 $\mathbf{A}_{\text{prev}}$는 은닉층의 입력으로 들어온 Flatten 벡터, 즉 Max Pooling 2의 출력이 1차원으로 평탄화된 값이다.)
   
   $$
   \mathbf{W}_{\text{fc}} \leftarrow \mathbf{W}_{\text{fc}} - \eta \frac{\partial L}{\partial \mathbf{W}_{\text{fc}}}, \quad b_{\text{fc}} \leftarrow b_{\text{fc}} - \eta \frac{\partial L}{\partial b_{\text{fc}}}
   $$

3. **Flatten 역전파 (Reshape)**:
   - Max Pooling 2의 3차원 출력 텐서 형태($32 \times 28 \times 28$)로 평탄화(Flatten)되었던 오차 벡터를 다시 원래의 3D 텐서 구조로 복원(Reshape)한다.
   
   $$
   \delta^{\text{pool2}} = \text{Reshape}(\mathbf{W}_{\text{fc}}^{\top} \delta^{\text{fc}}, (32, 28, 28))
   $$

#### 2.2.4 Max Pooling 2 단계 역전파

Flatten 단계에서 3D 텐서로 복원된 오차 신호 $\delta^{\text{pool2}}$를 Max Pooling 2 레이어의 입력 영역(ReLU 2의 출력 맵)으로 역전파하는 과정이다.

1. **Max Pooling 역전파 원리**:
   - Max Pooling은 순전파 시 윈도우(예: $2 \times 2$) 내에서 **최댓값(Maximum)을 가졌던 위치**로만 오차 신호를 그대로 전달하고, 나머지 위치의 오차는 $0$으로 처리한다.
   - 따라서 순전파 시 기억해 둔 인덱스(Mask)를 활용하여 오차를 분배한다.

2. **오차 맵 전달 ($\delta^{\text{relu2}}$)**:
   - 각 채널별로 Max Pooling 윈도우 내 최댓값 위치에만 $\delta^{\text{pool2}}$의 오차를 매핑하고, 나머지는 0으로 채워진 동일 크기의 오차 맵 $\delta^{\text{relu2}}$를 생성한다.
   
   $$
   \delta^{\text{relu2}} = \text{MaxUnpool}(\delta^{\text{pool2}}, \text{Mask}_{\text{pool2}})
   $$

#### 2.2.5 ReLU 2 및 Conv Layer 2 단계 역전파

Max Pooling 2를 거쳐 전달된 오차 신호 $\delta^{\text{relu2}}$를 바탕으로, ReLU 2 활성화 함수와 Conv Layer 2의 필터 가중치를 갱신하고 이전 레이어로 오차를 전파하는 과정이다.

1. **ReLU 2 역전파**:
   - ReLU 활성화 함수의 특성에 따라, 순전파 시 값이 $0$보다 컸던 위치에서는 오차를 그대로 통과시키고, $0$ 이하이었던 위치에서는 오차를 차단($0$으로 설정)한다.
   
   $$
   \delta^{\text{conv2\_out}} = \delta^{\text{relu2}} \odot \mathbb{I}(\mathbf{C}_2 > 0)
   $$
   
   (여기서 $\mathbb{I}$는 인디케이터 함수로, 조건이 참이면 1, 거짓이면 0을 반환한다.)

2. **Conv Layer 2 필터 가중치 및 편향 미분/갱신**:
   - Conv Layer 2의 커널 $\mathbf{K}_2$에 대한 손실 함수의 기울기는 순전파 때와 동일하게 패딩이 적용된 이전 입력 맵 $\text{Pad}(\mathbf{P}_1)$과 현재 오차 맵 $\delta^{\text{conv2\_out}}$ 간의 교차 상관(Cross-correlation) 연산으로 구한다.
   
   $$
   \frac{\partial L}{\partial \mathbf{K}_2} = \text{Pad}(\mathbf{P}_1) * \delta^{\text{conv2\_out}}
   $$
   
   - 각 출력 채널에 더해지는 편향 $b_2$에 대한 기울기는 해당 채널 오차 맵의 모든 공간 위치(높이, 너비)에 대해 오차를 합산(Sum)하여 구한다.
   
   $$
   \frac{\partial L}{\partial b_2} = \sum_{h, w} \delta^{\text{conv2\_out}}_{:, h, w}
   $$
   
   - 경사하강법을 통해 Conv 2 필터 가중치와 편향을 갱신한다.
   
   $$
   \mathbf{K}_2 \leftarrow \mathbf{K}_2 - \eta \frac{\partial L}{\partial \mathbf{K}_2}, \quad b_2 \leftarrow b_2 - \eta \frac{\partial L}{\partial b_2}
   $$

3. **이전 레이어로의 오차 전파 ($\delta^{\text{pool1}}$ 구하기)**:
   - Conv Layer 2의 필터 커널 $\mathbf{K}_2$를 180도 회전(Flip)시킨 $\mathbf{K}_2^{\text{rot180}}$을 현재 오차 맵 $\delta^{\text{conv2\_out}}$에 대해 Full Convolution(전체 합성곱) 연산하여, 이전 레이어인 Max Pooling 1 출력 위치에서의 오차 맵 $\delta^{\text{pool1}}$을 계산한다.
   
   $$
   \delta^{\text{pool1}} = \delta^{\text{conv2\_out}} *_{\text{full}} \mathbf{K}_2^{\text{rot180}}
   $$
   
   (여기서 $*_{\text{full}}$은 경계면 오차 손실을 방지하기 위해 경계 패딩을 포함하여 확장 연산하는 Full Convolution을 의미한다.)

#### 2.2.6 Max Pooling 1, ReLU 1 및 Conv Layer 1 단계 역전파

Conv Layer 2에서 전달받은 오차 신호 $\delta^{\text{pool1}}$을 받아, Feature Extractor의 최전단 레이어들(Max Pooling 1 $\rightarrow$ ReLU 1 $\rightarrow$ Conv Layer 1)에 대해 역전파를 수행하고 최초 필터 가중치 $\mathbf{K}_1$ 및 편향 $b_1$을 최종 갱신하는 과정이다.

1. **Max Pooling 1 역전파 ($\delta^{\text{relu1}}$)**:
   - 순전파 시 기억해 둔 인덱스 Mask($\text{Mask}_{\text{pool1}}$)를 활용하여, $2 \times 2$ 윈도우 내 최댓값이었던 위치로만 오차 $\delta^{\text{pool1}}$을 복원 및 전파한다.
   
   $$
   \delta^{\text{relu1}} = \text{MaxUnpool}(\delta^{\text{pool1}}, \text{Mask}_{\text{pool1}})
   $$

2. **ReLU 1 역전파 ($\delta^{\text{conv1\_out}}$)**:
   - 순전파 시 Conv 1의 선형 출력 $\mathbf{C}_1$이 $0$보다 컸던 위치의 오차만 통과시킨다.
   
   $$
   \delta^{\text{conv1\_out}} = \delta^{\text{relu1}} \odot (\mathbf{C}_1 > 0)
   $$

3. **Conv Layer 1 필터 가중치 및 편향 미분/갱신**:
   - 최초 입력 이미지 $\mathbf{X} \in \mathbb{R}^{3 \times 112 \times 112}$에 패딩을 적용한 $\text{Pad}(\mathbf{X})$와 현재 오차 맵 $\delta^{\text{conv1\_out}}$ 간의 교차 상관 연산을 통해 Conv 1의 필터 가중치 $\mathbf{K}_1$ 기울기를 구한다.
   
   $$
   \frac{\partial L}{\partial \mathbf{K}_1} = \text{Pad}(\mathbf{X}) * \delta^{\text{conv1\_out}}
   $$
   
   - 해당 채널 오차 맵의 모든 공간 위치(높이, 너비)를 합산하여 편향 $b_1$의 기울기를 구한다.
   
   $$
   \frac{\partial L}{\partial b_1} = \sum_{h, w} \delta^{\text{conv1\_out}}_{:, h, w}
   $$
   
   - 경사하강법을 이용하여 Conv 1의 필터 가중치 $\mathbf{K}_1$과 편향 $b_1$을 최종 갱신한다.
   
   $$
   \mathbf{K}_1 \leftarrow \mathbf{K}_1 - \eta \frac{\partial L}{\partial \mathbf{K}_1}, \quad b_1 \leftarrow b_1 - \eta \frac{\partial L}{\partial b_1}
   $$

4. **학습 파이프라인의 종결**:
   - $\mathbf{X}$는 외부 입력(웹캠 데이터)이므로 더 이상 이전 레이어로 오차를 전파하지 않고 1회 역전파(Backpropagation) 파이프라인을 종료한다.

### 2.3 Initialization

각 레이어의 특성에 맞춘 가중치 및 편향 초기화 전략을 적용한다.

#### 2.3.1 Weights Initialization
활성화 함수의 특성과 레이어 종류에 따라 분산이 소실되거나 폭발하지 않도록 적절한 초기화 기법을 이원화하여 적용한다.

1. **합성곱 층 (Conv Layers: $\mathbf{K}_1, \mathbf{K}_2$)**:
   - **적용 기법**: **He (Kaiming) Normal Initialization**
   - **선정 이유**: 합성곱 층 뒤에 **ReLU** 활성화 함수가 연결되므로, 음수 영역에서 값이 소실되는 ReLU의 특성을 고려하여 입력 분산을 효과적으로 유지해 주는 He 초기화를 적용한다.

2. **완전연결 층 (Fully Connected Layers: $\mathbf{W}_{\text{fc}}, \mathbf{W}_{\text{out}}$)**:
   - **적용 기법**: **Xavier (Glorot) Normal Initialization**
   - **선정 이유**: 은닉층과 출력층에 **Sigmoid** 활성화 함수를 사용하므로, 입출력의 분산을 일정하게 유지하여 초기 기울기 흐름을 안정화하는 Xavier 초기화를 적용한다. (기존 v0.5.x MLP 구조와의 호환성 유지)

#### 2.3.2 Bias Initialization

1. **모든 레이어의 편향 ($b_1, b_2, b_{\text{fc}}, b_{\text{out}}$)**:
   - **적용 기법**: **Zero Initialization**
   - **선정 이유**: 초기 학습 단계에서 뉴런이 한쪽으로 치우치거나 활성화가 죽는 현상을 방지하기 위해 편향은 모두 $0$으로 설정한다.

### 2.4 Stage 2 학습 데이터셋 전처리 및 증강 명세 (Dataset Preprocessing & Augmentation)

본 절은 오프라인 이미지 파일로부터 Stage 2 모듈의 학습에 사용될 데이터셋을 구축, 정규화 및 증강하는 파이프라인을 정의한다.

#### 2.4.1 디렉터리 구성 및 레이블 정의
* `./data/target/`: 타겟 인물 이미지 ($N_{\text{target}} = 30$, Target Label $t = 1.0$)
* `./data/non_target/`: 타인 및 비타겟 이미지 ($N_{\text{non\_target}} = 30$, Target Label $t = 0.0$)

#### 2.4.2 데이터 전처리 파이프라인
1. **로드**: OpenCV `cv2.imread()`를 통해 BGR 3채널 원본 이미지 로드.
2. **Stage 1 필터링**: `FaceDetector.detect_and_crop()`을 실행하여 유효한 얼굴 영역만 추출. (얼굴 미검출 시 해당 이미지는 제외)
3. **규격화 및 텐서화**: $112 \times 112$ Bilinear Resize $\rightarrow$ RGB 변환 $\rightarrow$ $[0.0, 1.0]$ Float32 정규화 $\rightarrow$ ($3 \times 112 \times 112$) PyTorch 텐서 변환.

#### 2.4.3 데이터 증강 알고리즘 (Data Augmentation Strategy)
소규모 데이터셋($30 + 30 = 60$ 장)의 과적합을 방지하고 조명/각도 변화에 강건성을 확보하기 위해, 검출된 $112 \times 112$ 정규화 이미지 1장당 6개의 변형본(원형 포함)을 생성하여 **6배 확장($60 \times 6 = 360$ 장)**한다.

1. **원본 (Original)**: 전처리 완료된 기본 텐서
2. **수평 반전 (Horizontal Flip)**: 얼굴의 좌우 대칭성을 고려하여 $W$ 축 기준 반전
   $$
   I_{\text{flip}}(c, y, x) = I(c, y, W - 1 - x)
   $$
3. **밝기 증가 (Brightness Boost)**: 조명이 밝은 환경 시뮬레이션 (상한선 $1.0$ 클리핑)
   $$
   I_{\text{bright}} = \text{clip}(I \times 1.25, 0.0, 1.0)
   $$
4. **밝기 감소 (Brightness Darken)**: 어두운 조명 환경 시뮬레이션
   $$
   I_{\text{dark}} = I \times 0.75
   $$
5. **대비 증가 (Contrast Adjustment)**: 윤곽 및 그림자 강조
   $$
   I_{\text{contrast}} = \text{clip}((I - 0.5) \times 1.3 + 0.5, 0.0, 1.0)
   $$
6. **복합 증강 (Flip + Brightness)**: 수평 반전 상태에서 밝기 변화($\times 1.15$) 결합

#### 2.4.4 모델 가중치 저장 (Serialization)
* 오프라인 배치 학습 완료 후 최적화된 파라미터를 PyTorch `state_dict` 규격인 `weights.pth`로 저장하여 실시간 파이프라인에서 호출할 수 있도록 한다.(위치: './data/weights.pth')

## 3. Indicator(지표)

프로그램의 개선에 대한 근사적인 상태 확인과 단기적인 개발의 방향성을 설정하는데 도움이 되기 위해 필요한 지표들을 도입한다.

### 3.1 Stage 2: CNN model parameter

- 목적: 모델의 크기를 나타내는 지표로 모델의 크기를 파악할 수 있다.
- 지표 산출: stage 2 모듈의 CNN 모델이 가지고 있는 가중치와 편향을 모두 포함한 개수를 계산한다.
- 결과물: README.md의 프로그램 개요 부분에 표시

### 3.2 Learning time taken per epoch

- 목적: 1 epoch의 학습마다 얼마의 시간이 소요되는지를 확인하면서 현재 프로그램을 구동 중인 하드웨어를 통해 학습 시간이 오래 걸린다면 이를 줄이기 위해 최적화하는 방안 등의 단기 방향성을 세우는데 도움을 줄 수 있다.
- 지표 산출: 하나의 파일 안에 다음을 표시한다.
   1. 10 epoch당 소요된 시간(1~10 1번 산출, 11~20 1번 산출, ...)
   2. 총 학습 진행 과정의 소요 시간 및 1 epoch당 소요된 평균 시간
- 결과물: 생성 된 파일을 './data/log/'에 저장.

### 3.3 Loss funtion graph

- 목적: 학습 과정에서 학습 데이터 한정으로 지도학습이 최적화가 잘 되었는지 판단할 수 있는 기준이다.
- 지표 산출: 1 epoch마다 산출 된 Loss function을 통해 x축은 epoch(횟수), y축은 Loss function으로 한 그래프를 생성한다.
- 결과물: 생성한 그래프를 './data/log/'에 저장.

### 3.4 Convergence Judgment and Learning Termination

- 목적: 블록 단위 학습(Section 2.2.1)이 종료 조건에 도달했는지를 정량 기준으로 판독한다. 판독 결과는 만족/불만족 지표로 기록되며, 이 기록에 따라 학습 종료 여부를 결정한다.

- 지표 산출: 판독마다 다음을 수행한다.
   1. 판독 시점 및 분석 창: $k$번째 블록이 종료되는 시점을 $e_k = 50k$라 하고, 분석 창은 직전 50 epoch 구간이다.
      $$
      W_k = \{ L(e) \mid 50(k-1) < e \le 50k \}
      $$
      창 내 epoch별 학습 Loss $L(e)$에 대해 전반 25 epoch 평균 $m_{\text{first}}$, 후반 25 epoch 평균 $m_{\text{second}}$, 창 전체 평균 $m$, 표준편차 $\sigma$를 정의한다.
   2. 기준 1 (수렴 추세, 만족/불만족 판정):
      $$
      r = \frac{m_{\text{first}} - m_{\text{second}}}{m_{\text{first}}}, \qquad \text{만족 조건: } |r| \le \theta_1 = 0.05
      $$
      후반부 평균이 전반부 평균보다 5% 이상 높은 경우($r < -0.05$)에는 불만족 판정과 동시에, 별도의 이상 이벤트로서 해당 블록과 $r$ 값을 로깅한다.
   3. 기준 2 (진동 진폭, 만족/불만족 판정):
      $$
      \Pi = \frac{\max_{W_k} L - \min_{W_k} L}{m}, \qquad \text{만족 조건: } \Pi \le \theta_2 = 0.10
      $$
   4. 기준 3 (하한 이탈, 로그 전용): $L(e) < m - 3\sigma$를 만족하는 epoch가 창 내 존재하면 하한 이탈로 판정하고, 발생 시 해당 epoch 번호와 그 epoch의 Loss 값을 전부 로깅한다. 존재하지 않으면 '없음'으로 로깅한다. 본 기준은 종료 조건에 포함하지 않는다.
   5. 학습 종료 판정: 직전 블록에서 기준 1과 기준 2가 동시에 만족되면 학습을 종료하고, 아니면 다음 블록을 이어 학습 후 재판독한다. 단, 총 학습 epoch가 1000에 도달하면 판독 결과와 무관하게 학습을 종료한다(안전 상한).

- 결과물: 블록별 판독 레코드를 누적 기록하고, 학습 종료 시 종료 요약(총 학습 epoch, 종료 사유)을 덧붙인 파일을 './data/log/'에 저장.