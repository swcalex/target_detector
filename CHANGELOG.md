# Changelog

## [v0.6.3] - 2026-10-14

### 1. 업데이트 내용

#### 지표(Indicator) 체계 도입

- `algorithm_spec.md`에 **Section 3 (Indicator)** 신설. 3개 지표를 정의:
  - **3.1 파라미터 수**: 모델 구조로부터 산출되는 정적 지표. README의 프로그램 개요에 표시.
  - **3.2 epoch당 소요 시간**: 10 epoch 블록별 소요 시간 + 전체 epoch당 평균 시간. `data/log/train_time_log.txt`에 저장.
  - **3.3 Loss 곡선**: epoch별 BCE 손실 그래프. `data/log/loss_curve.png`에 저장.

#### 하이퍼파라미터 명세화

- `algorithm_spec.md` Section 2.2.1에 학습 하이퍼파라미터를 직접 명시:
  - Loss: BCE
  - **Train Batch: 9** (v0.6.2의 16에서 변경, 360장이 정확히 나누어떨어짐)
  - Train Epoch: 50
  - Learning Rate: 1e-3
- Optimizer는 스펙에 명시하지 않고 Step 3(`programming_plan.md`)에서 Engine이 SGD(momentum=0)로 확정.

#### 가중치 경로 이관

- `weights.pth` 저장 위치를 루트(`./weights.pth`)에서 `./data/weights.pth`로 이관.
- `.gitignore`에서 `weights.pth` 라인 제거. (단, `data/` 무시 규칙으로 인해 `data/weights.pth`는 여전히 untracked)

#### 문서 구조 재편

- `algorithm_spec.md` Section 1 제목: `Program Definition` → `Program Specification`
- `algorithm_spec.md` Section 2 제목: `Stage 2 CNN Specification` → `Stage 2: CNN(in Program) Specification`
- `algorithm_spec.md`의 학습 관련 하위 절을 Section 2.2로 통합 (`Learning`, `Hyperparameter Setting` 포함)

#### 코드 개선

- `train_dataset.py`: Batch=9 반영, `data/log/` 자동 생성, 10 epoch 블록별 시간 로깅, matplotlib Loss 곡선 생성, `data/weights.pth` 저장.
- `main.py`: `weights_path="data/weights.pth"` 로 경로 수정.
- `matplotlib` 의존성 추가 (헤드리스 환경 호환을 위해 `Agg` 백엔드 사용).

### 2. 개발 일지

- 2026-10-13 (v0.6.2)
    - v0.6.2 개발 step 1~5까지 완료. epoch 50으로 학습을 진행하고 실행까지의 테스트를 진행한 결과 다음 사항들을 느꼈다.
        - 동일한 과정의 한 epoch에서 이전 버전에 비해 속도가 많이 향상되었다. 이는 옵티마이저에 Adam 대신 SGD를 써서 동일 작업 당 연산량이 줄어든게 주요 이유일 것으로 생각한다. 또한 MAE 보조지표 자체를 제외시켜 로깅 과정을 더욱 단순화 시켰다. BCE 손실함수 지표만 본다고 해서 이전에 비해 학습 진행정도를 더욱 알 수 없다고 보기는 어렵다. BCE 자체가 이진분류 모델의 학습에 최적화된 지표이기 때문이다. 구체적으로 연산량이 작아졌다는 것을 알 수 있는 지표 또는 로깅 프레임워크를 만들어보는게 프로젝트의 하나의 방향성이 될 수 있지 않을까 생각했다.
        - 학습률은 v0.6.1과 비교해 10배가 작아졌다. 이는 그레디언트에 따른 가중치 변화를 10배 작게 가져간다는 뜻이다. 사실 학습률 지정을 step 2의 설계 단계에서 실수로 따로 지정해주지 않아 별도 지시 없이 학습률이 변화할 수 있었던 것이다. 구현 과정에서 이를 발견했을 때, 나는 원래의 값으로 되돌리라는 지시를 하지 않았다. 그 이유는 이전 버전에서 학습 이후 학습 데이터를 그대로 웹캠에 노출시켰을 떄는 100% 또는 0%로 대부분 표시되었고, 실제 인물을 노출시켰을 떄 타겟임에도 불구하고 50% 미만의 수치로 타겟을 인식한다고 볼 수 없었다. 그래서 나는 과적합이 아닌지 의심하게 되었고, 이에 관하여 향후 모델의 크기 늘리기 / 데이터 셋 늘리기 / 학습 방식의 변화(실시간 촬영으로 학습 등) / 학습률 등의 하이퍼파라미터 조정 등을 철처한 통제 하에 진행하여 그 원인을 밝힘과 동시에 모델이 특정 인물을 잘 인식할 수 있도록 하는게 해당 프로젝트의 방향성이 될 수 있지 않을까 생각했다.

- 2026-10-14 (v0.6.3)
    - 이번 프로젝트의 핵심 방향성은 다음과 같다.
        1. 프로그램의 효율성이 어느정도인지, 모델의 성능이 어느정도인지, 특정 부분에서 개선하고 싶은 부분이 있다면 그 부분이 잘 개선되는지 등의 목적으로 어떠한 판단 기준을 세울 수 있다. 그 기준을 통해 추출된 데이터인 지표를 얻을 수 있다. 이러한 지표를 얻는 방식은 직접 기준을 만들어 평가하는 것, 외부에서 이미 만든 기준으로 평가하는 것으로 나뉠 수 있다. 이번 버전의 개발은 어떠한 기준을 세워 지표를 추출해 프로그램을 평가하고 방향성을 잡을 것인지에 관하여 의논할 것이다.
        2. 학습 과정 중에 사용되는 하이퍼파라미터 설정을 algorithm_spec.md에서 직접 지정하는 것이다. 이는 코드 구현의 문제가 아닌, 코드를 작성하는 방향성 자체를 설정하는 부분이기 때문이다.
    - v0.6.3 버전 개발 step 1~4 완료
    - 현재 engine 조합 - deepseek(web)+github copilot(window desktop app, education tier)
        - deepseek는 파일 생성, 수정을 통해 마크다운 코드블록으로 전체 파일을 출력할 때, 중첩 코드블록(markdown 안에text) 때문에 바깥 블록이 조기 종료되는 현상 발생. 해결책으로 바깥을 4개 백틱으로 감싸 재출력해야 함.
    - 학습 및 구동(main.py) 테스트 결과, BCE Loss function 그래프 기록에서 50 epoch의 학습 결과 값이 수렴한다고 하기에는 기울기가 있어 지속적으로 하향하는 추세를 보임. 따라서 다음 버전의 방향성을 다음과 같이 제시함.
        - epoch hyperparameter 더 높게 설정
        - 데이터 수집 및 전처리 프로세스 점검
        이는 Loss function 지표를 관찰한 결과, 기본적으로 loss function이 특정 값으로 수렴하는 추세여야 함+불규칙한 진동이 있을 순 있지만 최대 진폭의 정도가 일정 수준 이하여야 함+수렴하는 값 부근의 진동 범위를 넘어 더 낮은 값으로 튀는 경우가 없어야 함을 만족해야 적어도 해당 학습 데이터에서의 라벨링 학습이 적절히 진행되었다고 볼 수 있다고 판단함. 그리고 Loss function이 모델의 일반적인 라벨링 성능을 근사해서 나타낼 때, 신뢰도를 가지려면 데이터의 양과 질(실 사용에서 나타날 수 있는 다양한 패턴 정보)을 높여야 한다고 판단하였다. 현재 특정 폴더에 선별한 30+30=60장의 사진을 직접 넣는 방식으로 데이터 수집을 진행하고 있다. 데디터의 양과 질을 높이기 위해 데이터 수집 및 전처리 알고리즘에 변화를 주어야겠다고 판단하였다.

## [v0.6.2] - 2026-10-13

### 1. 업데이트 내용

#### 프로젝트 정체성 재정의

- 프로젝트 목표를 "AGI 개발"에서 **"특정 인물일 확률을 실시간으로 촬영하며 알려주는 탐지기 프로그램"** 으로 변경
- GitHub 저장소명을 `target_detector`로 변경

#### `ai_rule.md` 신규 정책 적용

- 6단계 개발 프로세스(Step 1~6) 도입
- 단계별 산출물(`algorithm_spec.md`, `programming_plan.md`, `report.md`, `README.md`, `CHANGELOG.md`) 체계 확립

#### 문서 체계 재구성

- `algorithm_spec.md` Section 1 명칭을 `Model Definition` → `Program Definition` 으로 변경
- `algorithm_spec.md` Section 1.1 구조에 최종 출력(확률) 명시
- `algorithm_spec.md` Section 1.2 및 2.1의 Stage 1/Stage 2 입출력 명세 보강
- MAE 지표 삭제 (16-class MLP 환경에서 도입된 유산, 이진 분류에서는 불필요)
- `algorithm_spec.md` Section 6(구현 코드 구조 설계)을 `programming_plan.md` 로 이관

#### 신규 문서

- `programming_plan.md` 최초 생성 (Repository 구조, 의존성, 모듈/클래스 명세)
- `report.md` 최초 생성 (구현 일치성, 테스트 결과)
- `requirements.txt` 최초 생성

#### 코드 개선

- 옵티마이저를 Adam(lr=0.01)에서 **SGD(lr=1e-3)로 변경**. 이로 인해 step당 연산량이 감소하여 학습 속도 향상
- MAE 지표 및 관련 로깅 제거, BCE 단일 손실로 단순화
- UI 버그 수정: 캐시 변수 도입으로 확률 표시 안정화, 얼굴 검출 영역 시각화 복원
- `FaceDetector.detect_bbox`, `FaceRecognitionPipeline.predict_with_bbox` 메서드 추가