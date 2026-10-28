import os
import sys
import time
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 호환
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from face_detector import FaceDetector
from model import SimpleCNN
from trainer import ModelTrainer
from convergence import judge_block, format_record


# --- algorithm_spec.md Section 2.2.1 Hyperparameter Setting ---
BATCH_SIZE = 9
BLOCK_EPOCHS = 50      # 학습을 50 epoch 단위 블록으로 반복 수행
SAFETY_CAP = 1000      # 안전 상한: 총 학습 epoch가 1000에 도달하면 강제 종료
LEARNING_RATE = 1e-3
# --------------------------------------------------------------

LOG_INTERVAL = 10          # Section 3.2: 10 epoch 구간 소요 시간 기록
LOG_DIR = "data/log"
WEIGHTS_PATH = "data/weights.pth"
REPORT_PATH = os.path.join(LOG_DIR, "convergence_report.txt")
TIME_LOG_PATH = os.path.join(LOG_DIR, "train_time_log.txt")
LOSS_CURVE_PATH = os.path.join(LOG_DIR, "loss_curve.png")


def load_images(data_dir: str, label: float):
    """디렉터리 내 이미지를 로드하고 Stage 1 전처리 및 6배 증강"""
    detector = FaceDetector()
    images = []
    labels = []

    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        cropped = detector.detect_and_crop(img)
        if cropped is None:
            continue

        # 전처리 (numpy 상태로 증강)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0  # (112,112,3)

        # 6배 증강
        for aug in augment(normalized):
            tensor = torch.from_numpy(aug).permute(2, 0, 1)  # (3,112,112)
            images.append(tensor)
            labels.append(torch.tensor([label], dtype=torch.float32))

    return torch.stack(images), torch.stack(labels)


def augment(img: np.ndarray):
    """algorithm_spec.md Section 2.4.3의 6가지 증강 기법"""
    # 1. 원본
    yield img
    # 2. 수평 반전
    yield img[:, ::-1, :].copy()
    # 3. 밝기 증가
    yield np.clip(img * 1.25, 0.0, 1.0)
    # 4. 밝기 감소
    yield img * 0.75
    # 5. 대비 증가
    yield np.clip((img - 0.5) * 1.3 + 0.5, 0.0, 1.0)
    # 6. 복합 증강 (Flip + Brightness)
    flip = img[:, ::-1, :].copy()
    yield np.clip(flip * 1.15, 0.0, 1.0)


def train():
    # 콘솔 인코딩 안전화 (리다이렉션 환경 대비)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # 디렉터리 보장
    os.makedirs("data", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 데이터 로드 (기존 로직 그대로)
    target_imgs, target_labels = load_images("data/target", 1.0)
    non_target_imgs, non_target_labels = load_images("data/non_target", 0.0)

    X = torch.cat([target_imgs, non_target_imgs], dim=0)
    y = torch.cat([target_labels, non_target_labels], dim=0)

    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    # 모델 및 학습기 초기화
    model = SimpleCNN()
    trainer = ModelTrainer(model, lr=LEARNING_RATE)

    # --- 지표 수집 상태 ---
    loss_history = []       # Section 3.3: 전체 epoch 학습 Loss
    report_lines = []       # Section 3.4: 블록별 판독 레코드
    time_log_lines = []     # Section 3.2: 10 epoch 구간 시간 라인
    total_epochs = 0
    stop_reason = None
    block_index = 0         # 0-based 현재 블록 인덱스

    train_start = time.time()
    interval_start = time.time()

    while True:
        block_losses = []
        for _ in range(BLOCK_EPOCHS):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                result = trainer.train_step(x_batch, y_batch)
                total_loss += result["loss"]
            avg_loss = total_loss / len(loader)
            block_losses.append(avg_loss)
            loss_history.append(avg_loss)
            total_epochs += 1
            print(f"Epoch {total_epochs}, Loss: {avg_loss:.4f}")

            # Section 3.2: 10 epoch 구간마다 소요 시간 기록
            if total_epochs % LOG_INTERVAL == 0:
                elapsed = time.time() - interval_start
                time_log_lines.append(
                    f"[{total_epochs} epoch] elapsed: {elapsed:.2f}s"
                )
                print(time_log_lines[-1])
                interval_start = time.time()

        # Section 3.4: 블록 판독 및 리포트 레코드 누적
        record = judge_block(block_index, block_losses)
        report_lines.append(format_record(record))
        print(report_lines[-1])
        block_index += 1

        # Section 3.4: 학습 종료 판정
        if record["stop"]:
            stop_reason = "수렴 조건 만족 (기준 1 & 2)"
            break
        if total_epochs >= SAFETY_CAP:
            stop_reason = "안전 상한 도달"
            break

    # Section 3.2: 총 학습 소요 시간 및 1 epoch당 평균 시간
    # (부분 구간 포함: train_start 이후 전 구간을 총 시간에 합산)
    total_time = time.time() - train_start
    avg_time_per_epoch = total_time / total_epochs
    time_log_lines.append(f"Total time: {total_time:.2f} s")
    time_log_lines.append(f"Average time per epoch: {avg_time_per_epoch:.2f} s")
    with open(TIME_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(time_log_lines) + "\n")
    print(time_log_lines[-2])
    print(time_log_lines[-1])

    # Section 3.4: 리포트 저장 (블록별 레코드 + 종료 요약)
    report_lines.append(
        f"[Termination] total_epochs={total_epochs}, reason={stop_reason}"
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(report_lines[-1])

    # Section 3.3: Loss 곡선 그래프 (학습 종료 시 1회 생성)
    plt.figure()
    plt.plot(range(1, total_epochs + 1), loss_history, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()

    # Section 2.4.4: 가중치 저장 (학습 종료 시점)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"{WEIGHTS_PATH} saved.")


if __name__ == "__main__":
    train()
