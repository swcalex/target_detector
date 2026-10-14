import os
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


# --- algorithm_spec.md Section 2.2.1 Hyperparameter Setting ---
BATCH_SIZE = 9
EPOCHS = 50
LEARNING_RATE = 1e-3
# --------------------------------------------------------------

LOG_INTERVAL = 10          # Section 3.2: 10 epoch당 소요 시간 기록
LOG_DIR = "data/log"
WEIGHTS_PATH = "data/weights.pth"


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
    # 디렉터리 보장
    os.makedirs("data", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 데이터 로드
    target_imgs, target_labels = load_images("data/target", 1.0)
    non_target_imgs, non_target_labels = load_images("data/non_target", 0.0)

    X = torch.cat([target_imgs, non_target_imgs], dim=0)
    y = torch.cat([target_labels, non_target_labels], dim=0)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 모델 및 학습기 초기화
    model = SimpleCNN()
    trainer = ModelTrainer(model, lr=LEARNING_RATE)

    # --- Section 3.2 & 3.3 지표 수집 ---
    loss_history = []
    time_log_lines = []
    block_start = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            result = trainer.train_step(x_batch, y_batch)
            total_loss += result["loss"]
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Section 3.2: 10 epoch 블록마다 소요 시간 기록
        if (epoch + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - block_start
            line = f"[{epoch+1} epoch] elapsed: {elapsed:.2f}s"
            time_log_lines.append(line)
            print(line)
            block_start = time.time()

    # Section 3.2: 전체 epoch당 평균 시간
    block_total = sum(
        float(line.split("elapsed: ")[1].rstrip("s")) for line in time_log_lines
    )
    avg_time_per_epoch = block_total / EPOCHS
    avg_line = f"Average time per epoch: {avg_time_per_epoch:.2f}s"
    time_log_lines.append(avg_line)
    print(avg_line)

    # Section 3.2: 로그 파일 저장
    with open(os.path.join(LOG_DIR, "train_time_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(time_log_lines) + "\n")

    # Section 3.3: Loss 곡선 그래프 저장
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "loss_curve.png"))
    plt.close()

    # Section 2.4.4: 가중치 저장 (경로 변경)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"{WEIGHTS_PATH} saved.")


if __name__ == "__main__":
    train()