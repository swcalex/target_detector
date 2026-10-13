import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from face_detector import FaceDetector
from model import SimpleCNN
from trainer import ModelTrainer


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

        # 전처리 (텐서 변환 전, numpy 상태로 증강)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0  # (112,112,3)

        # 6배 증강
        augmented = augment(normalized)
        for aug in augmented:
            tensor = torch.from_numpy(aug).permute(2, 0, 1)  # (3,112,112)
            images.append(tensor)
            labels.append(torch.tensor([label], dtype=torch.float32))

    return torch.stack(images), torch.stack(labels)


def augment(img: np.ndarray):
    """algorithm_spec.md 5.3의 6가지 증강 기법"""
    # 1. 원본
    yield img

    # 2. 수평 반전
    yield img[:, ::-1, :].copy()

    # 3. 밝기 증가
    bright = np.clip(img * 1.25, 0.0, 1.0)
    yield bright

    # 4. 밝기 감소
    dark = img * 0.75
    yield dark

    # 5. 대비 증가
    contrast = np.clip((img - 0.5) * 1.3 + 0.5, 0.0, 1.0)
    yield contrast

    # 6. 복합 증강 (Flip + Brightness)
    flip = img[:, ::-1, :].copy()
    flip_bright = np.clip(flip * 1.15, 0.0, 1.0)
    yield flip_bright


def train():
    # 데이터 로드
    target_imgs, target_labels = load_images("data/target", 1.0)
    non_target_imgs, non_target_labels = load_images("data/non_target", 0.0)

    X = torch.cat([target_imgs, non_target_imgs], dim=0)
    y = torch.cat([target_labels, non_target_labels], dim=0)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 모델 및 학습기 초기화
    model = SimpleCNN()
    trainer = ModelTrainer(model, lr=1e-3)

    # 학습 루프
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            result = trainer.train_step(x_batch, y_batch)
            total_loss += result["loss"]
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 가중치 저장
    torch.save(model.state_dict(), "weights.pth")
    print("weights.pth saved.")


if __name__ == "__main__":
    train()