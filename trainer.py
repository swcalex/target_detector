import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN


class ModelTrainer:
    """Stage 2 CNN 학습 관리"""

    def __init__(self, model: SimpleCNN, lr: float = 1e-3):
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr)

    def compute_loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE 손실 계산"""
        return self.criterion(y_pred, target)

    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> dict:
        """1회 학습 스텝 수행"""
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.compute_loss(y_pred, target)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}