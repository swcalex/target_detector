import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Stage 2 이진 분류 CNN"""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature Extractor
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 200)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(200, 1)
        self.sigmoid2 = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # Conv Layers: He Normal
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # FC Layers: Xavier Normal
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        return x  # (1, 1)