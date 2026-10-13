import numpy as np
import torch
from face_detector import FaceDetector
from model import SimpleCNN


class FaceRecognitionPipeline:
    """Stage 1 + Stage 2 통합 추론 파이프라인"""

    def __init__(self, weights_path: str = "weights.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.detector = FaceDetector()
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, frame: np.ndarray) -> float:
        """
        프레임을 입력받아 특정 인물일 확률을 반환한다.
        얼굴 미검출 시 0.0 반환.
        """
        cropped = self.detector.detect_and_crop(frame)
        if cropped is None:
            return 0.0

        tensor = self.detector.preprocess(cropped).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        return output.item()

    def predict_with_bbox(self, frame: np.ndarray):
        """확률과 bbox를 함께 반환. 얼굴 미검출 시 (0.0, None)."""
        bbox = self.detector.detect_bbox(frame)
        if bbox is None:
            return 0.0, None

        x_min, y_min, w, h = bbox
        cropped = frame[y_min:y_min + h, x_min:x_min + w]
        tensor = self.detector.preprocess(cropped).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        return output.item(), bbox

    def predict(self, frame: np.ndarray) -> float:
        prob, _ = self.predict_with_bbox(frame)
        return prob