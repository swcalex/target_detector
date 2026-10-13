import cv2
import numpy as np
import mediapipe as mp
import torch
from typing import Optional


class FaceDetector:
    """MediaPipe Face Detection을 이용한 얼굴 검출 및 전처리"""

    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0: short-range (2m 이내)
            min_detection_confidence=min_detection_confidence
        )

    def detect_and_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        BGR 프레임에서 얼굴을 검출하고 크롭한다.
        얼굴이 없으면 None 반환.
        """
        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None

        # 첫 번째 얼굴만 사용
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # 경계 보정
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        width = min(w - x_min, width)
        height = min(h - y_min, height)

        if width <= 0 or height <= 0:
            return None

        cropped = frame[y_min:y_min + height, x_min:x_min + width]
        return cropped

    def preprocess(self, cropped_face: np.ndarray) -> torch.Tensor:
        """
        크롭된 얼굴 이미지를 Stage 2 입력 텐서로 변환한다.
        - 112x112 Bilinear Resize
        - [0, 1] Float32 정규화
        - Transpose (HWC → CHW)
        - 차원 확장 (1, 3, 112, 112)
        """
        # BGR → RGB (MediaPipe는 RGB 기준)
        rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        # Resize
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        # 정규화
        normalized = resized.astype(np.float32) / 255.0
        # Transpose & 차원 확장
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)  # (1,3,112,112)
        return tensor

    def detect_bbox(self, frame: np.ndarray):
        """MediaPipe로 bounding box 검출 (시각화용).
        반환: (x_min, y_min, width, height) 또는 None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape
        x_min = max(0, int(bbox.xmin * w))
        y_min = max(0, int(bbox.ymin * h))
        width = min(w - x_min, int(bbox.width * w))
        height = min(h - y_min, int(bbox.height * h))

        if width <= 0 or height <= 0:
            return None

        return (x_min, y_min, width, height)

    def detect_and_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """bbox 검출 후 크롭. 얼굴 없으면 None."""
        bbox = self.detect_bbox(frame)
        if bbox is None:
            return None
        x_min, y_min, w, h = bbox
        return frame[y_min:y_min + h, x_min:x_min + w]