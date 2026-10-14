import cv2
import time
from pipeline import FaceRecognitionPipeline


def main():
    pipeline = FaceRecognitionPipeline(weights_path="data/weights.pth")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    fps = 2
    interval = 1.0 / fps
    last_time = 0.0

    # 2fps 주기로만 갱신되는 캐시 상태
    last_prob = 0.0
    last_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_time >= interval:
            last_prob, last_bbox = pipeline.predict_with_bbox(frame)
            last_time = current_time

        # 매 프레임: 마지막 상태를 그대로 표시
        if last_bbox is not None:
            x_min, y_min, w, h = last_bbox
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + w, y_min + h), (0, 255, 0), 2)
            text = f"Target: {last_prob * 100:.1f}%"
            color = (0, 255, 0)
        else:
            text = "No face detected"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Target Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()