import logging

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

# Default: stock YOLOv8n COCO weights (has a "cell phone" class out of the box,
# downloads automatically on first run) - no training required to try this out.
# After fine-tuning via train_phone_yolo.py, point this at
# runs/detect/phone_detector/weights/best.pt instead.
MODEL_PATH = "C:\\Users\\Jayakumar\\Downloads\\Karthik_Projects\\distraction-monitor\\model\\yolov8n.pt"

TARGET_LABEL_SUBSTRING = "phone"  # matches both COCO's "cell phone" and the
                                   # fine-tuned model's single "phone" class
CONFIDENCE_THRESHOLD = 0.3


class PhoneDetector:
    """Runs a YOLOv8 model on a frame and reports the highest-confidence
    'phone' detection, if any. Same detect() interface as the earlier
    mediapipe-backed version, so monitor.py doesn't need to change."""

    def __init__(self, model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD,
                 target_label_substring=TARGET_LABEL_SUBSTRING):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_label_substring = target_label_substring.lower()

    def detect(self, frame_bgr):
        """Returns (box, score) for the best phone detection, or None.
        box is (x_min, y_min, x_max, y_max) in pixel coordinates.
        ultralytics accepts a raw BGR numpy array directly - no manual
        color conversion needed, unlike the mediapipe Tasks API."""
        results = self.model.predict(frame_bgr, conf=self.confidence_threshold, verbose=False)

        best = None
        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls[0])].lower()
                if self.target_label_substring not in label:
                    continue

                score = float(box.conf[0])
                x_min, y_min, x_max, y_max = (int(v) for v in box.xyxy[0])

                if best is None or score > best[1]:
                    best = ((x_min, y_min, x_max, y_max), score)

        return best


def main():
    """Smoke test - hold a phone up to the webcam. ESC to quit."""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    detector = PhoneDetector()
    logger.info("Phone detector running (model=%s). ESC to quit.", MODEL_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        result = detector.detect(frame)
        if result is not None:
            (x_min, y_min, x_max, y_max), score = result
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(frame, f"phone {score:.2f}", (x_min, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Phone Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
