import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import time


class HandTracker:

    def __init__(self,min_detection_confidence: float, min_tracking_confidence: float):
        self.latest_result = None
        self.model_path = "C:\\Users\\Jayakumar\\Downloads\\Karthik_Projects\\distraction-monitor\\model\\hand_landmarker.task"
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._on_result,   # MediaPipe calls this for us later
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)


    def _on_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def submit_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def get_latest_landmarks(self):
        """Non-blocking: whatever the most recent callback gave us, or None."""
        if self.latest_result and self.latest_result.hand_landmarks:
            return self.latest_result.hand_landmarks
        return None

    def get_hand_coordinates(self,frame):

        self.submit_frame(frame)
        landmarks_list = self.get_latest_landmarks()

        if landmarks_list is None:
            return None 
        
        h, w, c = frame.shape
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h

        for hand in landmarks_list:
            for landmark in hand:
                # print(landmark.x, landmark.y, landmark.z)
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
        return {
            "x_max":x_max,
            "y_max":y_max,
            "x_min":x_min,
            "y_min":y_min
        }


def main():
    cap = cv2.VideoCapture(0)
    handtracker = HandTracker(0.3,0.5)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
    
        corodinates = handtracker.get_hand_coordinates(frame=frame)
        if corodinates is not None:
            cv2.rectangle(
                frame,
                (corodinates["x_min"], corodinates["y_min"]),
                (corodinates["x_max"], corodinates["y_max"]),
                (0, 255, 0), 2)
        cv2.imshow('MediaPipe Tasks Async Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()