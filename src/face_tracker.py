from pathlib import Path

import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACE_LANDMARKER_MODEL_PATH = PROJECT_ROOT / "model" / "face_landmarker_v2.task"


class FaceTracker:
    # runs in synchronous IMAGE mode so every call returns landmarks for the exact frame passed in, no stale results

    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float):
        self.face_indices = [1,152,33,263,61,291]
        self.model_path = str(FACE_LANDMARKER_MODEL_PATH)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def get_face_coordinates(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks_list = result.face_landmarks
        h, w, c = frame.shape
        face = landmarks_list[0]
        points = []
        for idx in self.face_indices:
            landmark = face[idx]
            points.append((landmark.x * w, landmark.y * h))  # stays float - solvePnP needs float32, not int

        x_max = 0
        y_max = 0
        x_min = w
        y_min = h

        for face in landmarks_list:
            for landmark in face:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

        return points,{
            "x_max": x_max,
            "y_max": y_max,
            "x_min": x_min,
            "y_min": y_min
        }


def main():
    cap = cv2.VideoCapture(0)
    facetracker = FaceTracker(0.3, 0.5)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        result = facetracker.get_face_coordinates(frame=frame)
        print(f"Raw returned value: {result}")
        print(f"Data type: {type(result)}")
        if result is not None:
            points, coordinates = result
            if coordinates is not None:
                cv2.rectangle(
                    frame,
                    (coordinates["x_min"], coordinates["y_min"]),
                    (coordinates["x_max"], coordinates["y_max"]),
                    (0, 255, 0), 2)

            cv2.imshow('MediaPipe Tasks Face Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
