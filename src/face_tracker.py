import cv2
import mediapipe as mp
import time


class FaceTracker:

    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float):
        self.latest_result = None
        self.face_indices = [1,152,33,263,61,291]
        self.model_path = "C:\\Users\\Jayakumar\\Downloads\\Karthik_Projects\\distraction-monitor\\model\\face_landmarker_v2.task"
        
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,  # Set maximum number of faces to detect
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            result_callback=self._on_result,   # MediaPipe calls this asynchronously
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def submit_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def get_latest_landmarks(self):
        """Non-blocking: whatever the most recent callback gave us, or None."""
        if self.latest_result and self.latest_result.face_landmarks:
            return self.latest_result.face_landmarks
        return None

    def get_face_coordinates(self, frame):
        self.submit_frame(frame)
        landmarks_list = self.get_latest_landmarks()

        if landmarks_list is None:
            return None 
        
        h, w, c = frame.shape
        face = landmarks_list[0]  # single face, since num_faces=1
        #print(f"Face idx -- \n {face}")
        points = []
        for idx in self.face_indices:
            landmark = face[idx]
            points.append((landmark.x * w, landmark.y * h))  # keep as float here, not int - solvePnP wants float32

        print(f" points finally \n {points}")
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h

        # landmarks_list contains an array of faces, where each face contains landmarks
        for face in landmarks_list:
            for landmark in face:
                x, y = int(landmark.x * w), int(landmark.y * h)
                #print(f"Landmark object - \n {landmark}")
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
                    
            cv2.imshow('MediaPipe Tasks Async Face Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break
            
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()