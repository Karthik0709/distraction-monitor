import statistics
import time

import cv2
import numpy as np

from face_tracker import FaceTracker

PITCH_DOWN_THRESHOLD = -10.0
PITCH_UP_THRESHOLD = 10.0
YAW_RIGHTSIDE_THRESHOLD = 30.0
YAW_LEFTSIDE_THRESHOLD = -30.0
MAX_CONSECUTIVE_FAILURES = 90

STAGE_SECONDS = 5.0     # hold time per calibration stage - 3 stages = ~15s total
YAW_MARGIN_DEGREES = 5.0  # buffer past the exact edges so hitting the boundary itself isn't flagged
MIN_ENVELOPE_WIDTH = 15.0  # if left/right calibration points end up this close, the user probably didn't turn


class HeadPoseEstimator():

    def __init__(self):
        self.real_world_corners = np.float32([[0.0, 0.0, 0.0],[0.0, -330.0, -65.0],
                            [-225.0,  170.0, -135.0],[225.0,  170.0, -135.0],
                            [-150.0, -150.0, -125.0],[150.0, -150.0, -125.0]])
        self.neutral_pitch = None
        self.neutral_yaw = None
        self.min_yaw = None
        self.max_yaw = None
        self._mode = None  # "point" (single monitor) or "range" (multi monitor) - set by whichever calibrate_* is called

    def calibrate_neutral(self, face_tracker, cap, num_samples=10):
        # single-monitor mode: one neutral point, fixed +/- threshold either side
        samples = []
        while len(samples) < num_samples:
            success, frame = cap.read()
            if not success:
                continue
            result = face_tracker.get_face_coordinates(frame=frame)
            if result is None:
                continue
            points, _ = result
            if points is None:
                continue
            pose = self.estimate_rotation(frame, points)
            if pose is not None:
                samples.append(pose)

        pitches, yaws = zip(*samples)
        self.neutral_pitch = sum(pitches) / len(pitches)
        self.neutral_yaw = sum(yaws) / len(yaws)
        pitch_spread = max(pitches) - min(pitches)
        yaw_spread = max(yaws) - min(yaws)
        print(f"\nNeutral values - pitch={self.neutral_pitch:.1f}, yaw={self.neutral_yaw:.1f} "
              f"(sample spread: pitch={pitch_spread:.1f}, yaw={yaw_spread:.1f})\n")
        if pitch_spread > 15 or yaw_spread > 15:
            # noisy calibration produces a bad baseline that every later frame gets compared against
            print("WARNING: calibration samples were noisy/inconsistent - "
                  "hold still and look at the screen during calibration, then recalibrate.")
        self._mode = "point"

    def calibrate_range(self, face_tracker, cap, on_status=None, stage_seconds=STAGE_SECONDS):
        # multi-monitor mode: record yaw at each monitor edge and only flag distraction outside that envelope,
        # instead of one neutral point with a fixed +/- threshold
        on_status = on_status or (lambda msg: None)
        stages = [
            ("center", "Look at your main/center working position"),
            ("left", "Now turn to the FAR LEFT edge of your work area (leftmost monitor)"),
            ("right", "Now turn to the FAR RIGHT edge of your work area (rightmost monitor)"),
        ]

        stage_pitch = {}
        stage_yaw = {}
        for key, prompt in stages:
            on_status(f"{prompt} - hold still for {int(stage_seconds)}s...")
            pitches, yaws = self._collect_pose_samples(face_tracker, cap, stage_seconds)
            stage_pitch[key] = statistics.median(pitches)
            stage_yaw[key] = statistics.median(yaws)
        on_status(None)

        self.neutral_pitch = stage_pitch["center"]
        self.min_yaw = min(stage_yaw["left"], stage_yaw["right"]) - YAW_MARGIN_DEGREES
        self.max_yaw = max(stage_yaw["left"], stage_yaw["right"]) + YAW_MARGIN_DEGREES

        print(f"\nCalibrated range - neutral_pitch={self.neutral_pitch:.1f}, "
              f"yaw envelope=[{self.min_yaw:.1f}, {self.max_yaw:.1f}]\n")
        if (self.max_yaw - self.min_yaw) < MIN_ENVELOPE_WIDTH:
            print("WARNING: left/right calibration points were very close together - "
                  "make sure you actually turned toward each monitor edge, then recalibrate.")
        self._mode = "range"

    def _collect_pose_samples(self, face_tracker, cap, duration_seconds):
        pitches, yaws = [], []
        start = time.time()
        while time.time() - start < duration_seconds:
            success, frame = cap.read()
            if not success:
                continue
            result = face_tracker.get_face_coordinates(frame=frame)
            if result is None:
                continue
            points, _ = result
            if points is None:
                continue
            pose = self.estimate_rotation(frame, points)
            if pose is not None:
                pitches.append(pose[0])
                yaws.append(pose[1])

        if not pitches:
            raise RuntimeError("No face detected during calibration stage - retry with better lighting/framing.")
        return pitches, yaws

    def estimate_rotation(self, frame,points) -> tuple:
        points = np.array(points, dtype=np.float32)
        frame_height, frame_width, c = frame.shape
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        center_x, center_y = center
        camera_matrix = np.array([[focal_length, 0, center_x],
                           [0, focal_length, center_y],
                           [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros(4)
        # fresh solve every frame - no useExtrinsicGuess warm-start, since a bad frame
        # would otherwise permanently drag every later estimate off with it
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=self.real_world_corners, imagePoints=points,
            cameraMatrix=camera_matrix, distCoeffs=dist_coeffs,
        )
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
            pitch, yaw, roll = retval
            if pitch > 90:
                pitch = 180 - pitch
            elif pitch < -90:
                pitch = -180 - pitch
            return pitch,yaw
        return None

    def is_face_away(self,frame,points) -> bool | None:
        result = self.estimate_rotation(frame=frame,points=points)
        if result is not None:
            pitch,yaw = result
            pitch_delta = pitch - self.neutral_pitch
            pitch_away = pitch_delta < PITCH_DOWN_THRESHOLD or pitch_delta > PITCH_UP_THRESHOLD

            if self._mode == "range":
                yaw_away = yaw < self.min_yaw or yaw > self.max_yaw
            else:
                yaw_delta = yaw - self.neutral_yaw
                yaw_away = yaw_delta > YAW_RIGHTSIDE_THRESHOLD or yaw_delta < YAW_LEFTSIDE_THRESHOLD

            return pitch_away or yaw_away
        else:
            return None

def main():
    cap = cv2.VideoCapture(0)
    facetracker = FaceTracker(0.3, 0.5)
    pose_estimator = HeadPoseEstimator()
    consecutive_failures=0
    pose_estimator.calibrate_neutral(face_tracker=facetracker,cap=cap)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        result = facetracker.get_face_coordinates(frame=frame)

        status_text = ""

        if result is not None:
            points, coordinates = result
            if points is None:
                consecutive_failures += 1
                status_text = "no face detected"
            else:
                result = pose_estimator.is_face_away(frame, points)
                if result is None:
                    consecutive_failures += 1
                    status_text = "pose estimation failed"
                elif result:
                    consecutive_failures = 0
                    status_text = "face is away"
                else:
                    consecutive_failures = 0
                    status_text = "face forward"

            if coordinates is not None:
                cv2.rectangle(frame, (coordinates["x_min"], coordinates["y_min"]),
                              (coordinates["x_max"], coordinates["y_max"]), (0, 255, 0), 2)
        else:
            consecutive_failures += 1
            status_text = "no face detected"

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError(f"{MAX_CONSECUTIVE_FAILURES} consecutive frames failed - stopping.")

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"consecutive_failures={consecutive_failures}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Head Pose Test", frame)
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
