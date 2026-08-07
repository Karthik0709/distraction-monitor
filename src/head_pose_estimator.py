import cv2
import numpy as np

from face_tracker import FaceTracker

PITCH_DOWN_THRESHOLD = -10.0  # Flag if looking down past -5 degrees
PITCH_UP_THRESHOLD = 10.0 # Flag if looking up past 12 degrees
YAW_RIGHTSIDE_THRESHOLD = 30.0 # Flag if turning sideways past +/- 20 degrees
YAW_LEFTSIDE_THRESHOLD = -30.0 # Flag if turning sideways past +/- 20 degrees
MAX_CONSECUTIVE_FAILURES = 90

class HeadPoseEstimator():

    def __init__(self):
        self.real_world_corners = np.float32([[0.0, 0.0, 0.0],[0.0, -330.0, -65.0],
                            [-225.0,  170.0, -135.0],[225.0,  170.0, -135.0],
                            [-150.0, -150.0, -125.0],[150.0, -150.0, -125.0]])
        self.last_rvec = None
        self.last_tvec = None
        self.neutral_pitch = None
        self.neutral_yaw = None

    def calibrate_neutral(self, face_tracker, cap, num_samples=10):
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
        print(f"\n Neutral  values - {self.neutral_pitch} , {self.neutral_yaw} \n")

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
        if self.last_rvec is not None and self.last_tvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=self.real_world_corners, imagePoints=points,
                cameraMatrix=camera_matrix, distCoeffs=dist_coeffs,
                rvec=self.last_rvec, tvec=self.last_tvec,
                useExtrinsicGuess=True,
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=self.real_world_corners, imagePoints=points,
                cameraMatrix=camera_matrix, distCoeffs=dist_coeffs,
            )
        if success:
            self.last_rvec, self.last_tvec = rvec, tvec
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
            yaw_delta = yaw - self.neutral_yaw
            print(f" Printing pitch and Yaw !!! \n {pitch,yaw} ")
            print(f"\n Printing Delta pitch and Yaw !!! \n {pitch_delta,yaw_delta} ")
            return ((pitch_delta < PITCH_DOWN_THRESHOLD or pitch_delta > PITCH_UP_THRESHOLD) or (abs(yaw_delta) > YAW_RIGHTSIDE_THRESHOLD or abs(yaw_delta) < YAW_LEFTSIDE_THRESHOLD))
        else:
            return None  #raise RuntimeError("Unable to get rotation matrix from estimation_rotation function")
    
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

        status_text = ""  # FIX: default so putText below can never see an unset variable

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
                    print("Face is away !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    # FIX: this branch (detected + not away) was previously missing,
                    # leaving status_text unset whenever you were just looking forward
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
            raise RuntimeError(
                f"{MAX_CONSECUTIVE_FAILURES} consecutive frames failed - "
                "stopping. (Note: this crash-on-failure behavior is fine for "
                "this smoke test but should NOT carry into live.py as-is.)"
            )

        # FIX: unindented out of `if result is not None:` so the window is shown
        # and waitKey pumped every iteration, even on frames with no face at all
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








# import cv2
# import numpy as np

# from face_tracker import FaceTracker

# PITCH_DOWN_THRESHOLD = -15.0  # Flag if looking down past -15 degrees
# YAW_SIDEWAYS_THRESHOLD = 20.0 # Flag if turning sideways past +/- 20 degrees
# MAX_CONSECUTIVE_FAILURES = 5

# class HeadPoseEstimator():

#     def __init__(self):
#         self.real_world_corners = np.float32([[0.0, 0.0, 0.0],[0.0, -330.0, -65.0],
#                             [-225.0,  170.0, -135.0],[225.0,  170.0, -135.0],
#                             [-150.0, -150.0, -125.0],[150.0, -150.0, -125.0]])

#     def estimate_rotation(self, frame,points) -> tuple:
#         points = np.array(points, dtype=np.float32)
#         frame_height, frame_width, c = frame.shape
#         focal_length = frame_width
#         center = (frame_width / 2, frame_height / 2)
#         center_x, center_y = center 
#         camera_matrix = np.array([[focal_length, 0, center_x],
#                            [0, focal_length, center_y],
#                            [0, 0, 1]], dtype=np.float32)
#         dist_coeffs = np.zeros(4)
#         success,rvec,tvec = cv2.solvePnP(objectPoints=self.real_world_corners,imagePoints=points,cameraMatrix=camera_matrix,distCoeffs=dist_coeffs)
#         if success:
#             rotation_matrix, _ = cv2.Rodrigues(rvec)
#             retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
#             pitch, yaw, roll = retval
#             return pitch,yaw
#         return None

#     def is_face_away(self,frame,points) -> bool | None:
#         result = self.estimate_rotation(frame=frame,points=points)
#         if result is not None:
#             pitch,yaw = result
#             if pitch < PITCH_DOWN_THRESHOLD or abs(yaw) > YAW_SIDEWAYS_THRESHOLD:
#                 return True
#             return False
#         else:
#             return None  #raise RuntimeError("Unable to get rotation matrix from estimation_rotation function")
    
# def main():
#     cap = cv2.VideoCapture(0)
#     facetracker = FaceTracker(0.3, 0.5)
#     pose_estimator = HeadPoseEstimator()
#     consecutive_failures=0
    
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#         result = facetracker.get_face_coordinates(frame=frame)
#         print(f"Raw returned value: {result}")
#         print(f"Data type: {type(result)}")
#         if result is not None:
#             points, coordinates = result
#             if points is None:
#                 consecutive_failures += 1
#                 status_text = "no face detected"
#             else:
#                 result = pose_estimator.is_face_away(frame, points)
#                 if result is None:
#                     consecutive_failures += 1
#                     status_text = "pose estimation failed"
#                 elif result:
#                     print("Face is away !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
#             if coordinates is not None:
#                 cv2.rectangle(frame, (coordinates["x_min"], coordinates["y_min"]),
#                               (coordinates["x_max"], coordinates["y_max"]), (0, 255, 0), 2)
#             if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 raise RuntimeError(
#                     f"{MAX_CONSECUTIVE_FAILURES} consecutive frames failed - "
#                     "stopping. (Note: this crash-on-failure behavior is fine for "
#                     "this smoke test but should NOT carry into live.py as-is.)"
#                 )
    
#             cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, (0, 255, 255), 2)
#             cv2.putText(frame, f"consecutive_failures={consecutive_failures}",
#                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
#             cv2.imshow("Head Pose Test", frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
