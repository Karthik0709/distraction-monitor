import logging
from pathlib import Path

import cv2

from audio_player import AudioLoopPlayer
from face_tracker import FaceTracker
from gif_player import GifPlayer
from head_pose_estimator import HeadPoseEstimator
from phone_detector import PhoneDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

# This file lives at <repo_root>/src/monitor.py, so the repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Drop this file in yourself - any gif works. Point this at it.
GIF_PATH = str(PROJECT_ROOT / "assets" / "Dei_parama_padi_da_Tamil_meme_templates.mp4")
# Debounce: require N consecutive frames before flipping state, so a single
# flickery misdetection doesn't start/stop the gif. Exit threshold is higher
# than enter - easier to trigger the nag, harder to dismiss it by accident.
ENTER_DISTRACTION_FRAMES = 180
EXIT_DISTRACTION_FRAMES = 15

WEBCAM_WINDOW = "Distraction Monitor"
GIF_WINDOW = "STOP SCROLLING"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    face_tracker = FaceTracker(min_detection_confidence=0.3, min_tracking_confidence=0.5)
    pose_estimator = HeadPoseEstimator()
    phone_detector = PhoneDetector()
    gif_player = GifPlayer(GIF_PATH)
    audio_player = AudioLoopPlayer(GIF_PATH)

    logger.info("Calibrating neutral head pose - look at the screen normally...")
    pose_estimator.calibrate_neutral(face_tracker, cap)
    logger.info("Calibration done. Monitoring started. ESC to quit.")

    distracted_streak = 0
    clear_streak = 0
    is_distracted = False
    gif_window_open = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning("Frame grab failed - skipping.")
            continue

        # --- signal 1: face turned/tilted away from the screen ---
        face_away = None
        face_box = None
        face_result = face_tracker.get_face_coordinates(frame)
        if face_result is not None:
            points, face_box = face_result
            if points is not None:
                face_away = pose_estimator.is_face_away(frame, points)

        # --- signal 2: phone visible in frame ---
        phone_result = phone_detector.detect(frame)
        phone_box, phone_score = (phone_result if phone_result is not None else (None, None))

        distracted_now = bool(face_away) or phone_box is not None

        if distracted_now:
            distracted_streak += 1
            clear_streak = 0
        else:
            clear_streak += 1
            distracted_streak = 0

        if not is_distracted and distracted_streak >= ENTER_DISTRACTION_FRAMES:
            is_distracted = True
            audio_player.start()
            logger.info("Distraction detected (face_away=%s, phone=%s) - nag started.", face_away, phone_box is not None)
        elif is_distracted and clear_streak >= EXIT_DISTRACTION_FRAMES:
            is_distracted = False
            gif_player.reset()
            audio_player.stop()
            logger.info("Back on task - nag stopped.")

        # --- draw debug overlay on the webcam window ---
        display = frame.copy()
        if face_box is not None:
            cv2.rectangle(display, (face_box["x_min"], face_box["y_min"]),
                          (face_box["x_max"], face_box["y_max"]), (0, 255, 0), 2)
        if phone_box is not None:
            x_min, y_min, x_max, y_max = phone_box
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(display, f"phone {phone_score:.2f}", (x_min, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status = (f"face_away={face_away} phone={phone_box is not None} "
                  f"distracted={is_distracted} (enter_streak={distracted_streak}, exit_streak={clear_streak})")
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow(WEBCAM_WINDOW, display)

        # --- gif window follows is_distracted, independent of the webcam window ---
        if is_distracted:
            cv2.imshow(GIF_WINDOW, gif_player.next_frame())
            gif_window_open = True
        elif gif_window_open:
            cv2.destroyWindow(GIF_WINDOW)
            gif_window_open = False

        if cv2.waitKey(1) & 0xFF == 27:
            break

    audio_player.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
