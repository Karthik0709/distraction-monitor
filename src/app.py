import logging
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from audio_player import AudioLoopPlayer
from face_tracker import FaceTracker
from gif_player import GifPlayer
from head_pose_estimator import HeadPoseEstimator
from phone_detector import PhoneDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

# This file lives at <repo_root>/src/app.py, so the repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GIF_PATH = str(PROJECT_ROOT / "assets" / "Dei_parama_padi_da_Tamil_meme_templates.mp4")

# Debounce: phone reacts fast (a handful of frames), face-away is slower/more
# forgiving since head pose is noisier. Matches the reviewed change to
# monitor.py - kept in sync here rather than importing monitor.py's module-
# level constants, since this file owns its own session loop.
PHONE_ENTER_FRAMES = 8
FACE_ENTER_FRAMES = 120
EXIT_DISTRACTION_FRAMES = 15

VIDEO_UPDATE_MS = 30


@dataclass
class SessionStats:
    start_time: float = 0.0
    end_time: float = 0.0
    distraction_events: int = 0
    total_distracted_seconds: float = 0.0
    _distraction_started_at: float | None = field(default=None, repr=False)

    def mark_distracted_start(self, now):
        if self._distraction_started_at is None:
            self._distraction_started_at = now
            self.distraction_events += 1

    def mark_distracted_end(self, now):
        if self._distraction_started_at is not None:
            self.total_distracted_seconds += now - self._distraction_started_at
            self._distraction_started_at = None

    def finalize(self, now):
        self.mark_distracted_end(now)
        self.end_time = now

    @property
    def total_seconds(self):
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def focused_seconds(self):
        return max(0.0, self.total_seconds - self.total_distracted_seconds)


class DistractionEngine:
    """Owns the camera, detectors, and per-frame distraction logic - no
    Tkinter dependency. Runs its own capture loop on a background thread and
    exposes only get_latest_frame() (thread-safe) for the UI to poll from
    the main thread via `after()`. Never touch Tkinter widgets from inside
    this class - only the App class is allowed to do that."""

    def __init__(self, on_calibration_status=None):
        self.stats = SessionStats()
        self._stop_event = threading.Event()
        self._latest_frame_rgb = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self._on_calibration_status = on_calibration_status or (lambda msg: None)

    def start(self):
        self.stats = SessionStats(start_time=time.time())
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.stats.finalize(time.time())

    def get_latest_frame(self):
        with self._frame_lock:
            return self._latest_frame_rgb

    def _set_latest_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._frame_lock:
            self._latest_frame_rgb = rgb

    def _run(self):
        # Opening the camera is what actually triggers the OS permission
        # prompt - deliberately done here, inside the worker thread spawned
        # by the Start Session button, not at import time or app startup.
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera.")
            self._on_calibration_status("Could not open camera - check permissions and try again.")
            return

        face_tracker = FaceTracker(min_detection_confidence=0.3, min_tracking_confidence=0.5)
        pose_estimator = HeadPoseEstimator()
        phone_detector = PhoneDetector()
        gif_player = GifPlayer(GIF_PATH)
        audio_player = AudioLoopPlayer(GIF_PATH)

        self._on_calibration_status("Calibrating - look at the screen normally...")
        pose_estimator.calibrate_neutral(face_tracker, cap)
        self._on_calibration_status(None)

        distracted_streak = 0
        clear_streak = 0
        is_distracted = False

        try:
            while not self._stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    continue

                face_away = None
                face_box = None
                face_result = face_tracker.get_face_coordinates(frame)
                if face_result is not None:
                    points, face_box = face_result
                    if points is not None:
                        face_away = pose_estimator.is_face_away(frame, points)

                phone_result = phone_detector.detect(frame)
                phone_box, phone_score = (phone_result if phone_result is not None else (None, None))

                distracted_now = bool(face_away) or phone_box is not None
                if distracted_now:
                    distracted_streak += 1
                    clear_streak = 0
                else:
                    clear_streak += 1
                    distracted_streak = 0

                now = time.time()
                if phone_box is not None and distracted_streak >= PHONE_ENTER_FRAMES:
                    if not is_distracted:
                        is_distracted = True
                        audio_player.start()
                        self.stats.mark_distracted_start(now)
                        logger.info("Distraction detected (phone=%.2f) - nag started.", phone_score)
                elif not is_distracted and distracted_streak >= FACE_ENTER_FRAMES:
                    is_distracted = True
                    audio_player.start()
                    self.stats.mark_distracted_start(now)
                    logger.info("Distraction detected (face_away=%s) - nag started.", face_away)
                elif is_distracted and clear_streak >= EXIT_DISTRACTION_FRAMES:
                    is_distracted = False
                    gif_player.reset()
                    audio_player.stop()
                    self.stats.mark_distracted_end(now)
                    logger.info("Back on task - nag stopped.")

                display = frame.copy()
                if face_box is not None:
                    cv2.rectangle(display, (face_box["x_min"], face_box["y_min"]),
                                  (face_box["x_max"], face_box["y_max"]), (0, 255, 0), 2)
                if phone_box is not None:
                    x_min, y_min, x_max, y_max = phone_box
                    cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(display, f"phone {phone_score:.2f}", (x_min, max(0, y_min - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                status = f"face_away={face_away} phone={phone_box is not None} distracted={is_distracted}"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if is_distracted:
                    display = self._composite_alert(display, gif_player.next_frame())

                self._set_latest_frame(display)
        finally:
            audio_player.stop()
            cap.release()

    @staticmethod
    def _composite_alert(frame, gif_frame):
        """Darkens the webcam frame and pastes the nag clip centered on top,
        with a red border and banner text - a picture-in-picture alert look,
        composited into a single frame so the UI only ever displays one image."""
        h, w = frame.shape[:2]
        darkened = (frame.astype(np.float32) * 0.35).astype(np.uint8)

        box_w = int(w * 0.55)
        box_h = int(box_w * gif_frame.shape[0] / gif_frame.shape[1])
        if box_h > int(h * 0.55):
            box_h = int(h * 0.55)
            box_w = int(box_h * gif_frame.shape[1] / gif_frame.shape[0])
        box_w, box_h = max(1, box_w), max(1, box_h)

        resized_gif = cv2.resize(gif_frame, (box_w, box_h))

        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2

        border = 6
        cv2.rectangle(
            darkened,
            (x0 - border, y0 - border),
            (x0 + box_w + border, y0 + box_h + border),
            (0, 0, 255), -1,
        )
        darkened[y0:y0 + box_h, x0:x0 + box_w] = resized_gif

        banner = "FOCUS! You're distracted"
        (text_w, text_h), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_x = max(0, (w - text_w) // 2)
        text_y = max(text_h + 10, y0 - border - 15)
        cv2.putText(darkened, banner, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        return darkened


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Distraction Monitor")
        self.geometry("1000x750")
        self.configure(bg="#111111")

        self.engine = None
        self._video_photo = None  # keep a reference - Tkinter drops PhotoImages with no live reference
        self._update_job = None

        self._build_ui()
        self._show_start_screen()

    def _build_ui(self):
        self.top_bar = tk.Frame(self, bg="#1c1c1c", height=60)
        self.top_bar.pack(side="top", fill="x")

        self.timer_label = tk.Label(self.top_bar, text="00:00", font=("Segoe UI", 16, "bold"),
                                     fg="white", bg="#1c1c1c")
        self.timer_label.pack(side="left", padx=20, pady=10)

        self.status_label = tk.Label(self.top_bar, text="", font=("Segoe UI", 11),
                                      fg="#aaaaaa", bg="#1c1c1c")
        self.status_label.pack(side="left", padx=10)

        self.start_button = ttk.Button(self.top_bar, text="Start Session", command=self._on_start_clicked)
        self.stop_button = ttk.Button(self.top_bar, text="Stop Session", command=self._on_stop_clicked)

        self.video_label = tk.Label(self, bg="black", fg="white", font=("Segoe UI", 16))

        self.summary_frame = tk.Frame(self, bg="#111111")
        tk.Label(self.summary_frame, text="Session Summary", font=("Segoe UI", 20, "bold"),
                 fg="white", bg="#111111").pack(pady=(40, 20))
        self.summary_stats_label = tk.Label(self.summary_frame, text="", font=("Consolas", 14),
                                             fg="white", bg="#111111", justify="left")
        self.summary_stats_label.pack()
        ttk.Button(self.summary_frame, text="Start New Session",
                   command=self._show_start_screen).pack(pady=30)

    def _show_start_screen(self):
        self.summary_frame.pack_forget()
        self._video_photo = None
        self.video_label.configure(image="", text="Click 'Start Session' to begin")
        self.video_label.pack(side="top", fill="both", expand=True)
        self.stop_button.pack_forget()
        self.start_button.pack(side="right", padx=20, pady=10)
        self.timer_label.configure(text="00:00")
        self.status_label.configure(text="")

    def _on_start_clicked(self):
        self.start_button.pack_forget()
        self.status_label.configure(text="Requesting camera access...")
        self.engine = DistractionEngine(on_calibration_status=self._on_calibration_status)
        self.engine.start()
        self.stop_button.pack(side="right", padx=20, pady=10)
        self._update_frame_loop()

    def _on_calibration_status(self, message):
        # Called from the worker thread - route through `after()` so the
        # actual widget update always happens on the main thread.
        self.after(0, lambda: self.status_label.configure(text=message or ""))

    def _on_stop_clicked(self):
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="Stopping...")
        threading.Thread(target=self._stop_engine_and_show_summary, daemon=True).start()

    def _stop_engine_and_show_summary(self):
        self.engine.stop()
        self.after(0, self._show_summary)

    def _update_frame_loop(self):
        if self.engine is None:
            return

        frame_rgb = self.engine.get_latest_frame()
        if frame_rgb is not None:
            widget_w = max(self.video_label.winfo_width(), 640)
            widget_h = max(self.video_label.winfo_height(), 480)
            image = Image.fromarray(frame_rgb)
            image = self._fit_image(image, widget_w, widget_h)
            self._video_photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=self._video_photo, text="")

        elapsed = time.time() - self.engine.stats.start_time
        self.timer_label.configure(text=self._format_duration(elapsed))

        self._update_job = self.after(VIDEO_UPDATE_MS, self._update_frame_loop)

    @staticmethod
    def _fit_image(image, target_w, target_h):
        img_w, img_h = image.size
        scale = min(target_w / img_w, target_h / img_h)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        return image.resize(new_size, Image.Resampling.BILINEAR)

    @staticmethod
    def _format_duration(seconds):
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _show_summary(self):
        if self._update_job is not None:
            self.after_cancel(self._update_job)
            self._update_job = None

        stats = self.engine.stats
        self.video_label.pack_forget()
        self.stop_button.pack_forget()
        self.status_label.configure(text="")

        distracted_pct = (stats.total_distracted_seconds / stats.total_seconds * 100) if stats.total_seconds else 0.0
        summary_text = (
            f"Total session time:     {self._format_duration(stats.total_seconds)}\n"
            f"Focused time:            {self._format_duration(stats.focused_seconds)}\n"
            f"Distracted time:         {self._format_duration(stats.total_distracted_seconds)}"
            f"  ({distracted_pct:.0f}%)\n"
            f"Number of distractions:  {stats.distraction_events}\n"
        )
        self.summary_stats_label.configure(text=summary_text)
        self.summary_frame.pack(side="top", fill="both", expand=True)

    def destroy(self):
        if self.engine is not None:
            self.engine.stop()
        super().destroy()


def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.destroy)
    app.mainloop()


if __name__ == "__main__":
    main()
