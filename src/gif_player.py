import time

import cv2

DEFAULT_FPS = 30.0


class GifPlayer:
    # loads all frames up front, then picks the right one by elapsed wall-clock time - keeps playback speed correct regardless of the caller's loop rate

    def __init__(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = DEFAULT_FPS

        self.frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            self.frames.append(frame)
        cap.release()

        if not self.frames:
            raise ValueError(f"No frames could be read from video: {video_path}")

        self.frame_duration = 1.0 / fps
        self.cycle_length_sec = self.frame_duration * len(self.frames)
        self._start_time = None

    def reset(self):
        self._start_time = None

    def next_frame(self):
        if self._start_time is None:
            self._start_time = time.time()

        elapsed = (time.time() - self._start_time) % self.cycle_length_sec
        idx = int(elapsed / self.frame_duration)
        idx = min(idx, len(self.frames) - 1)
        return self.frames[idx]


def main():
    # smoke test - loops the clip in a window until ESC is pressed
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gif_player.py <path_to_video_or_gif>")
        return

    player = GifPlayer(sys.argv[1])
    while True:
        cv2.imshow("Gif Player Test", player.next_frame())
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
