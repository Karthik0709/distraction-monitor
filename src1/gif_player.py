import time

import cv2

DEFAULT_FPS = 30.0  # fallback if the video file doesn't report a valid FPS


class GifPlayer:
    """Loads a short video clip's frames up front, then hands back whichever
    frame should be showing right now based on elapsed wall-clock time - so
    playback speed matches the clip's real FPS regardless of your main loop's
    frame rate. Despite the name, this reads any format cv2.VideoCapture can
    open (mp4, mov, avi, gif, ...) - "gif" here just means "the nag clip"."""

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
        """Call this when the clip is not being shown, so it restarts from
        frame 0 next time instead of resuming mid-cycle."""
        self._start_time = None

    def next_frame(self):
        """Returns the BGR frame (numpy array) that should be displayed right now."""
        if self._start_time is None:
            self._start_time = time.time()

        elapsed = (time.time() - self._start_time) % self.cycle_length_sec
        idx = int(elapsed / self.frame_duration)
        idx = min(idx, len(self.frames) - 1)
        return self.frames[idx]


def main():
    """Smoke test - loops the clip in a window until ESC is pressed."""
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
