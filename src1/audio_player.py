import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"


class AudioLoopPlayer:
    """cv2 has no audio support at all - VideoCapture only ever decodes
    frames. This extracts the source clip's audio track once (cached next to
    the source file as a .wav) and loops it via the stdlib `winsound` module -
    no extra dependency, no compiled package, just ships with Python on
    Windows. Call start()/stop() whenever the video loop starts/stops -
    they're idempotent, safe to call every frame.

    winsound is Windows-only. On other platforms this degrades to silent
    (logged once) rather than crashing the monitor."""

    def __init__(self, video_path):
        self.audio_path = self._extract_audio(video_path)
        self._playing = False

        if not IS_WINDOWS:
            logger.warning("winsound is Windows-only - nag audio will be silent on this platform.")

    def _extract_audio(self, video_path):
        cache_path = os.path.splitext(video_path)[0] + "_audio.wav"
        if os.path.exists(cache_path):
            return cache_path

        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            logger.warning("imageio-ffmpeg not installed - nag will be silent. Run: pip install imageio-ffmpeg")
            return None

        cmd = [
            ffmpeg_exe, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            cache_path,
        ]
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0 or not os.path.exists(cache_path):
            logger.warning(
                "Could not extract audio from %s (no audio track, or ffmpeg failed) - nag will be silent.",
                video_path,
            )
            return None

        logger.info("Extracted audio to %s", cache_path)
        return cache_path

    def start(self):
        if self.audio_path is None or self._playing or not IS_WINDOWS:
            return

        import winsound
        winsound.PlaySound(self.audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        self._playing = True

    def stop(self):
        if not self._playing or not IS_WINDOWS:
            return

        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
        self._playing = False


def main():
    """Smoke test - loops the clip's audio until Enter is pressed."""
    if len(sys.argv) < 2:
        print("Usage: python audio_player.py <path_to_video>")
        return

    player = AudioLoopPlayer(sys.argv[1])
    player.start()
    input("Playing on loop - press Enter to stop.\n")
    player.stop()


if __name__ == "__main__":
    main()
