import logging
import time
import cv2
from typing import Final
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONITNOUS_READING: Final = True

class WebCamStream:

    def __init__(self,camera_index:int = 0, window_name: str = "Captured Frame"):
        self.running = CONITNOUS_READING
        self.camera_index = camera_index
        self.window_name = window_name
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError("Camera not open")
        

    def get_frame(self):
        ret,frame = self.capture.read()
        if ret:
            return frame
        else:
            logger.warning("Frame grab failed - skipping.")
            return None

    def display_cam(self,frame):
        cv2.imshow("Captured Frame", frame)
            
    def should_exit(self) -> bool:
        key_pressed = cv2.waitKey(1) & 0xFF == ord(' ')
        window_closed = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        return key_pressed or window_closed
    
    def _stop_recording(self):
        if self.should_exit():
            self.running = False
            cv2.destroyAllWindows()
            self.capture.release()
            logger.info(f"Camera {self.camera_index} released.")

    
def main():
    webcam = WebCamStream(camera_index=0)
 
    while webcam.running:
        frame = webcam.get_frame()
        if frame is None:
            continue
 
        # <-- next step: hand_tracker/head_pose processing goes here,
        #     between get_frame() and display(), operating on `frame`
        #     before it's drawn.
 
        webcam.display_cam(frame)
 
        if webcam.should_exit():
            webcam.stop()
 
if __name__ == "__main__":
    main()