import numpy as np
import cv2
import mss


class ScreenCapture:
    def __init__(self, monitor_index=1):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]

    def grab(self):
        frame = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)