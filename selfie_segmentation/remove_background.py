import mediapipe as mp
import cv2
import numpy as np
from module.selfie_segmentation import SelfieSegmentation
from openvisionkit.capture.video_template import video_capture_template
selfie_segmentation = SelfieSegmentation()
def remove_background_logic(frame):
  return selfie_segmentation.fast_remove_background(frame)
  # return selfie_segmentation.remove_background(frame)
    
if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=False,      
        custom_logic=remove_background_logic,
        window_name="Remove BackGround",
        resolution=(680, 320),
        center_window=True,
        draw_fps=True
    )
