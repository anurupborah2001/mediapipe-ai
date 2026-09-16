from module.hair_segmentation import HairSegmentation
from openvisionkit.capture.video_template import video_capture_template
import cv2
import numpy as np

hair_segmentation = HairSegmentation()
def hair_segmentation_logic(frame):
  frame = cv2.flip(frame, 1)
  return hair_segmentation.detect(frame)

if __name__ == "__main__":
  video_capture_template(
    video_source=0,
    loop_forever=True,
    custom_logic=hair_segmentation_logic,
    window_name="Hair Segmentation",
    resolution=(650, 500),
    center_window=True,
    draw_fps=True
  )