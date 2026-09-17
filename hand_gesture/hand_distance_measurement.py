
from collections import deque

import cv2

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
from util import utility
import numpy as np

calibration_samples = [
        (280, 20),
        (120, 50),
        (145, 40),
        (115, 50),
        (95, 60),
        (80, 70),
        (68, 80),
        (300, 20),  # Sample at 20 cm where 300 is the pixel width of the palm at 20 cm distance
        (245, 25),  # Sample at 25 cm
        (200, 30),  # Sample at 30 cm
        (170, 35),  # Sample at 35 cm 
        (145, 40),  # Sample at 40 cm
        (130, 45),  # Sample at 45 cm
        (112, 50),  # Sample at 50 cm
        (103, 55),  # Sample at 55 cm
        (93, 60),   # Sample at 60 cm
        (87, 65),   # Sample at 65 cm
        (80, 70),   # Sample at 70 cm
        (75, 75),   # Sample at 75 cm
        (70, 80),   # Sample at 80 cm
        (67, 85),   # Sample at 85 cm
        (62, 90),   # Sample at 90 cm
        (59, 95),   # Sample at 95 cm
        (57, 100)   # Sample at 100 cm
    ]
hand_detector  = HandDetector(max_hands=1, detection_confidence=0.8, calibration_samples=calibration_samples)
distance_smoothing = deque(maxlen=8)

def hand_distance_measurement(imgFrame):
  annotated_image, hand_landmarks = hand_detector.draw_landmarks(imgFrame, to_draw_bounding_box=False, to_draw_center_point=False, to_draw_landmark=False)
  if len(hand_landmarks) != 0:
    landmarks, bouding_box_list, landmark_params, _ = hand_landmarks[0]
    landmark_list = landmarks if len(landmarks) > 0 else []
    # bouding_box_list = landmark_params["bounding_box"] if "bounding_box" in landmark_params else []
    h, w, _ = imgFrame.shape
    x, y, w, h = bouding_box_list
    ## Distance between INDEX_MCP and PINKY_MCP is more stable than distance between INDEX_TIP and PINKY_TIP, as the tips can be 
    # more easily bent and thus less consistent for distance measurement.
    palm_width_px,  _, _ = hand_detector.palm_width_px(annotated_image, landmark_list, draw_landmarks=True)
    
    distance_cm = hand_detector.estimate_distance_cm( palm_width_px)
    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    utility.put_text_rect(annotated_image, f'{int(distance_cm)} cm', (x+5, y-10), colorR=(255, 255, 0), scale=2, colorT=(255, 255, 255))
  
  return annotated_image

if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=True,
        custom_logic=hand_distance_measurement,
        window_name="Hand Distance Measurement",
        enable_screenshot=True
    )