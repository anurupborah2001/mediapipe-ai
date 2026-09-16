import cv2
import time
from module.face_detector import FaceDetector
from openvisionkit.capture.image_template import image_template
from openvisionkit.capture.video_template import video_capture_template
from mediapipe.tasks.python import vision

from util import utility
from util.utility import put_text_think_corners
path, _= utility.get_calling_folder()

SHOW_IMG: bool = "IMAGE"
face_detector = FaceDetector(running_mode=SHOW_IMG, min_detection_confidence=0.5)
start_time = time.time()

def detect_faces(frame):
  ##Needed for video mode to get the timestamp in milliseconds for accurate detection results. For image mode, it can be set to None or ignored.
  timestamp_ms = int((time.time() - start_time) * 1000)
  image, detection_params = face_detector.detect_faces(frame, timestamp_ms=timestamp_ms, to_draw_bounding_box=True, to_draw_landmarks=True)
  for bbox_info in detection_params:
    if not bbox_info["bounding_boxes"]:
      continue
    x, y, w, h = bbox_info["coordinates"]
    image = put_text_think_corners(image, (x, y, w, h))
  return image

if __name__ == "__main__":
    if SHOW_IMG == "IMAGE":
      image_template(
        image_path="./face_detection/assets/images/faces.jpg", 
        custom_logic=detect_faces,
        resolution=(680, 320),
        window_name="Detected Faces in Images",
        center_window=True
      )
    else:
      video_capture_template(
        video_source="./face_detection/assets/videos/faces.mp4",
        loop_forever=False,      
        custom_logic=detect_faces,
        window_name="Face Detection in Videos",
        resolution=(680, 320),
        center_window=True,
        draw_fps=True
      )
