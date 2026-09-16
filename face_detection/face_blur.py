
import time
from module.face_detector import FaceDetector
import cv2

face_detector = FaceDetector(max_faces=2, running_mode="VIDEO", min_detection_confidence=0.5)
start_time = time.time()

def face_blur_logic(frame):
  timestamp_ms = int((time.time() - start_time) * 1000)
  image, detection_params = face_detector.detect_faces(frame, timestamp_ms=timestamp_ms, to_draw_bounding_box=False, to_draw_landmarks=False)
  if detection_params:
    for detection in detection_params:
      bbox =  detection["bbox"]
      x, y, w, h = bbox
      imgCrop = image[y:y+h, x:x+w]
      imgCrop = cv2.GaussianBlur(imgCrop, (99, 99), 30)
      # imgCrop = cv2.blur(imgCrop, (40, 40))
      image[y:y+h, x:x+w] = imgCrop
  return image

if __name__ == "__main__":
    from openvisionkit.capture.video_template import video_capture_template

    video_capture_template(
        video_source=0,
        custom_logic=face_blur_logic,
        window_name="Face Blur",
        loop_forever=True,
        resolution=(680, 320),
        center_window=False,
        draw_fps=True
    )