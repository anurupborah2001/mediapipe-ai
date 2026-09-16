import cv2
import time
import numpy as np

from module.object_detector import ObjectDetector
from openvisionkit.capture.image_template import image_template
from openvisionkit.capture.video_template import video_capture_template
from util import utility

path, _= utility.get_calling_folder()


detect_type="VIDEO"
object_detector = ObjectDetector(running_mode=detect_type)
start_time = time.time()

def object_detection(image):
    timestamp_ms = int((time.time() - start_time) * 1000)
    annotated_image = object_detector.detect_objects(image,timestamp_ms=timestamp_ms)
    # Placeholder for object detection logic
    # This function should return a list of detected objects with their bounding boxes and labels
    detected_objects = []
    return annotated_image

if __name__ == "__main__":
  if detect_type == "IMAGE":
    image_template(
      image_path=f"{path}/assets/image/coco_test_image.jpg",
      custom_logic=object_detection,
      window_name="COCO Object Detection"  
    )
  elif detect_type == "VIDEO":
    video_capture_template(
      video_source=0,
      custom_logic=object_detection,
      window_name="COCO Object Detection",
    )
    