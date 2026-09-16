import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ObjectDetector:
    def __init__(
        self,
        model_path: str = './models/efficientdet_lite.tflite', 
        max_results=5, 
        running_mode="IMAGE", 
        display_names_locale=b"en",
        category_allowlist=None,
        category_denylist=None):
        self.running_mode = getattr(vision.RunningMode, running_mode)
        base_options = python.BaseOptions(
            model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,         
            score_threshold=0.5,
            max_results=max_results,
            running_mode=self.running_mode,
            display_names_locale=display_names_locale,
            category_allowlist=category_allowlist,
            category_denylist=category_denylist)
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.MARGIN = 15 
        self.FONT_THICKNESS = 2
        self.ROW_SIZE = 10
        self.TEXT_COLOR = (0, 255, 0) 
        self.FONT_SIZE = 1


    def _to_mp_image(self, image):
      """
      Convert a BGR image (as used by OpenCV) to an mp.Image format suitable for MediaPipe processing.
      Args:
        image: The input image in BGR format (as used by OpenCV).
      Returns:
        An mp.Image object in RGB format suitable for MediaPipe processing.
      """
      rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
    def detect(self, image, timestamp_ms=None):
      """
      Detect objects in the given image using the MediaPipe Object Detector.
      Args:
        image: The input image in BGR format (as used by OpenCV).
        timestamp_ms: Optional timestamp in milliseconds for video processing (ignored in IMAGE mode).
      Returns:
        A list of detected objects with their bounding boxes and labels.
      """
      rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mp_image = self._to_mp_image(rgb)
      if self.running_mode == vision.RunningMode.IMAGE:
        result = self.detector.detect(mp_image)
      else:
        result = self.detector.detect_for_video(mp_image, timestamp_ms or 0)
      return result, mp_image
    
    def visualize_detections(self, image, detection_result):
      """
      Visualize detected objects on the image using MediaPipe's visualization utilities.
      Args:
        image: The input image in BGR format (as used by OpenCV).
        detection_result: The result from the detect method containing detected objects.
      Returns:
        An annotated image with detected objects visualized.
      """
      for detection in detection_result.detections:
          bbox = detection.bounding_box
          category = detection.categories[0] if detection.categories else None
          label = category.category_name if category else "Unknown"
          score = category.score if category else 0.0

          start_point = (int(bbox.origin_x), int(bbox.origin_y))
          end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
          
          cv2.rectangle(image, start_point, end_point, self.TEXT_COLOR, self.FONT_THICKNESS)
          text = f"{label}: {score:.2f}"
          text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.FONT_THICKNESS)
          text_origin = (start_point[0], start_point[1] + self.MARGIN)
          cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)
      return image
    
    def detect_objects(self, image, timestamp_ms=None):
        # Placeholder for object detection logic
        # This function should return a list of detected objects with their bounding boxes and labels
        detected_objects = []
        detection_result, mp_image = self.detect(image, timestamp_ms=timestamp_ms)
        
        image_copy = np.copy(mp_image.numpy_view())
        annotated_image = self.visualize_detections(image_copy, detection_result)
        return annotated_image