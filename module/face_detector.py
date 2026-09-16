import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

class FaceDetector:
  """
  FaceDetector class that utilizes MediaPipe's Face Detection solution to detect faces in images or video frames. It provides options to draw bounding boxes and landmarks on the detected faces.
  """
  def __init__(
      self, 
      model_path: str="./models/face_detector.tflite",
      max_faces=5,
      running_mode="IMAGE",  # IMAGE | VIDEO | LIVE_STREAM
      min_detection_confidence: float=0.5,
      min_suppression_threshold: float=0.3
    ):
      self.running_mode = getattr(vision.RunningMode, running_mode)
      self.base_options = python.BaseOptions(
        model_asset_path=model_path
     )
      self.max_faces = max_faces
      self.min_detection_confidence = min_detection_confidence
      self.min_suppression_threshold = min_suppression_threshold  
      self.options = vision.FaceDetectorOptions(
          base_options=self.base_options, 
          running_mode=self.running_mode,  # IMAGE | VIDEO | LIVE_STREAM
          min_detection_confidence=self.min_detection_confidence, 
          min_suppression_threshold=self.min_suppression_threshold
      )
      self.detector = vision.FaceDetector.create_from_options(self.options)
      self.mp_drawing_utils = mp.tasks.vision.drawing_utils
      self.mp_drawing_styles = mp.tasks.vision.drawing_styles
      
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
    Detect faces in the input image using MediaPipe's Face Detector.
    Args: 
      image: The input image in which to detect faces (BGR format).
      timestamp_ms: An optional timestamp in milliseconds for video processing (required for VIDEO and LIVE_STREAM modes).
    Returns:
      The raw detection result from MediaPipe's Face Detector, which includes information about detected faces such
    """
    mp_image = self._to_mp_image(image)
    if self.running_mode == vision.RunningMode.IMAGE:
        result = self.detector.detect(mp_image)
    else:
        result = self.detector.detect_for_video(mp_image, timestamp_ms or 0)

    return result

  def _parse_detections(self, result, shape):
    """
    Parse the raw detection results from MediaPipe and extract relevant information such as bounding boxes, keypoints, and confidence scores.
    Args:
      result: The raw detection result from MediaPipe's Face Detector.
      shape: The shape of the input image (height, width).
    Returns:
      A list of parsed detections, where each detection is a dictionary containing information about the detected face, including its bounding box, confidence score, keypoints, and other relevant attributes.
    """
    
    H, W = shape[:2]
    parsed = []
    bounding_boxes = []
    key_points = []
    categories = []
    if not result.detections:
      return parsed

    for i, detection in enumerate(result.detections):
        """
          Detection(bounding_box=BoundingBox(origin_x=180, origin_y=145, width=701, height=701), categories=[Category(index=0, score=0.9549353718757629, display_name=None, category_name=None)], keypoints=[NormalizedKeypoint(x=0.18383397161960602, y=0.2978437542915344, label=None, score=0.0), NormalizedKeypoint(x=0.33176130056381226, y=0.2957031726837158, label=None, score=0.0), NormalizedKeypoint(x=0.25055351853370667, y=0.4610801339149475, label=None, score=0.0), NormalizedKeypoint(x=0.2593384385108948, y=0.543393611907959, label=None, score=0.0), NormalizedKeypoint(x=0.12543150782585144, y=0.3002464771270752, label=None, score=0.0), NormalizedKeypoint(x=0.4346682131290436, y=0.2893249988555908, label=None, score=0.0)])
        """
        score = detection.categories[0].score if detection.categories else 0
        bbox = detection.bounding_box
        bounding_boxes.append(bbox)
        key_points.append(detection.keypoints)
        categories.append(detection.categories)
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        x2 = x + w
        y2 = y + h

        bounding_box_coordinates = (x, y, w, h)
        parsed.append({
            "id": i,
            "score": score,
            "bbox": (x, y, w, h),
            "bbox_xyxy": (x, y, x2, y2),
            "center": (x + w // 2, y + h // 2),
            "coordinates": bounding_box_coordinates,
            "area": w * h,
            "normalized_keypoints": self._normalize_keypoints(detection.keypoints, W, H),
            "bounding_boxes": bounding_boxes,
            "key_points": key_points,
            "categories": categories
        })
    return parsed

  def detect_faces(self, image, timestamp_ms=None, to_draw_bounding_box=True, to_draw_landmarks=True):
    """
    Detect faces in the input image and optionally draw bounding boxes and landmarks on the detected faces.
    Args:
      image: The input image in which to detect faces (BGR format).
      timestamp_ms: An optional timestamp in milliseconds for video processing (required for VIDEO and LIVE_STREAM modes).
      to_draw_bounding_box: Whether to draw bounding boxes around detected faces.
      to_draw_landmarks: Whether to draw facial landmarks on the detected faces.  
    Returns:
      The image with detected faces (and optionally drawn bounding boxes and landmarks).
    """
    # Implement face detection logic here
    detection_result = self.detect(image, timestamp_ms)
  
    detections = self._parse_detections(detection_result, image.shape)
    
    if self.options.min_detection_confidence is not None:
      detections = self.filter_by_confidence(detections, self.options.min_detection_confidence)

    if self.max_faces is not None:
      detections = self.sort_faces(detections)[:self.max_faces]

    if to_draw_bounding_box:
        image = self.draw_detections(image, detections, to_draw_landmarks)
         
    return image, detections

  def draw_detections(self, image, detections, draw_landmarks=True):
    """
    Draw bounding boxes and landmarks for detected faces on the input image.
    Args:
      image: The input image on which to draw detections (BGR format). 
      detections: A list of detected faces with their bounding box and landmark information.
      draw_landmarks: Whether to draw facial landmarks on the detected faces.
    Returns:
      The image with drawn bounding boxes and landmarks for detected faces.
    """
    for det in detections:
        x, y, x2, y2 = det["bbox_xyxy"]
        fontface = 2 if self.running_mode == vision.RunningMode.IMAGE else 0.8
        cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 255), 2)

        cv2.putText(
            image,
            f'{int(det["score"] * 100)}%',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontface,
            (0, 255, 255),
            2
        )

        if draw_landmarks:
            for (kx, ky) in det["normalized_keypoints"]:
                cv2.circle(image, (kx, ky), 2, (0, 255, 0), -1)

    return image

  def filter_by_confidence(self, detections, threshold=0.5):
    """
    Filter detected faces based on a confidence threshold.
    Args:
      detections: A list of detected faces with their confidence scores.
      threshold: The confidence threshold for filtering detections.
    Returns:
      A list of detections that have confidence scores above the specified threshold. 
    """  
    return [d for d in detections if d["score"] >= threshold]

  def get_largest_face(self, detections):
    """
    Get the largest detected face based on the area.
    Args:
      detections: A list of detected faces with their bounding box information.
    Returns:
      The detection with the largest area, or None if no detections are available.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d["area"])

  def crop_faces(self, image, detections, margin=0):
    """
    Crop detected faces from the input image based on their bounding boxes.
    Args:
      image: The input image from which to crop faces (BGR format).
      detections: A list of detected faces with their bounding box information.
      margin: An optional margin to add around the bounding box when cropping (default is 0).
    Returns:
      A list of cropped face images.
    """
    faces = []
    H, W = image.shape[:2]
    for det in detections:
      x, y, w, h = det["bbox"]
      x1 = max(0, x - margin)
      y1 = max(0, y - margin)
      x2 = min(W, x + w + margin)
      y2 = min(H, y + h + margin)
      faces.append(image[y1:y2, x1:x2])
    return faces

  def sort_faces(self, detections, by="area", descending=True):
    """
    Sort detected faces based on a specified attribute.
    Args:
      detections: A list of detected faces with their attributes.
      by: The attribute to sort by (default is "area").
      descending: Whether to sort in descending order (default is True).
    Returns:
      A list of sorted detections.
    """
    return sorted(detections, key=lambda x: x[by], reverse=descending)
  
  def get_iou(self, boxA, boxB):
    """
    useful for tracking / NMS
    A box is defined by its top-left corner (x1, y1) and bottom-right corner (x2, y2).
    
    Args:
        boxA: A tuple (x1, y1, x2, y2) representing the first bounding box.
        boxB: A tuple (x1, y1, x2, y2) representing the second bounding box.  
        
    Returns:
        The Intersection over Union (IoU) value between the two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)

  def _normalize_keypoints(self, keypoints, W, H):
    """
    Normalize keypoints to the image dimensions.
    Args:
      keypoints: A list of keypoints with x and y coordinates normalized between 0 and 1.
      W: The width of the image.
      H: The height of the image.
    Returns:
      A list of keypoints with coordinates scaled to the image dimensions.
    """
    if not keypoints:
        return []
    return [(int(k.x * W), int(k.y * H)) for k in keypoints]
      
      