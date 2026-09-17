import math
import cv2
import numpy as np
import mediapipe as mp
import numpy as np

# Correct imports for Tasks API
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

class FaceMeshDetector:

  # ====================== LANDMARK INDICES (MediaPipe Face Landmarker v2 - 478 points) ======================
  # Iris centers (added by the iris model)
  LEFT_IRIS_CENTER  = 468
  RIGHT_IRIS_CENTER = 473
  LEFT_IRIS = [474, 475, 476, 477]
  RIGHT_IRIS = [469, 470, 471, 472]
  
  FOREHEAD_CENTER    = 10
  

  # Eye corners
  LEFT_EYE_OUTER    = 33
  LEFT_EYE_INNER    = 133
  RIGHT_EYE_OUTER   = 362
  RIGHT_EYE_INNER   = 263

  # Mouth landmarks for openness ratio
  UPPER_LIP_CENTER  = 13
  LOWER_LIP_CENTER  = 14      # Common pair used in many projects
  MOUTH_LEFT        = 61
  MOUTH_RIGHT       = 291
  LIP_LEFT_CORNERS =  61
  LIP_RIGHT_CORNERS = 291
  LIP_CORNERS       = [LIP_LEFT_CORNERS, LIP_RIGHT_CORNERS]
  LIP_CENTER_TOP       = 13
  LIP_CENTER_BOTTOM    = 14
  LIP_CENTER          = [LIP_CENTER_TOP, LIP_CENTER_BOTTOM]

  # Face width landmarks (for normalization)
  LEFT_CHEEK        = 234
  RIGHT_CHEEK       = 454
  # Nose tip (for head pose estimation)
  NOSE_TIP          = 1 
  # ====================== END OF LANDMARK INDICES ======================
  
  LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
  ]
  
  RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
      263, 466, 388, 387, 386, 385, 384, 398
  ]
  
  LEFT_EYE_BLINK = [33, 160, 158, 133, 153, 144]
  
  RIGHT_EYE_BLINK = [362, 385, 387, 263, 373, 380]

  """FaceMeshDetector class uses MediaPipe's FaceLandmarker to detect facial landmarks and draw them on the input image.
  
  468 facial landmarks are detected per face, and the class also extracts blendshape coefficients for facial expressions, head pose transformation matrices, and bounding boxes for each detected face. The class is designed
  **Extended features (beyond original landmarks drawing):**
  - Configurable options in constructor (num_faces, confidence thresholds, running mode, blendshapes, matrices).
  - Returns face blendshapes (52 facial expression coefficients per detected face).
  - Returns facial transformation matrices (4x4 head-pose matrices per detected face).
  - Computes and returns axis-aligned bounding boxes for each detected face (true "face detection" feature).
  - Supports IMAGE mode by default (VIDEO / LIVE_STREAM can be enabled via constructor; detect_for_video would require additional timestamp handling).

  The class can be used for face tracking, expression recognition, AR effects, head-pose estimation, etc.
  
  Args:
    model_path (str): The path to the face landmarker model file. 
      Default is './models/face_landmarker_v2_with_blendshapes.task'. 
  """
  def __init__(self, 
               model_path='./models/face_landmarker_v2_with_blendshapes.task',
               num_faces: int = 2,
               min_face_detection_confidence: float = 0.5,
               min_face_presence_confidence: float = 0.5,
               min_tracking_confidence: float = 0.5,
               output_face_blendshapes: bool = True,
               output_facial_transformation_matrixes: bool = True,
               running_mode: VisionTaskRunningMode = vision.RunningMode.IMAGE): 
      """Initializes the FaceMeshDetector with flexible options.
      
      Args:
        model_path (str): Path to the .task model file.
        num_faces (int): Maximum number of faces to detect.
        min_face_detection_confidence (float): Minimum confidence for face detection.
        min_face_presence_confidence (float): Minimum confidence that a face is present.
        min_tracking_confidence (float): Minimum confidence for tracking (used in VIDEO/LIVE_STREAM).
        output_face_blendshapes (bool): Whether to output 52 blendshape scores for expressions.
        output_facial_transformation_matrixes (bool): Whether to output 4x4 head-pose matrices.
        running_mode (RunningMode): IMAGE (default), VIDEO, or LIVE_STREAM.
      """   
      options = FaceLandmarkerOptions(
          base_options=BaseOptions(model_asset_path=model_path),
          num_faces=num_faces,
          min_face_detection_confidence=min_face_detection_confidence,
          min_face_presence_confidence=min_face_presence_confidence,
          min_tracking_confidence=min_tracking_confidence,
          output_face_blendshapes=output_face_blendshapes,
          output_facial_transformation_matrixes=output_facial_transformation_matrixes,
          running_mode=running_mode
      )
      self.face_detector = FaceLandmarker.create_from_options(options)
      self.drawing_utils = mp.tasks.vision.drawing_utils 
      self.drawing_styles = mp.tasks.vision.drawing_styles
      
  def euclidean_distance(self, p1, p2):
    """
    2D pixel distance between two landmarks [x, y].
    
    Args:
        p1 (list[int]): [x, y] coordinates of the first point.
        p2 (list[int]): [x, y] coordinates of the second point.
        
    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])
  
  def get_landmark_point(self, landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

  def iris_center(self, face_landmarks, iris_indices, width, height):
    points = []

    for idx in iris_indices:
        lm = face_landmarks.landmark[idx]
        x, y = self.get_landmark_point(lm, width, height)
        points.append((x, y))

    points = np.array(points)
    cx, cy = points.mean(axis=0).astype(int)

    return int(cx), int(cy)

  def distance_between_points(self, p1, p2):
    return math.dist(p1, p2)

  def get_mouth_openness_ratio(self, face):
    """Mouth openness ratio (height / width). 0 = closed, ~0.5+ = wide open."""
    if len(face) < 478:
        return 0.0
    upper = face[self.UPPER_LIP_CENTER]
    lower = face[self.LOWER_LIP_CENTER]
    left  = face[self.MOUTH_LEFT]
    right = face[self.MOUTH_RIGHT]

    mouth_height = self.euclidean_distance(upper, lower)
    mouth_width  = self.euclidean_distance(left, right)
    return mouth_height / mouth_width if mouth_width > 0 else 0.0

  def get_eye_gaze_direction(self, face, is_left_eye=True):
    """
    Simple gaze direction (Left / Center / Right) using iris vs eye corners.
       Returns: 'Left', 'Center', or 'Right' (horizontal gaze only).
       
    Args:
    - face: List of 478 [x, y] landmarks for a detected face.
    - is_left_eye: Whether to analyze the left eye (True) or right eye (False).

    Returns:
    - str: 'Left', 'Center', or 'Right' indicating gaze direction.  
    
    Note: This is a very basic heuristic and may not be highly accurate. For more robust gaze estimation, consider using a dedicated gaze tracking model.
    """
    if len(face) < 478:
        return "Unknown"

    if is_left_eye:
        iris_center = face[self.LEFT_IRIS_CENTER]
        eye_outer   = face[self.LEFT_EYE_OUTER]
        eye_inner   = face[self.LEFT_EYE_INNER]
    else:
        iris_center = face[self.RIGHT_IRIS_CENTER]
        eye_outer   = face[self.RIGHT_EYE_OUTER]
        eye_inner   = face[self.RIGHT_EYE_INNER]

    eye_center_x = (eye_outer[0] + eye_inner[0]) / 2
    eye_width    = abs(eye_outer[0] - eye_inner[0])
    if eye_width == 0:
        return "Center"

    delta_x = iris_center[0] - eye_center_x
    ratio   = delta_x / eye_width

    if ratio < -0.18:
        return "Left"
    elif ratio > 0.18:
        return "Right"
    else:
        return "Center"

  def get_inter_pupillary_distance(self, face, normalized=False):
      """
      Approximate eye-to-eye (pupil) distance in pixels.
        If normalized=True → divided by face width (useful for real-world scaling).
        
      Args:
      - face: List of 478 [x, y] landmarks for a detected face.
      - normalized: Whether to return distance normalized by face width.
      
      Returns:  
      - float: Inter-pupillary distance in pixels (or normalized ratio if specified).
      """
      if len(face) < 478:
          return 0.0

      left_iris  = face[self.LEFT_IRIS_CENTER]
      right_iris = face[self.RIGHT_IRIS_CENTER]
      ipd = self.euclidean_distance(left_iris, right_iris)

      if normalized:
          face_width = self.euclidean_distance(face[self.LEFT_CHEEK], face[self.RIGHT_CHEEK])
          return ipd / face_width if face_width > 0 else 0.0
      return ipd

  def overlay_ar_filter(self, frame, face, filter_img, filter_type="glasses"):
      """Basic AR overlay (sunglasses example). 
        Returns the frame with filter drawn on top.
        filter_img must be a PNG with alpha channel (RGBA).

      Args:
          frame (numpy.ndarray): The input image/frame in BGR format.
          face (list): List of 478 [x, y] landmarks for a detected face.
          filter_img (numpy.ndarray): The AR filter image with alpha channel (RGBA).
          filter_type (str): Type of filter to apply (e.g. "glasses"). Currently only "glasses" is implemented.
          
      Returns:
          numpy.ndarray: The output image/frame with the AR filter overlaid.  
      """
      if len(face) < 478 or filter_img is None:
          return frame

      if filter_type == "glasses":
          # Use eye corners to position and scale glasses
          left_outer  = face[self.LEFT_EYE_OUTER]
          right_outer = face[self.RIGHT_EYE_OUTER]

          # Center between eyes
          center_x = int((left_outer[0] + right_outer[0]) / 2)
          center_y = int((left_outer[1] + right_outer[1]) / 2)

          # Scale based on eye-to-eye distance
          eye_dist = self.euclidean_distance(left_outer, right_outer)
          scale_factor = int(eye_dist * 1.8)   # adjust multiplier for your filter image

          # Resize filter
          filter_resized = cv2.resize(filter_img, (scale_factor, scale_factor))

          # Position (top-left of filter)
          x = int(center_x - scale_factor / 2)
          y = int(center_y - scale_factor * 0.45)   # slightly above eyes

          # Alpha blending (assuming filter_img has alpha channel)
          h, w = filter_resized.shape[:2]
          if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
              return frame  # out of bounds

          overlay = filter_resized[:, :, :3]
          mask    = filter_resized[:, :, 3:] / 255.0

          roi = frame[y:y+h, x:x+w]
          blended = (1.0 - mask) * roi + mask * overlay
          frame[y:y+h, x:x+w] = blended.astype(np.uint8)

      return frame
      
  def face_mesh_detection(self, img, drawLandMarks=True):
      """
      Detects facial landmarks in the input image and draws them if specified.
      Now also extracts blendshapes, transformation matrices, and bounding boxes.
      
      Args:
        img (numpy.ndarray): The input image in BGR format.
        drawLandMarks (bool): Whether to draw the detected landmarks on the image. Default is True.

      Returns:
        tuple: (annotated_image, faces, blendshapes, transformation_matrices, bboxes)
          - annotated_image (numpy.ndarray): Image with landmarks drawn (if requested).
          - faces (list): List of detected faces; each face is a list of [x, y] pixel coordinates (478 landmarks).
          - blendshapes (list): List of dicts (one per face) with {blendshape_name: score} for facial expressions.
          - transformation_matrices (list): List of 4x4 transformation matrices (as numpy arrays) for head pose.
          - bboxes (list): List of [min_x, min_y, max_x, max_y] bounding boxes (one per face).
      """
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
      result = self.face_detector.detect(mp_image)
      return self.landmarks_on_image(img, result, drawLandMarks)

  def landmarks_on_image(self, image, detection_result, drawLandMarks=True):
      """
      Draws facial landmarks on the input image and extracts extended detection features.
      
      Args:
        image (numpy.ndarray): The input image on which to draw the landmarks.
        detection_result: The FaceLandmarkerResult from detection.
        drawLandMarks (bool): Whether to draw the detected landmarks on the image. Default is True. 
        
      Returns:
        tuple: (annotated_image, faces, blendshapes, transformation_matrices, bboxes)
          - annotated_image (numpy.ndarray): Annotated image.
          - faces (list[list[list[int]]]): Landmark pixel coordinates per face.
          - blendshapes (list[dict]): Blendshape scores per face (e.g. {'eyeBlinkLeft': 0.92, ...}).
          - transformation_matrices (list[np.ndarray]): 4x4 head-pose matrices per face.
          - bboxes (list[list[int]]): Bounding boxes [min_x, min_y, max_x, max_y] per face.
      """
      annotated_image = image.copy()
      faces = []
      blendshapes = []
      transformation_matrices = []
      bboxes = []

      if not detection_result.face_landmarks:
          return annotated_image, faces, blendshapes, transformation_matrices, bboxes

      h, w, _ = annotated_image.shape
      face_landmarks_list = detection_result.face_landmarks
      for idx, face_landmarks in enumerate(face_landmarks_list):

          # Draw mesh (same as original)
          if drawLandMarks:
              self.drawing_utils.draw_landmarks(
                  image=annotated_image,
                  landmark_list=face_landmarks,
                  connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=self.drawing_styles.get_default_face_mesh_tesselation_style()
              )

              self.drawing_utils.draw_landmarks(
                  image=annotated_image,
                  landmark_list=face_landmarks,
                  connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=self.drawing_styles.get_default_face_mesh_contours_style()
              )

              self.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_face_mesh_iris_connections_style()
              )
              
              self.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_face_mesh_iris_connections_style()
              )

          ## Extract landmark coordinates + compute bounding box
          face = []
          min_x, min_y = w, h
          max_x, max_y = 0, 0
          ## print(face_landmarks)
          ## NormalizedLandmark(x=0.5348694324493408, y=0.34117743372917175, z=-0.0013497794279828668, visibility=None, presence=None, name=None)
          
          for lm in face_landmarks:
              x, y = int(lm.x * w), int(lm.y * h)
              face.append([x, y])
              min_x = min(min_x, x)
              min_y = min(min_y, y)
              max_x = max(max_x, x)
              max_y = max(max_y, y)

              if drawLandMarks:
                  cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

          faces.append(face)
          bboxes.append([min_x, min_y, max_x, max_y])

          ## Extract blendshapes (52 facial expression coefficients) for each detected face. The code checks if the detection result contains face blendshapes and if the index is within bounds. If so, it iterates through the classifications in the blendshapes proto and constructs a dictionary mapping blendshape category names to their corresponding scores. This dictionary is then appended to the blendshapes list, which will contain the blendshape information for each detected face.
          if (detection_result.face_blendshapes and 
              idx < len(detection_result.face_blendshapes)):
              face_blend = {}
              # Standard MediaPipe Tasks API access for blendshapes
              blendshapes_proto = detection_result.face_blendshapes[idx]
              # print(blendshapes_proto)
              ##  Category(index=7, score=3.274757887083979e-07, display_name=None, category_name='cheekSquintLeft')
              for classification in blendshapes_proto:
                  face_blend[classification.category_name] = classification.score
              blendshapes.append(face_blend)
          else:
              blendshapes.append({})

          ## Extract facial transformation matrix (head pose) for each detected face. The code checks if the detection result contains facial transformation matrices and if the index is within bounds. If so, it retrieves the matrix, reshapes it into a 4x4 numpy array, and appends it to the transformation_matrices list. This list will contain the head pose information for each detected face. If the matrix is not available, it appends None to maintain the list structure.
          if (detection_result.facial_transformation_matrixes and 
              idx < len(detection_result.facial_transformation_matrixes)):
              matrix = detection_result.facial_transformation_matrixes[idx]
              transformation_matrices.append(np.array(matrix).reshape(4, 4))
          else:
              transformation_matrices.append(None)

      return annotated_image, faces, blendshapes, transformation_matrices, bboxes
  
  def distance_between_landmarks(self, p1, p2, img=None, draw=True, color=(255, 0, 255), thickness=3, radius=8):
        """
        Find the Euclidean distance between two landmarks and optionally draw on the image.
        
        Args:
            p1 (tuple[int, int]): (x, y) coordinates of the first point
            p2 (tuple[int, int]): (x, y) coordinates of the second point
            img (numpy.ndarray, optional): Image on which to draw. If None, no drawing
            draw (bool): Whether to draw the points and line on the image.
            color (tuple): Color for drawing (BGR). Default is magenta (255, 0, 255).
            thickness (int): Thickness of the line. Default is 3.
            radius (int): Radius of the circles at the points. Default is 8.  
            
        Returns:
            float: The Euclidean distance between the two points.
            tuple: (x1, y1, x2, y2, cx, cy) coordinates of the two points and their center.
            numpy.ndarray (optional): Annotated image if img is provided and draw=True.
        """
        x1, y1 = p1
        x2, y2 = p2
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        length = self.euclidean_distance((x2, y2), (x1, y1))  ## Euclidean distance
        
        if img is not None and draw:
            ## Draw circles at both points
            cv2.circle(img, (x1, y1), radius, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, color, cv2.FILLED)
            
            ## Draw connecting line
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
            ## Draw center point
            cv2.circle(img, (cx, cy), radius//2, (0, 255, 0), cv2.FILLED)
            
            ## Optional: Show distance value near the line
            cv2.putText(img, f"{length:.1f}px", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return length, (x1, y1, x2, y2, cx, cy), img
        
        return length, (x1, y1, x2, y2, cx, cy)

  def get_head_pose_angles(self, matrix):
    """Extract yaw, pitch, roll (in degrees) from the 4x4 facial transformation matrix.
    
    Args:
        matrix (np.ndarray): 4x4 facial transformation matrix.
    
    Returns:
        tuple: Yaw, pitch, and roll angles in degrees.
    """
    if matrix is None:
        return 0.0, 0.0, 0.0
    
    ## Rotation part of the matrix
    R = matrix[:3, :3]
    
    ## Yaw (left/right head turn)
    yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
    
    ## Pitch (up/down head tilt)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * 180 / np.pi
    
    ## Roll (head tilt left/right)
    roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
    
    return yaw, pitch, roll


  def get_emotion(self, blend):
      """
      Simple rule-based emotion detection using blendshapes (very fast).
      
      Args:
          blend (dict): Dictionary of blendshape coefficients.
      
      Returns:
          str: Detected emotion as a string.
      """
      if not blend:
          return "Unknown"
      
      smile = blend.get('mouthSmileLeft', 0) + blend.get('mouthSmileRight', 0)
      frown = blend.get('mouthFrownLeft', 0) + blend.get('mouthFrownRight', 0)
      brow_up = (blend.get('browInnerUp', 0) +
                blend.get('browOuterUpLeft', 0) +
                blend.get('browOuterUpRight', 0))
      eye_blink = blend.get('eyeBlinkLeft', 0) + blend.get('eyeBlinkRight', 0)
      mouth_open = (blend.get('mouthLowerDownLeft', 0) +
                    blend.get('mouthLowerDownRight', 0) +
                    blend.get('jawOpen', 0))
      
      if smile > 0.45 and eye_blink < 0.4:
          return "😊 Happy"
      elif frown > 0.4:
          return "😠 Angry"
      elif eye_blink > 0.75:
          return "😲 Surprised"
      elif mouth_open > 0.45:
          return "😮 Shocked"
      elif brow_up > 0.5:
          return "🤨 Confused"
      else:
          return "😐 Neutral"