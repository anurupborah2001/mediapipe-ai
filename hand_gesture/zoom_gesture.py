import cv2
from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
from util import utility

path, _= utility.get_calling_folder()

hand_detector = HandDetector(detection_confidence=0.8, max_hands=2)
finger_distance = None
scale = 0
center_x, center_y = 640, 360 

# screen = ScreenCapture()
# recorder = VideoRecorder(output_format="mp4")

def zoom_gesture_logic(frame):
    global finger_distance, center_x, center_y, scale
    img = cv2.imread(f"{path}/images/zoom_gesture/mickey_mouse.jpg")
    # Placeholder for zoom gesture logic
    annotated_image, hand_landmark = hand_detector.draw_landmarks(frame)
    thumb_tip = hand_detector.fingerTips[0]
    index_finger_tip = hand_detector.fingerTips[1]
    if len(hand_landmark) == 2:
      right_hand_landmark = hand_landmark[0]
      left_hand_landmark = hand_landmark[1]
      
      ## Landmarks list for right and left hand
      right_hand_landmark_list, _, _, _ = right_hand_landmark
      left_hand_landmark_list, _, _, _ = left_hand_landmark
      
      is_finger_joined_right  = hand_detector.is_fingers_joined_2(thumb_tip, index_finger_tip, annotated_image, right_hand_landmark_list)
      is_finger_joined_left = hand_detector.is_fingers_joined_2(thumb_tip, index_finger_tip, annotated_image, left_hand_landmark_list)
      
      right_thumb_x, right_thumb_y, _ = right_hand_landmark_list[thumb_tip][1:]  # Right hand thumb tip coordinates
      left_thumb_x, left_thumb_y, _ = left_hand_landmark_list[thumb_tip][1:]  # Left hand thumb tip coordinates
      if is_finger_joined_right and is_finger_joined_left:
          if finger_distance is None:
            length, annotated_image, co_ords = hand_detector.get_distance((right_thumb_x, right_thumb_y), (left_thumb_x, left_thumb_y), annotated_image)
            finger_distance = length
          
          length, annotated_image, co_ords = hand_detector.get_distance((right_thumb_x, right_thumb_y), (left_thumb_x, left_thumb_y), annotated_image)
          center_x, center_y = co_ords[4:]
          scale = int((length - finger_distance) // 2)
    else:
      finger_distance = None
      
    try:
      height, width, _ = img.shape

      new_width = max(50, width + scale)
      new_height = max(50, height + scale)

      img_resized = cv2.resize(img, (new_width, new_height))

      h_frame, w_frame, _ = annotated_image.shape

      # -------- Compute safe ROI --------
      x1 = max(0, center_x - new_width // 2)
      y1 = max(0, center_y - new_height // 2)
      x2 = min(w_frame, center_x + new_width // 2)
      y2 = min(h_frame, center_y + new_height // 2)

      # -------- Crop image if needed --------
      img_crop = img_resized[
          0:(y2 - y1),
          0:(x2 - x1)
      ]

      # -------- Overlay --------
      annotated_image[y1:y2, x1:x2] = img_crop

    except Exception as e:
        print("Overlay error:", e)
      
    return annotated_image
  
if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        custom_logic=zoom_gesture_logic,
        window_name="Zoom Gesture",
        resolution=(1280, 720),
        center_window=True,
        draw_fps=True,
        enable_manual_recording=True,
        record_format="gif",
    )