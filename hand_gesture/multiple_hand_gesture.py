

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
from util import utility


path, _, _ = utility.get_currect_path()
hand_detector = HandDetector(max_hands=2, detection_confidence=0.6)

def multiple_hand_gesture_logic(frame):
    annotated_image, landmarks_list_arr = hand_detector.draw_landmarks(frame)
    if len(landmarks_list_arr) != 0:  
      hand_landmarks, _, landmark_params, _ = landmarks_list_arr[0]
      hand1_center = landmark_params["center_point"]
      finger_hand_1 = hand_detector.fingers_up(hand_landmarks)
      print(f"Fingers up for hand 1: {finger_hand_1}")

      if len(landmarks_list_arr) == 2:
        hand2_landmarks, _, hand2_params, _ = landmarks_list_arr[1]
        print("Two hands detected")
        hand2_center = hand2_params["center_point"]
        finger_hand_2 = hand_detector.fingers_up(hand2_landmarks)
        print(f"Fingers up for hand 2: {finger_hand_2}")
        
        ## Draw line between the center points of the two hands. The code uses OpenCV's line drawing function to draw a line between the center points of the two detected hands on the annotated image. This visual representation can help users see the relationship between their hands and can be used for various gesture-based interactions.
        _ , annotated_image, _ = hand_detector.get_distance(hand1_center, hand2_center, annotated_image, to_draw_line=True, to_draw_circle_key_point=True)
      
    return annotated_image

if __name__ == "__main__":
    
    video_capture_template(
        video_source=0,
        custom_logic=multiple_hand_gesture_logic,
        window_name="Hand Gesture - Multiple Hand Detection",
        loop_forever=False,
        resolution=(680, 320),
        center_window=True,
        draw_fps=True
    )