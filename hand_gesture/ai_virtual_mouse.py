from openvisionkit.capture.video_template import video_capture_template
from module.hand_detector import HandDetector
import cv2
import time
import pyautogui
import numpy as np


hand_detector = HandDetector(max_hands=1, detection_confidence=0.7)
screen_width, screen_height = pyautogui.size()
box_screen_control_screen = 100

## Scrolling parameters
THRESHOLD_CLICK = 0.25
SCROLL_SMOOTHING = 5
SCROLL_SENSITIVITY = 30
SCROLL_DEADZONE = 5
DOUBLE_CLICK_THRESHOLD = 0.4 
scroll_buffer = []
previous_loc_x, previous_loc_y = 0, 0
prev_scroll_y = None
last_click_time = 0
SMOOTHING_THRESHOLD = 0.3
def ai_virtual_mouse_logic(frame):
  global previous_loc_x, previous_loc_y, prev_scroll_y, last_click_time, scroll_buffer
  annotated_image, landmarks_list_arr = hand_detector.draw_landmarks(frame, to_draw_bounding_box=False)
  
  if len(landmarks_list_arr) != 0:
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    hand_landmarks, _, _, _ = landmarks_list_arr[0]
    fingersUp = hand_detector.fingers_up(hand_landmarks)
    print(f"Box screen control screen: {box_screen_control_screen}")
    cv2.rectangle(annotated_image, (box_screen_control_screen, box_screen_control_screen), (frame_width -  box_screen_control_screen, frame_height -  box_screen_control_screen), (0, 255, 0), 2)
      ##Check if the index finger is up and the middle finger is down. The code checks the state of the fingers using the handDetector's fingers_up() method, which returns a list indicating whether each finger is up (1) or down (0). The condition checks if the index finger (fingersUp[1]) is up and the middle finger (fingersUp[2]) is down, which indicates that the user intends to move the mouse cursor.
    index_finger_up = fingersUp[1]
    middle_finger_up = fingersUp[2]
    index_finger_mediapipe_no = hand_detector.fingerTips[1]
    middle_finger_mediapipe_no = hand_detector.fingerTips[2]
    
    middle_x , middle_y, _  = hand_landmarks[middle_finger_mediapipe_no][1:]  # Middle finger tip coordinates (x2, y2, z2)
    # Index finger tip
    index_finger_x, index_finger_y, _ = hand_landmarks[index_finger_mediapipe_no][1:] 
    if index_finger_up and not middle_finger_up:
      print("Moving Mode")
  
      ## Map the index finger tip coordinates to the screen coordinates. The code uses numpy's interp function to map the x-coordinate of the index finger tip from the frame's coordinate system to the screen's coordinate system. The mapping is done based on the defined box_screen_control_screen, which creates a control area within the frame for mouse movement. The resulting x_mouse_pointer value represents the corresponding x-coordinate on the screen where the mouse cursor should be moved.
      x_mouse_pointer = np.interp(index_finger_x, (box_screen_control_screen, frame_width - box_screen_control_screen), (0, screen_width))
      y_mouse_pointer = np.interp(index_finger_y, (box_screen_control_screen, frame_height - box_screen_control_screen), (0, screen_height))

      current_loc_x = previous_loc_x + (x_mouse_pointer - previous_loc_x) * SMOOTHING_THRESHOLD
      current_loc_y = previous_loc_y + (y_mouse_pointer - previous_loc_y) * SMOOTHING_THRESHOLD

      # Move the mouse cursor to the mapped coordinates
      pyautogui.moveTo(current_loc_x, current_loc_y)
      previous_loc_x, previous_loc_y = current_loc_x, current_loc_y
      
      print(f"Moving mouse to: ({x_mouse_pointer}, {y_mouse_pointer})")
      print(f"Index finger tip coordinates: ({index_finger_x}, {index_finger_y})")
      print(f"Screen size: ({screen_width}, {screen_height}), Frame size: ({frame_width}, {frame_height})")
      cv2.circle(annotated_image, (index_finger_x, index_finger_y), 15, (255, 0, 0), cv2.FILLED)
    
    ## Check if both the index finger and middle finger are up. The code checks if both the index finger (fingersUp[1]) and middle finger (fingersUp[2]) are up, which indicates that the user intends to perform a click action. When this condition is met, the code triggers a mouse click using pyautogui.click() and visually indicates the clicking mode by drawing a filled rectangle around the index finger tip on the annotated image.
    if index_finger_up and middle_finger_up:
      print("Clicking Mode")
      ## Get the length between the index finger tip and middle finger tip. The code calculates the distance between the index finger tip and middle finger tip using the handDetector's get_distance() method, which returns the length (distance) between the two specified landmarks. This distance can be used to determine if the fingers are close enough to be considered a click action.
      length, annotated_image, get_line_coords  = hand_detector.get_distance((index_finger_x, index_finger_y), (middle_x , middle_y), annotated_image)
      
      wrist_x, wrist_y = (hand_landmarks[hand_detector.wrist][1], hand_landmarks[hand_detector.wrist][2])  # Wrist coordinates (x0, y0, z0)
      middle_finger_mcp_x, middle_finger_mcp_y = (hand_landmarks[hand_detector.finger_mcp[2]][1], hand_landmarks[hand_detector.finger_mcp[2]][2])  # Middle finger MCP joint coordinates (x9, y9, z9)
      ## Get the reference length between the wrist and the middle finger MCP joint. The code calculates a reference length between the wrist (landmark 0) and the middle finger MCP joint (landmark 9) using the handDetector's get_distance() method. This reference length can be used to normalize the distance between the index and middle finger tips, allowing for a more consistent click detection regardless of hand size or distance from the camera.
      ref_length, _, _ = hand_detector.get_distance((wrist_x, wrist_y), (middle_finger_mcp_x, middle_finger_mcp_y), annotated_image, to_draw_circle_key_point=False, to_draw_line=False)
      
      middle_point_coords_x = get_line_coords[4] 
      middle_point_coords_y = get_line_coords[5]
      
      ## Normalize the distance between the index and middle finger tips by dividing it by the reference length. The code calculates a normalized distance by dividing the length between the index and middle finger tips by the reference length between the wrist and middle finger MCP joint. This normalization helps to account for variations in hand size and distance from the camera, providing a more reliable measure for determining if a click action should be triggered.
      normalized_length = length / ref_length if ref_length != 0 else 0
      ## Check if the distance between the index finger tip and middle finger tip is joined
      print(f"Normalized length between index and middle finger tips: {normalized_length:.2f}")
      if normalized_length < THRESHOLD_CLICK:
        print("Joined")
        ##Double click detection: Check if the time since the last click is within the double click threshold. The code compares the current time with the last click time to determine if a double click should be registered. If the time difference is less than the defined DOUBLE_CLICK_THRESHOLD, it triggers a double click action using pyautogui.doubleClick(). Otherwise, it performs a single click and updates the last_click_time to the current time.
        current_time = time.time()
        # Use previous mouse location for clicking if moving mode wasn't active
        click_x = previous_loc_x if previous_loc_x else screen_width // 2
        click_y = previous_loc_y if previous_loc_y else screen_height // 2
        if current_time - last_click_time < DOUBLE_CLICK_THRESHOLD:
          ## Move the mouse cursor to the click coordinates before performing the double click action. The code uses pyautogui.moveTo() to move the mouse cursor to the calculated click coordinates (click_x, click_y) before executing the double click action with pyautogui.doubleClick(). This ensures that the double click is performed at the correct location on the screen, even if the mouse cursor was not recently moved due to the smoothing mechanism.
          pyautogui.moveTo(click_x, click_y)
          ## Perform a double click action at the specified coordinates. The code uses pyautogui.doubleClick() to simulate a double click at the given (click_x, click_y) coordinates on the screen. The interval parameter is set to 0.2 seconds to ensure that the double click is registered correctly, especially on macOS where timing can be more sensitive for double click detection.
          pyautogui.doubleClick(x=click_x, y=click_y, interval=0.2)  # macOS-friendly
          print("Double Click!")
          last_click_time = 0  # reset to prevent triple click
        else:
          pyautogui.click(click_x, click_y)
          print("Single Click!")
          last_click_time = current_time
        cv2.circle(annotated_image, (middle_point_coords_x, middle_point_coords_y), 15, (0, 255, 0), cv2.FILLED)
      else:
        print("Apart")
        # Use middle finger Y for scroll control
        current_y = middle_y

        if prev_scroll_y is not None:
            delta = current_y - prev_scroll_y

            # Deadzone to prevent jitter
            if abs(delta) > SCROLL_DEADZONE:
                scroll_buffer.append(delta)
                scroll_buffer = scroll_buffer[-SCROLL_SMOOTHING:]
                smooth_delta = sum(scroll_buffer) / len(scroll_buffer)
                scroll_amount = int(-smooth_delta * SCROLL_SENSITIVITY)
                pyautogui.scroll(scroll_amount)
                print(f"Scrolling: {scroll_amount}")
                
        prev_scroll_y = current_y
        cv2.putText(
            annotated_image,
            "Scroll Mode",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

  return annotated_image



      


if __name__ == "__main__":
  video_capture_template(
    video_source=0,
    loop_forever=True,
    custom_logic=ai_virtual_mouse_logic,
    window_name="AI Virtual Mouse",
    resolution=(650, 500),
    center_window=True,
    draw_fps=True
  )