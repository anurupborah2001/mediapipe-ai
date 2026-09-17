import cv2
import math
from pynput.keyboard import Controller
import pygame
import time

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
from util import utility
from util.utility import rectangle_corners


hand_detector = HandDetector(max_hands=1, detection_confidence=0.7)
path, _= utility.get_calling_folder()
pygame.mixer.init()
click_sound = pygame.mixer.Sound(f"{path}/audio/click.wav")
PINCH_THRESHOLD = 0.25

keyboard_keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[" , "]"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
pressed_keys = ""

keyboard = Controller()

class Button:
    def __init__(self, pos, text, size=(60, 60)):
        self.pos = pos
        self.size = size
        self.text = text

keyboard_buttons = []   
for keyboard_row in keyboard_keys:
    for key_index, key in enumerate(keyboard_row):
        x = 100 + key_index * 70
        y = 100 + keyboard_keys.index(keyboard_row) * 70
        keyboard_buttons.append(Button((x, y), key))
        

def draw_keyboard(frame, largest_width, largest_height):
  for button in keyboard_buttons:
    x, y = button.pos
    w, h = button.size
    if  x + w > largest_width:
      largest_width = x + w
    if  y + h > largest_height:
      largest_height = y + h  
    rectangle_corners(frame, (x, y, w, h), 30)
    cv2.rectangle(frame, (x, y), (x + w, y + h),  (255, 0, 255), cv2.FILLED)
    cv2.putText(frame, button.text, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  ## Add a transparent black rectangle background for the keyboard area
  overlay = frame.copy()
  cv2.rectangle(overlay, (80, 80), (largest_width + 30, largest_height + 30), (0, 0, 0), cv2.FILLED)
  alpha = 0.5
  cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
  return frame , largest_width, largest_height
        
def ai_virtual_keyboard_logic(frame):
  largest_height , largest_width = 0, 0
  global pressed_keys
  annotated_image, landmarks_list_arr = hand_detector.draw_landmarks(frame, to_draw_bounding_box=False, to_draw_center_point=True)
  ## Show the keyboard on  when the screen loads
  annotated_image, modified_largest_length, modified_largest_width = draw_keyboard(annotated_image, largest_width, largest_height)
  
  if len(landmarks_list_arr) != 0:
    landmarks_list, _, _, _ = landmarks_list_arr[0]
    for button in keyboard_buttons:
      x, y = button.pos
      w, h = button.size
      ##check if the index finger tip is hovering over the button area. The code checks if the x and y coordinates of the index finger tip are within the boundaries of the button area. If the index finger tip is hovering over the button, it changes the button color to light pink and displays the button text in white, indicating that the user is hovering over the button.
      if x < landmarks_list[hand_detector.fingerTips[1]][1] < x + w and y < landmarks_list[hand_detector.fingerTips[1]][2] < y + h:
        ## make the  background color for the hovering part darker
        cv2.rectangle(annotated_image, (x, y), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
        cv2.putText(annotated_image, button.text, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ##find the distance between the index finger tip and the middle finger tip. The code calculates the distance between the index finger tip and the middle finger tip using the coordinates obtained from the landmarks_list. The distance is calculated using the Euclidean distance formula, which takes into account the x and y coordinates of both fingertips.
        index_finger_coords = (landmarks_list[hand_detector.fingerTips[1]][1], landmarks_list[hand_detector.fingerTips[1]][2])
        middle_finger_coords = (landmarks_list[hand_detector.fingerTips[2]][1], landmarks_list[hand_detector.fingerTips[2]][2])
        
        ###LOGIC: 1 logic to check if the fingers are joined together to register a click. The code calculates the distance between the index finger tip and middle finger tip, and then normalizes this distance using the palm width (the distance between the wrist and the middle finger MCP joint). If the normalized distance is less than a defined threshold (PINCH_THRESHOLD), it is considered a click action, and the corresponding button is registered as clicked.
        # length, annotated_image, _ = hand_detector.get_distance(index_finger_coords, middle_finger_coords, annotated_image)
        # # Normalize with palm width so pinch detection is stable across camera distances.
        # palm_width = math.hypot(
        #   landmarks_list[17][1] - landmarks_list[5][1],
        #   landmarks_list[17][2] - landmarks_list[5][2],
        # )
        # normalized_length = length / max(palm_width, 1.0)
        # ## check if the index finger and middle finger are close enough to register a click. If the normalized distance is less than the threshold, it indicates that the user has made a click gesture by bringing the index and middle fingertips close together. The code then changes the button color to light pink, displays the button text in white, plays a click sound, and adds the button text to the pressed_keys string. 
        # print(f"Normalized length: {normalized_length} and Length: {length}")      
        # if normalized_length < PINCH_THRESHOLD:
            

        ##LOGIC: 2 logic to check if the fingers are joined together to register a click. The code uses the hand_detector's is_fingers_joined() method to determine if the index finger tip and middle finger tip are joined together. This method takes the indices of the fingertips, the annotated image, the landmarks list, and a threshold as arguments, and returns a boolean indicating whether the fingers are joined based on the specified threshold. If the fingers are joined, it registers a click action for the corresponding button.
        is_joined = hand_detector.is_fingers_joined(hand_detector.fingerTips[1], hand_detector.fingerTips[2], annotated_image, landmarks_list, threshold=0.30)
        print(f"Is joined: {is_joined}")

        if is_joined:
            print(f"Button {button.text} clicked!")
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h),  (255, 182, 193), cv2.FILLED)
            cv2.putText(annotated_image, button.text, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            click_sound.play()
            pressed_keys += button.text
            keyboard.press(button.text)
            time.sleep(1)  # Add a small delay to prevent multiple rapid clicks
            keyboard.release(button.text)

  ##Add a text label "AI Virtual Keyboard" above the keyboard area
  cv2.putText(annotated_image, "AI Virtual Keyboard", (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  ## Add a text box to display the pressed keys. The code initializes an empty string pressed_keys to store the keys that have been pressed. When a button is clicked, the corresponding key is added to the pressed_keys string. The text box is displayed at the top of the screen, showing the keys that have been pressed by the user.
  cv2.rectangle(annotated_image, (80, modified_largest_width + 40), (modified_largest_length + 30, modified_largest_width + 70), (255, 255, 255), cv2.FILLED)
  cv2.putText(annotated_image, pressed_keys, (90, modified_largest_width + 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

  return annotated_image

if __name__ == "__main__":
    print("This is a module for the AI Virtual Keyboard project. Please import this module in your main script to use its functionality.")
    video_capture_template(
        video_source=0, 
        custom_logic=ai_virtual_keyboard_logic,
        window_name="AI Virtual Keyboard",
        resolution=(1280, 720),
        center_window=True,
        draw_fps=True
    )
    