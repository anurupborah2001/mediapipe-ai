
from glob import glob
import os
import time

import cv2
import pyautogui

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
from util import utility


hWIDTH, hHEIGHT = 700, 500
  
width, height = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3, hWIDTH)
cap.set(4, hHEIGHT)

handDetector = HandDetector()
previousTime = 0

##Get the list of the fingers images
path, _= utility.get_calling_folder()
folder_path = f"{path}/images/fingers"
# get_full_path = f"{os.getcwd() + '/' + folder_path}"
finder_img_list = sorted(
    os.listdir(folder_path),
    key=lambda x: int(os.path.splitext(x)[0])
)
finger_images = []
for fingerPath in finder_img_list:
  image = cv2.imread(f"{folder_path}/{os.path.basename(fingerPath)}")
  finger_images.append(image)

def finger_counter(imgFrame):
  annotated_image, landmarks_list_arr = handDetector.draw_landmarks(imgFrame, to_draw_bounding_box=False, to_draw_center_point=False)
  
  if len(landmarks_list_arr) != 0:
    hand_landmarks, _, _, _ = landmarks_list_arr[0]
    fingersUp = handDetector.fingers_up(hand_landmarks)
    print (f"Fingers up: {fingersUp}")
    total_fingers = fingersUp.count(1)
    print (f"Total fingers up: {total_fingers}")

    img_height, img_width, _ = finger_images[total_fingers - 1].shape
    annotated_image[0:img_height, 0:img_width] = finger_images[total_fingers - 1]
    cv2.rectangle(annotated_image, (200, 500), (20, 800), (0, 255, 0), cv2.FILLED)
    cv2.putText(annotated_image, str(total_fingers), (60, 700), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
  return annotated_image

if __name__ == "__main__":
  video_capture_template(
      video_source=0,
      loop_forever=True,      
      custom_logic=finger_counter,
      window_name="Finger Counter",
      resolution=(700, 500),
      center_window=True,
      draw_fps=True
  ) 