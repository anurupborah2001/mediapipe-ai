

from glob import glob
import os
import time
import sys
from pathlib import Path

import cv2
import pyautogui
import numpy as np

from module.hand_detector import HandDetector
from util import utility

hWIDTH, hHEIGHT = 1280, 720

width, height = pyautogui.size()
screen_width = width
screen_height = height

## Calculate center position
x = int((screen_width - hWIDTH) / 2)
y = int((screen_height - hHEIGHT) / 2)

## Move window to center
cv2.moveWindow("AI Painter", x, y)

cap = cv2.VideoCapture(0)
cap.set(3, hWIDTH)
cap.set(4, hHEIGHT)
# Create resizable window
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

## Setting thickness of eraser and brush. The code defines the thickness of the eraser and brush tools used for drawing on the canvas. The eraser thickness is set to 100 pixels, while the brush thickness is set to 15 pixels. These values can be adjusted based on user preference for a more customized drawing experience.
eraser_thickness = 100
brush_thickness = 15

# cap = cv2.VideoCapture(0)
# cap.set(3, hWIDTH)
# cap.set(4, hHEIGHT)

x_previous, y_previous = 0, 0
drawColor = (255, 102, 196)
ERASER_COLOR = (255, 255, 255)

handDetector = HandDetector()
currentTime = 0
previousTime = 0

img_canvas = np.zeros((hHEIGHT, hWIDTH, 3), np.uint8)
path, _= utility.get_calling_folder()
print(f"Calling folder path: {path}")
##Get the list of the fingers images
folder_path = f"{path}/images/paint"
# get_full_path = f"{os.getcwd() + '/' + folder_path}"
finder_img_list = sorted(
    os.listdir(folder_path),
    key=lambda x: int(os.path.splitext(x)[0])
)
header_painter_img = []
for img_path in finder_img_list:
    img = cv2.imread(os.path.join(folder_path, img_path))
    header_painter_img.append(img)
    
header_image = header_painter_img[0]

while True:
  success, imgFrame = cap.read()
  if not success:
    break
  
  # Flip for natural interaction
  annotated_image, landmarks_list_arr = handDetector.draw_landmarks(imgFrame, to_draw_bounding_box=False, to_draw_center_point=False)
  
  if len(landmarks_list_arr) != 0:
    hand_landmarks, _, _, _ = landmarks_list_arr[0]
    fingersUp = handDetector.fingers_up(hand_landmarks)
    
    ## Getting the tip of the finger for index and middle finger. The code retrieves the coordinates of the tip of the index and middle fingers from the landmarks_list using the handDetector's tipIds attribute, which contains the indices of the fingertip landmarks.
    x1, y1, z1 = hand_landmarks[handDetector.fingerTips[1]][1:]  # Index finger tip
    x2, y2, z2 = hand_landmarks[handDetector.fingerTips[2]][1:]  # Middle finger tip

    if fingersUp[1] and fingersUp[2]:
      print("Selection Mode")
      if y1 < 135:
        if 250 < x1 < 450:
          header_image = header_painter_img[0]
          drawColor = (196, 102, 255)
        elif 550 < x1 < 750:
          header_image = header_painter_img[1]
          drawColor = (0, 191, 99)
        elif 750 < x1 < 950:
          header_image = header_painter_img[2]
          drawColor =  (255, 0, 0)
        elif 950 < x1 < 1280:
          header_image = header_painter_img[3]
          drawColor = (0, 0, 0)
          
      cv2.rectangle(annotated_image, (x1, y1 - 25), (x2, y2 + 25), drawColor if drawColor != (0,0,0) else ERASER_COLOR, cv2.FILLED)
      
    if fingersUp[1] and not fingersUp[2]:
      print("Drawing Mode")
      is_eraser = drawColor == (0, 0, 0)
      
      # Show cursor
      if is_eraser:
          cv2.circle(annotated_image, (x1, y1), 20, ERASER_COLOR, cv2.FILLED)
      else:
          cv2.circle(annotated_image, (x1, y1), 10, drawColor, cv2.FILLED)

      if x_previous == 0 and y_previous == 0:
        x_previous, y_previous = x1, y1
        
      # cv2.line(annotated_image, (x_previous, y_previous), (x1, y1), drawColor, brush_thickness)
      if is_eraser:
        cv2.line(img_canvas, (x_previous, y_previous), (x1, y1), (0, 0, 0), eraser_thickness)
      else:
        ## Draw a line on the annotated_image from the previous point (xp, yp) to the current point (x1, y1) using the specified drawColor and brush_thickness. The code checks if the previous point is not None (i.e., there is a valid previous point) before drawing the line.
        ## This allows for continuous drawing as the user moves their finger across the screen.
        cv2.line(img_canvas, (x_previous, y_previous), (x1, y1), drawColor, brush_thickness)

      x_previous, y_previous = x1, y1

  ##FPS Measurement: Calculate and display the frames per second (FPS) of the video feed. The code calculates the time difference between the current frame and the previous frame to determine the FPS, which is then displayed on the screen using OpenCV's putText function.
  currentTime = time.time()
  fps = 1 / (currentTime - previousTime) if (currentTime - previousTime) > 0 else 0
  previousTime = currentTime

  cv2.putText(annotated_image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  
  img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
  _, img_inverted = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
  img_inverted = cv2.cvtColor(img_inverted, cv2.COLOR_GRAY2BGR)
  annotated_image = cv2.bitwise_and(annotated_image, img_inverted)
  annotated_image = cv2.bitwise_or(annotated_image, img_canvas)
  

  annotated_image[0:190, 0:1280] = header_image
  cv2.imshow("AI Virtual Painter", annotated_image)
  # cv2.imshow("Canvas", img_canvas)
  # cv2.imshow("Inv", img_inverted)
  ## Stop the loop and close the application when the 'Esc' key is pressed. The waitKey function waits for a key event for a specified amount of time (in this case, 1 millisecond) and checks if the 'Esc' key (ASCII code 27) is pressed to break the loop and release resources.
  if cv2.waitKey(1) & 0xFF == 27:
    break
  
cap.release()
cv2.destroyAllWindows()
    