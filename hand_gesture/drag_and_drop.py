import cv2
import pyautogui
import numpy as np

from module.hand_detector import HandDetector
from util.utility import rectangle_corners


hWIDTH, hHEIGHT = 1280, 720
width, height = pyautogui.size()
screen_width = width
screen_height = height

## Calculate center position
x = int((screen_width - hWIDTH) / 2)
y = int((screen_height - hHEIGHT) / 2)

## Move window to center
cv2.moveWindow("Drag and Drop", x, y)


cap = cv2.VideoCapture(0)
cap.set(3, hWIDTH)
cap.set(4, hHEIGHT)

handDetector = HandDetector(detection_confidence=0.5)
colorR = (255, 0, 255)
center_x, center_y, width, height = 100, 100, 200, 200

class DraggingRectangle:
  """
  A class representing a draggable rectangle that can be moved based on hand gestures.
  Attributes:
      posCenter: A list [x, y] representing the center position of the rectangle.
      posRect: A list [width, height] representing the size of the rectangle.
  Methods:
      update_position(posCursor): Updates the position of the rectangle based on the cursor position if the cursor is within the rectangle's area.
  """
  def __init__(self, positionCenter, posRect = [200, 200]):
    self.posCenter = positionCenter
    self.rectSize = posRect

  ## Update the position of the rectangle based on the cursor position if the cursor is within the rectangle's area
  def update_position(self, positionCursor):
    center_x, center_y = self.posCenter
    width, height = self.rectSize
    ## if the index finger tip is within the rectangle, update the rectangle position to follow the cursor
    if (center_x - width // 2 < positionCursor[0] < center_x + width // 2) and (center_y - height // 2 < positionCursor[1] < center_y + height // 2):
      print("Cursor is within the rectangle, updating position")
      self.posCenter = positionCursor


##Simulate 6 rectangles with different positions and sizes
rectangleList = []
for i in range(5):
  rectangleList.append(DraggingRectangle(positionCenter = [i * 250 + 150, 150]))

while True:
  ret, imgFrame = cap.read()
  if not ret:
    break
  imgFrame = cv2.flip(imgFrame, 1)
  annotated_image, landmarks_list_arr = handDetector.draw_landmarks(imgFrame, to_draw_bounding_box=False, to_draw_center_point=False)
    
  if len(landmarks_list_arr) != 0:
    hand_landmarks, _, _, _ = landmarks_list_arr[0]
    ## Get the distance between index and middle finger tips
    index_finger_coords = (hand_landmarks[handDetector.fingerTips[1]][1], hand_landmarks[handDetector.fingerTips[1]][2])  # Index finger tip coordinates (x1, y1)
    middle_finger_coords = (hand_landmarks[handDetector.fingerTips[2]][1], hand_landmarks[handDetector.fingerTips[2]][2])  # Middle finger tip coordinates (x2, y2)
    distance, _ , _ = handDetector.get_distance(index_finger_coords, middle_finger_coords, annotated_image, to_draw_circle_key_point=False) 
    print("Distance between index and middle finger tips:", distance)
    if distance < 50:
      indexPosition = hand_landmarks[8]  ## Get the position of the index finger tip
      cursorPostion = (indexPosition[1], indexPosition[2])  ## Convert to tuple for easier handling
      print("closing the index and middle fingers")
      for rectangle in rectangleList:
        rectangle.update_position(cursorPostion)  ## Update the position of the rectangle based on the index finger tip position

  ## Draw Transperency
  imgNew = np.zeros_like(annotated_image, np.uint8)
  for rect in rectangleList:
      cx, cy = rect.posCenter
      w, h = rect.rectSize
      cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                    (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
      rectangle_corners(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

  out = annotated_image.copy()
  alpha = 0.5
  mask = imgNew.astype(bool)
  out[mask] = cv2.addWeighted(annotated_image, alpha, imgNew, 1 - alpha, 0)[mask]
  
  cv2.imshow("Drag and Drop", out)
  
  if cv2.waitKey(1) & 0xFF == 27:
    break
  
cap.release()
cv2.destroyAllWindows()