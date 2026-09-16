import os
import cv2
import pyautogui
import numpy as np
import time
from pathlib import Path
import math

from module.hand_detector import HandDetector

# hWIDTH, hHEIGHT = 1280, 720
width, height = pyautogui.size()
screen_width = width
screen_height = height

image_size = 300
offset_value = 20

# ## Calculate center position
# x = int((screen_width - hWIDTH) / 2)
# y = int((screen_height - hHEIGHT) / 2)

## Move window to center
# cv2.moveWindow("Drag and Drop", x, y)


cap = cv2.VideoCapture(0)
# cap.set(3, hWIDTH)
# cap.set(4, hHEIGHT)

handDetector = HandDetector(max_hands=1,detection_confidence=0.7)

letter = "A"
folder_dataset = f"hand-gesture/hand-sign-detection/dataset/{letter}"

# INTER_AREA (Downsampling/Shrinking)
# - Best for: Shrinking (downscaling) images.
# - Method: Resamples using pixel area relation, avoiding moiré patterns.
# - Behavior: When used to enlarge images, it acts similar to INTER_NEAREST. 

# INTER_LINEAR (Default/General)
# - Best for: General resizing, zooming (upsampling), and high speed.
# - Method: Bilinear interpolation (uses 2x2 neighborhood).
# Behavior: A good compromise between speed and quality. It is the default method in OpenCV. 

# INTER_CUBIC (Upsampling/High Quality)
# - Best for: Enlarging (upscaling) images for higher quality.
# - Method: Bicubic interpolation (uses 4x4 neighborhood).
# - Behavior: Slower than INTER_LINEAR but produces smoother, sharper

while True:
  ret, imgFrame = cap.read()
  if not ret:
    break
  ##Convert the captured image to RGB format for processing with MediaPipe. The code uses OpenCV's cvtColor function to convert the captured image from BGR color space (used by OpenCV) to RGB color space (used by MediaPipe). This conversion is necessary because MediaPipe expects images in RGB format for accurate hand detection and landmark estimation.
  annotated_image, landmarks = handDetector.draw_landmarks(imgFrame, to_draw_center_point=False, to_draw_bounding_box=True, to_put_handle_label=False)
  if landmarks:
    landmarks_list, hand_bounding_box, landmark_params, hand_type = landmarks[0]
  else:
      landmarks_list, hand_bounding_box, landmark_params, hand_type = [], [], {}, None

  if len(landmarks_list) != 0:
    bbox_x, bbox_y, bbox_w, bbox_h = hand_bounding_box
    ## Create a white image of the same size as the annotated image and resize it to a specified image size. The code creates a white image using NumPy's ones_like function, which generates an array of the same shape as the annotated image filled with ones (white). Then, it resizes this white image to the desired dimensions (image_size x image_size) using OpenCV's resize function, which allows for interpolation to maintain image quality during resizing.
    img_white = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    # img_white =  cv2.resize(img_white, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    ##Crop out the hand region from the annotated image using the bounding box coordinates and an offset value. The code extracts a region of interest (ROI) from the annotated image based on the bounding box coordinates (bbox_x, bbox_y, bbox_w, bbox_h) and an additional offset value to ensure that the entire hand is captured. This cropped image can then be used for further processing or analysis, such as gesture recognition or hand tracking.
    cropped_image = annotated_image[bbox_y - offset_value : bbox_y + bbox_h + offset_value, bbox_x - offset_value : bbox_x + bbox_w + offset_value]
    
    ##check if the hand landmarklist is greater than white image size, if it is, then resize the cropped image to fit within the white image. The code checks if the width or height of the cropped image exceeds the specified image size. If either dimension is larger than the image size, it resizes the cropped image to fit within the white image while maintaining the aspect ratio. This ensures that the cropped hand image can be properly displayed within the white background without distortion.
    # if cropped_image.shape[0] > image_size or cropped_image.shape[1] > image_size:
    #   cropped_image = cv2.resize(cropped_image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
      
    # Safety check to ensure cropped image is not empty before processing. The code checks if the cropped image is None or has a size of zero, which can occur if the bounding box coordinates are invalid or if there is an issue with the image capture. If either condition is true, it continues to the next iteration of the loop, preventing errors that could arise from attempting to process an empty or non-existent image.
    if cropped_image is None or cropped_image.size == 0:
        continue

    # Prevent divide by zero
    if bbox_w == 0 or bbox_h == 0:
        continue
    
    aspect_ratio = bbox_h / bbox_w
    
    ##If height is greater than width, then resize the cropped image to fit within the white image while maintaining the aspect ratio. The code checks if the aspect ratio of the bounding box (height divided by width) is greater than 1, indicating that the height is greater than the width. If this condition is true, it resizes the cropped image to fit within the white image while maintaining the aspect ratio, ensuring that the hand image is properly displayed without distortion.
    if aspect_ratio > 1:
      new_width = math.ceil(image_size / aspect_ratio)
      if new_width <= 0:
        continue
      cropped_image = cv2.resize(cropped_image, (new_width, image_size), interpolation=cv2.INTER_LINEAR)
      ##Provide gap between the cropped image and the white image by centering the cropped image within the white image. The code calculates the new width of the cropped image based on the aspect ratio and resizes it accordingly. Then, it places the resized cropped image onto the white background, centering it horizontally by calculating the appropriate starting x-coordinate. This creates a visually balanced composition where the hand image is centered within the white background.
      x_offset = (image_size - new_width) // 2
      img_white[0:cropped_image.shape[0], x_offset:x_offset + cropped_image.shape[1]] = cropped_image
    else:
      new_height = math.ceil(image_size * aspect_ratio)
      if new_height <= 0:
        continue
      cropped_image = cv2.resize(cropped_image, (image_size, new_height), interpolation=cv2.INTER_LINEAR)
      y_offset = (image_size - new_height) // 2
      img_white[y_offset:y_offset + cropped_image.shape[0],:] = cropped_image
      
    # cv2.imshow("ImageCrop", cropped_image)
    cv2.imshow("ImageWhite", img_white)


  cv2.imshow("Hand Detection Capture", annotated_image)
  key = cv2.waitKey(1)
  if key == ord('s'):
    if not os.path.exists(folder_dataset):
      os.makedirs(folder_dataset)
    cv2.imwrite(f"{folder_dataset}/{letter}_{time.time()}.jpg", img_white)
    print(f"Image saved to {folder_dataset}/{letter}_{time.time()}.jpg")
    
  if key & 0xFF == 27:
    break
  
cap.release()
cv2.destroyAllWindows()