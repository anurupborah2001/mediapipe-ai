import sys
import time
import subprocess
from ctypes import POINTER, cast
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from module.hand_detector import HandDetector

hWIDTH, hHEIGHT = 700, 500

width, height = pyautogui.size()
screen_width = width
screen_height = height

## Calculate center position
x = int((screen_width - hWIDTH) / 2)
y = int((screen_height - hHEIGHT) / 2)

## Move window to center
cv2.moveWindow("Pose Window", x, y)

cap = cv2.VideoCapture(0)
cap.set(3, hWIDTH)
cap.set(4, hHEIGHT)
# Create resizable window
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

currentTime = 0
previousTime = 0  
handDetector = HandDetector()

## VOLUME STATE
volumeMeasure = 0
prevVolume = 0
volumeColor = (0, 255, 0)

volumeBar = 400
## AUTO CALIBRATION
cal_min = None
cal_max = None


##SMOOTHING
smoothness = 5
alpha = 0.25


##check if the os is Mac or Windows and set the appropriate font for text display. The code checks the platform using the sys module and sets the font variable accordingly. If the platform is Windows, it uses cv2.FONT_HERSHEY_SIMPLEX, while for other platforms (like Mac), it uses cv2.FONT_HERSHEY_DUPLEX.
if sys.platform.startswith('win'):
  from pycaw.pycaw import AudioUtilities
  from comtypes import CLSCTX_ALL
  device = AudioUtilities.GetSpeakers()
  interface = device.Activate(AudioUtilities.IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
  volume = cast(interface, POINTER(AudioUtilities.IAudioEndpointVolume))
  volumeMeasure = volume.GetMasterVolumeLevelScalar()
  currentVolume = volumeMeasure * 100
  volumeRange = volume.GetVolumeRange()
  minVol = volumeRange[0]
  maxVol = volumeRange[1]
elif sys.platform.startswith('darwin'):
  volumeMeasure = subprocess.check_output(
    "osascript -e 'output volume of (get volume settings)'",
    shell=True
  )
  currentVolume = volumeMeasure.decode().strip()
  print("Current Volume:", volumeMeasure.decode().strip())
else:
  print("Unsupported platform for audio control.")

while True:
  ret, imgFrame = cap.read()
  if not ret:
    break
  
  ## Detect the hand landmarks in the RGB image using the handDetector instance. The detected landmarks are stored in the hand_landmarks_list attribute of the handDetector object, which can be accessed for further processing or visualization.
  annotated_image, landmarks_list_arr =  handDetector.draw_landmarks(imgFrame, to_draw_bounding_box=True, to_draw_center_point=True)

  ##Create a line in btween thumb and index finger tips
  if len(landmarks_list_arr) != 0:
    hand_landmarks, bounding_box, landmark_params, _ = landmarks_list_arr[0]
    print(f"bounding_box: {bounding_box}")
    ## Restict the volume control with hands to a specific area of the screen. The code checks if the bounding box of the detected hand is within a certain region (defined by the coordinates of the bounding box) before allowing volume control based on hand gestures. 
    ## This helps to ensure that volume control is only activated when the hand is in a specific area, preventing unintended adjustments.
    length = bounding_box[2]
    breadth = bounding_box[3]
    area = length * breadth // 1000
    print(f"Area of the bounding box: {area}")
    
    ## Check if the area of the bounding box is within a certain range (between 350 and 1000 in this case) to determine if the hand is within the control area for volume adjustment. If the area is within this range, it indicates that the hand is at an appropriate distance from the camera for accurate gesture recognition and volume control.
    if 350 < area < 1000:
      
      ## Getting the distance between the thumb and index finger tips using the get_distance method of the handDetector object. The method takes the indices of the thumb tip (4) and index finger tip (8) as arguments, along with the annotated image, and returns the length of the line between these two points, as well as their coordinates and the center point. This information is used to control the system volume based on the distance between the thumb and index finger tips.
      thumb_tip_coords = (hand_landmarks[handDetector.fingerTips[0]][1], hand_landmarks[handDetector.fingerTips[0]][2])  # Thumb tip coordinates (x4, y4)
      index_finger_tip_coords = (hand_landmarks[handDetector.fingerTips[1]][1], hand_landmarks[handDetector.fingerTips[1]][2])  # Index finger tip coordinates (x8, y8)
      length, annotated_image, coords = handDetector.get_distance(thumb_tip_coords, index_finger_tip_coords, annotated_image)
      print(f"Length between thumb and index finger tips: {length}")
      
      ## check if the little finger is down by comparing the y-coordinate of the little finger tip (landmark 20) with the y-coordinate of the middle finger tip (landmark 12). If the little finger tip is below the middle finger tip, it indicates that the little finger is down. This condition is used to determine whether to activate volume control based on hand gestures.
      ## Little finger will control the volume. ie, if its down then the volume will be controlled by the distance between the thumb and index finger tips. If the little finger is not down, the volume control will not be activated, allowing for more precise control over when to adjust the volume based on hand gestures.
      fingers = handDetector.fingers_up(hand_landmarks)
      print(f"Fingers status (1 for up, 0 for down): {fingers}")


      if cal_min is None or length < cal_min:
                cal_min = length
      if cal_max is None or length > cal_max:
          cal_max = length

      # prevent divide-by-zero
      if cal_max - cal_min < 30:
          continue
        
      norm = np.interp(
        length,
        [cal_min, cal_max],
        [0, 100]
      )

      # quantize (smooth steps)
      norm = smoothness * round(norm / smoothness)
      
      volumeMeasure = alpha * norm + (1 - alpha) * prevVolume
      volumeMeasure = int(np.clip(volumeMeasure, 0, 100))
      prevVolume = volumeMeasure
      
      ## Only control the volume if the little finger is down. This condition ensures that volume control is activated only when the user intentionally lowers their little finger, providing a more intuitive and deliberate way to adjust the volume based on hand gestures.
      if fingers[4] == 0:
        volumeColor = (255, 0, 0)
        volumeMeasure = volumeMeasure
        if length < 120:
          print("Volume set to minimum")
          cv2.circle(annotated_image, (coords[4], coords[5]), 10, (0, 255, 0), cv2.FILLED)
      else:
        volumeColor = (0, 255, 0)

      print(f"Operatinng system {sys.platform} detected")
      ## Contol the system volume based on the length between the thumb and index finger tips. If the length is less than 50, it sets the volume to a specific level (e.g., -20 dB for Windows or 50% for Mac). The code checks the platform using the sys module and executes the appropriate command to adjust the system volume accordingly.
      if sys.platform.startswith('win'):
        volume.SetMasterVolumeLevelScalar(volumeMeasure, None)
      elif sys.platform.startswith('darwin'):
        subprocess.run([
              "osascript",
              "-e",
              f"set volume output volume {int(volumeMeasure)}"
          ])
      else:
        print("Unsupported platform for audio control.")
    

  volumeBar = np.interp(volumeMeasure, [0, 100], [400, 150])
  ## Convert RGB → BGR for OpenCV display
  ## Add a volume bar on the right side of the screen to visually represent the current volume level. The volume bar is drawn as a filled rectangle that changes its height based on the current volume level, providing a visual indication of the volume control.
  cv2.rectangle(annotated_image, (50, 150), (85, 400), (0, 255, 0), 3)
  cv2.rectangle(annotated_image, (50, int(volumeBar)), (85, 400), (0, 255, 0), cv2.FILLED)
  print(f"Volume Measure: {volumeMeasure}, Volume Bar Position: {volumeBar}")
  ## Add text to display the current volume level as a percentage. The text is positioned at the bottom left corner of the screen and updates dynamically based on the current volume level, providing a clear visual representation of the volume control.
  cv2.putText(annotated_image, f'{int(volumeMeasure)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 3)
  ## Add text to display the set volume level when the little finger is down. The text is displayed at the top right corner of the screen and updates dynamically based on the current volume level, providing feedback to the user about the volume setting when the little finger is used to control the volume.
  cv2.putText(annotated_image, f'Vol Set: {int(volumeMeasure)}', (1300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, volumeColor, 3)
  
  ##FPS Measurement: Calculate and display the frames per second (FPS) of the video feed. The code calculates the time difference between the current frame and the previous frame to determine the FPS, which is then displayed on the screen using OpenCV's putText function.
  currentTime = time.time()
  fps = 1 / (currentTime - previousTime) if (currentTime - previousTime) > 0 else 0
  previousTime = currentTime
  
  cv2.putText(annotated_image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  cv2.imshow("Hand Tracking", annotated_image)
  
  ## Stop the loop and close the application when the 'Esc' key is pressed. The waitKey function waits for a key event for a specified amount of time (in this case, 1 millisecond) and checks if the 'Esc' key (ASCII code 27) is pressed to break the loop and release resources.
  if cv2.waitKey(1) & 0xFF == 27:
    break
  
cap.release()
cv2.destroyAllWindows()