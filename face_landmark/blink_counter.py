import os
import sys
import time

import cv2

from module.face_mesh_detector import FaceMeshDetector
from module.live_plot import LivePlot
from openvisionkit.capture.video_template import video_capture_template
from util.utility import stack_images_grid
from collections import deque

detector = FaceMeshDetector(
    model_path='./models/face_landmarker_v2_with_blendshapes.task',
    num_faces=2
)
blinks = 0
live_plot = LivePlot(640, 360, [30, 50])
ratioList = deque(maxlen=3)  # Store the eye aspect ratio values for plotting

def count_blinks(frame,blinks = 0):
    threshold = 0.5
    annotated_frame, faces, blendshapes, transformation_matrices, bboxes = detector.face_mesh_detection(frame, drawLandMarks=False)
    if faces:
      face = faces[0]
      ## Draw eye landmarks for visualization for the left eye
      for eyeLandmark in detector.LEFT_EYE_BLINK:
        cv2.circle(annotated_frame, face[eyeLandmark], 5,(255, 0, 255), cv2.FILLED)
        

      lefteye_up = face[159]
      lefteye_down = face[23]
      lefteye_left = face[130]
      lefteye_right = face[243]
      vertical_alignment, _= detector.distance_between_landmarks(lefteye_up, lefteye_down)
      horizontal_alignment, _= detector.distance_between_landmarks(lefteye_left, lefteye_right)

      cv2.line(annotated_frame, lefteye_up, lefteye_down, (0, 200, 0), 3)
      cv2.line(annotated_frame, lefteye_left, lefteye_right, (0, 200, 0), 3)
      
      ## Calculate the eye aspect ratio and update the live plot
      ratioList.append(int((vertical_alignment / horizontal_alignment) * 100))
      ratioAvg = sum(ratioList) / len(ratioList)
      
      imgPlot = live_plot.update(ratioAvg)
      annotated_frame = cv2.resize(annotated_frame, (640, 360))
      annotated_frame = stack_images_grid([annotated_frame, imgPlot], 2, 1)
    else:
      annotated_frame = cv2.resize(annotated_frame, (640, 360))
      annotated_frame = stack_images_grid([annotated_frame, annotated_frame], 2, 1)
      
    if not blendshapes:
      return blinks, annotated_frame
    for i, (face, bbox, blend, matrix) in enumerate(zip(faces, bboxes, blendshapes, transformation_matrices)):
        eye_left = blend.get('eyeBlinkLeft', 0)
        eye_right = blend.get(' eyeBlinkRight', 0)
        if eye_left > threshold:
            blinks += 1
        if eye_right > threshold:
            blinks += 1

    return blinks, annotated_frame

def bink_counter(frame):
    global blinks
    blink_count, annotated_frame = count_blinks(frame, blinks)
    blinks = blink_count  
    cv2.putText(annotated_frame, f"Blinks: {blink_count}", (20, 150), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    return annotated_frame
if __name__ == "__main__":
  video_capture_template(
    video_source=0,
    resolution=(1920, 1080),
    window_name="Blink Count",
    custom_logic=bink_counter,
    draw_fps=True,
    center_window=True
  )
