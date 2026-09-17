import math
import os
import random
import cv2
from openvisionkit.capture.video_template import video_capture_template

from module.face_mesh_detector import FaceMeshDetector
from util import utility
screen_resolution = (680, 320)
window_name = "Kids Game - Eatables"
path, _= utility.get_calling_folder()
eatable_items_path = f"{path}/images/eatable"
non_eatable_items_path = f"{path}/images/non_eatable"

eatable_items_dir_list = os.listdir(eatable_items_path)
non_eatable_items_dir_list = os.listdir(non_eatable_items_path)

eatable_items = []
non_eatable_items = []

def resizeImg(imgPath, target_size=(100, 100)):
  img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
  return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

for item in eatable_items_dir_list:
  eatable_items.append(resizeImg(f'{eatable_items_path}/{item}'))  
    
for item in non_eatable_items_dir_list:
  non_eatable_items.append(resizeImg(f'{non_eatable_items_path}/{item}'))
    
face_mesh_detector = FaceMeshDetector(num_faces=1, min_face_detection_confidence=0.8)

lipsList = face_mesh_detector.LIP_CORNERS + face_mesh_detector.LIP_CENTER
position_of_items = [300,0]
is_eatable = True
speed = 5
count = 0
current_item = eatable_items[0]
game_over = False

def reset_items():
  global position_of_items, is_eatable, current_item
  position_of_items[0] = random.randint(50, screen_resolution[1]-50)
  position_of_items[1] = 0 
  rand_no = random.randint(0, 2)
  if rand_no == 0:
    # position_of_items[0] = 300
    current_item =  non_eatable_items[random.randint(0, len(non_eatable_items)-1)]
    is_eatable = False 
  else:
    # position_of_items[0] = 300
    current_item =  eatable_items[random.randint(0, len(eatable_items)-1)]
    is_eatable = True
    
  return current_item

def face_distance_measurement_logic(frame):
  global current_item, position_of_items, is_eatable, game_over, count
  # Placeholder for face distance measurement logic
  annotated_frame, faces, _, _, _  = face_mesh_detector.face_mesh_detection(frame, drawLandMarks=False)
  if game_over is False and faces:
    face = faces[0]  # Assuming we're only interested in the first detected face
    
    for landmark_index in lipsList:
      cv2.circle(annotated_frame,  face[landmark_index], 5, (0, 255, 0), cv2.FILLED)
      
    position_of_items[1] += speed
    
    if position_of_items[1] > screen_resolution[0]:
      current_item = reset_items()

    ##Draw the item on the screen
    annotated_frame = utility.overlay_transparent(annotated_frame, current_item, position_of_items)
      
    upper_lip = face[face_mesh_detector.LIP_CENTER_TOP]
    lower_lip = face[face_mesh_detector.LIP_CENTER_BOTTOM]
    left_corner_lip = face[face_mesh_detector.LIP_LEFT_CORNERS]
    right_corner_lip = face[face_mesh_detector.LIP_RIGHT_CORNERS]
    upper_down_lip_distance, _ = face_mesh_detector.distance_between_landmarks(upper_lip, lower_lip)
    left_right_lip_distance, _ = face_mesh_detector.distance_between_landmarks(left_corner_lip, right_corner_lip)
    
    ##Get the center of the mouth
    center_mouth_x , center_mouth_y = (upper_lip[0] + lower_lip[0]) // 2, (upper_lip[1] + lower_lip[1]) // 2
    middle_point_item_x, middle_point_item_y = position_of_items[0] + 50, position_of_items[1] + 50
    cv2.line(annotated_frame, (center_mouth_x, center_mouth_y), (middle_point_item_x, middle_point_item_y), (0, 255, 0), 3)
    ratio_distance = round((upper_down_lip_distance / left_right_lip_distance)*100, 2) if left_right_lip_distance != 0 else 0
      
    if ratio_distance > 50:
      mount_state = "Open"
    else:
      mount_state = "Closed"
    cv2.putText(annotated_frame, f"Lips: {mount_state} ({ratio_distance}%)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 
    ##Find the distance of mouth center to the item
    distance_to_item, _ = face_mesh_detector.distance_between_landmarks((center_mouth_x, center_mouth_y), (middle_point_item_x, middle_point_item_y))
    if distance_to_item < 50 and mount_state == "Open":
      if is_eatable:
        current_item = reset_items()
        count += 1
      else:
        game_over = True
        cv2.putText(annotated_frame, "Game Over", (center_mouth_x + 30, center_mouth_y + 30), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 10)
    cv2.putText(annotated_frame, f"Success: {count}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
  else:     
    cv2.putText(annotated_frame, "Game Over", (300, 400), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 10)
  
    key = cv2.waitKey(1)
    if key == ord('r'):
      reset_items()
      game_over = False
      count = 0
      current_item = eatable_items[0]
      is_eatable = True
      
  return annotated_frame
if __name__ == "__main__":
    print("This is the kids_game_eatable module.")
    video_capture_template(
        video_source=0,
        loop_forever=False,      
        custom_logic=face_distance_measurement_logic,
        window_name=window_name,
        resolution=screen_resolution,
        draw_fps=False,
        enable_screenshot=True,
    )