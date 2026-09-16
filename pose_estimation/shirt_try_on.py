import cv2
import os
from openvisionkit.capture.video_template import video_capture_template
from module.pose_detector import PoseDetector
from util import utility
screen_resolution = (1280, 720)
path, _= utility.get_calling_folder()
detector = PoseDetector()
asset_folder = f"{path}/images/shirt_try_on/shirts/" 
list_shirts = os.listdir(asset_folder)
shirt_sizes = []
for shirt in list_shirts:
  #Get the size of the shirt and save it in array
  shirt_image = utility.load_image(f"{asset_folder}/{shirt}", False)
  shirt_sizes.append(shirt_image.shape)
  print(f"Shirt: {shirt}, Size: {shirt_image.shape}")

left_button = cv2.imread(f"{path}/images/shirt_try_on/buttons/left.png", cv2.IMREAD_UNCHANGED)
right_button = cv2.imread(f"{path}/images/shirt_try_on/buttons/right.png", cv2.IMREAD_UNCHANGED)
left_button = cv2.resize(left_button, (100, 100))
right_button = cv2.resize(right_button, (100, 100))
  
##Import the Shits from the assets folder
shirt_no = 0

def try_shirt_on_logic(frame):
  global shirt_no
  # Detect pose landmarks in the frame
  # for shirt in list_shirts:
  #   shirt_image = utility.load_image(f"{path}/images/shirts/{shirt}", False)
  annotated_frame, pose_landmarks = detector.detect(frame, draw_landmarks=True)
  landmark_list = detector.get_all_postion(annotated_frame, pose_landmarks)
  if landmark_list is not None:
    # print("Landmarks detected:", landmark_list)
    if len(landmark_list) > 12:  # Ensure that there are enough landmarks detected
      shoulder_measurements = detector.get_measurements_between_landmarks(annotated_frame, landmark_list[11], landmark_list[12])
      width_of_shoulder = shoulder_measurements["width_px"]
      neck = detector.get_neck_landmark(annotated_frame, landmark_list)

      shirt_img = cv2.imread(f"{asset_folder}{list_shirts[shirt_no]}", cv2.IMREAD_UNCHANGED)
      shirt_height_width_ratio = shirt_sizes[shirt_no][0] / shirt_sizes[shirt_no][1]
      shirt_width_px = int(width_of_shoulder * 1.6)  # wider than joint-to-joint for realistic coverage
      shirt_img = cv2.resize(shirt_img, (shirt_width_px, int(shirt_width_px * shirt_height_width_ratio)))

      # Collar sits ~15% from top of shirt image; align that point to neck position
      shirt_x = int(neck["pixel_x"]) - shirt_img.shape[1] // 2
      shirt_y = int(neck["pixel_y"]) - int(shirt_img.shape[0] * 0.15)

      try:
          annotated_frame = utility.overlay_transparent(annotated_frame, shirt_img, (shirt_x, shirt_y))
      except Exception as e:
          print(f"Overlay error: {e}")
          
      annotated_frame = utility.overlay_transparent(annotated_frame, left_button, (100, screen_resolution[1] // 2))
      annotated_frame = utility.overlay_transparent(annotated_frame, right_button, (1100, screen_resolution[1] // 2))
      
      
      if landmark_list[16]["x"] < 1100 and landmark_list[16]["x"] > 100 and landmark_list[16]["y"] > screen_resolution[1] // 2:
        if landmark_list[16]["x"] < 200:  # Left button area
          shirt_no = (shirt_no - 1) % len(list_shirts)
        elif landmark_list[16]["x"] > 1100:  # Right button area
          shirt_no = (shirt_no + 1) % len(list_shirts)
        else:
          shirt_no = 0
      
      
      

      # print(f"Adjusted width: {width_of_shirt_adjusted} and adjusted height: {width_of_shirt_adjusted * shirt_heght_width_ratio}")
      # ##resize the image and scale based on the width of the shoulder
      # shirt_img = cv2.resize(shirt_img, (int(width_of_shirt_adjusted), int(width_of_shirt_adjusted * shirt_heght_width_ratio)))
      # annotated_frame = utility.overlay_transparent(annotated_frame, shirt_img, (int(neck["pixel_x"]), int(neck["pixel_y"])))
  
  # Placeholder for try shirt on logic
  # You can implement the logic to overlay a shirt on the detected person in the frame
  # For now, we will just return the original frame
  
  return annotated_frame

if __name__ == "__main__":
  video_capture_template(
    resolution=screen_resolution,
    custom_logic=try_shirt_on_logic,
    window_name="Try Shirt On",
    draw_fps=False,
    enable_screenshot=True
  )