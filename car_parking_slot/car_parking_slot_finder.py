
from openvisionkit.capture.video_template import video_capture_template
import pickle
import cv2
import numpy as np

from util.utility import put_text_rect
base_folder = "car_parking_slot"


width , height = 104, 48


def check_parking_slot_based_on_dilate(original_img, img_dilate, position_car_list):
  car_empty_space = 0
  for idx, (pos_x, pos_y) in enumerate(position_car_list):
    img_crop = img_dilate[pos_y:pos_y + height, pos_x:pos_x + width]
    count_non_zero = cv2.countNonZero(img_crop)
    if count_non_zero < 850:
      color = (0, 255, 0)  ## Green for empty slot
      status = "Empty"
      car_empty_space+=1
    else:
      color = (0, 0, 255)  # Red for occupied slot
      status = "Occupied"

    cv2.rectangle(original_img, (pos_x, pos_y), (pos_x + width, pos_y + height), color, 1)
    put_text_rect(original_img, str(count_non_zero), (pos_x, pos_y + height - 2), scale=0.5,
                           thickness=2, offset=0, colorR=color, font=cv2.FONT_HERSHEY_SIMPLEX)
    # cv2.putText(original_img, f"{status} ({count_non_zero})", (pos_x, pos_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
  cv2.putText(original_img, f"Empty Slots: {car_empty_space}/{len(position_car_list)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  return original_img
   

def detect_car_parking_slot_empty(frame, position_car_list):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    img_median = cv2.medianBlur(img_threshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_median, kernel, iterations=1)
    return img_dilate

def car_parking_slot_finder_logic(frame):
  # Load parking slot positions from the pickle file
  with open(f"{base_folder}/car_parking_slots.pkl", "rb") as f:
    position_car_list = pickle.load(f)
      
  img_dilate = detect_car_parking_slot_empty(frame, position_car_list)
  check_parking_slot_based_on_dilate(frame, img_dilate, position_car_list)

  # Draw rectangles around the parking slots
  # for pos_x, pos_y in position_car_list:
  #   cv2.rectangle(frame, (pos_x, pos_y), (pos_x + width, pos_y + height), (0, 255, 0), 2)

  return frame

if __name__ == "__main__":
    print("This is the main module for the car parking slot finder.")
    print("Please run 'car_parking_pickle_download.py' first to set up the parking slot coordinates.")
    video_capture_template( 
        video_source=f"{base_folder}/assets/car_parking.mp4",  # Use the video file as the source
        loop_forever=True,
        custom_logic=car_parking_slot_finder_logic,
        window_name="Car Parking Slot Finder",
        resolution=(1040, 480),  # Set to the size of the parking lot
        center_window=True,
        draw_fps=False
    )