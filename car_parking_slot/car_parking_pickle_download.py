
import pickle
import cv2
import requests

width , height = 104, 48
base_folder = "car_parking_slot"
##check if the pickle file exists, if not create it by downloading the image and saving the coordinates of the parking slots
try:  
  with open(f"{base_folder}/car_parking_slots.pkl", "rb") as f:
    position_car_list = pickle.load(f)
except FileNotFoundError:
  position_car_list = []

def mouse_callback(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    print(f"Mouse clicked at: ({x}, {y})")
    position_car_list.append((x, y))
  if event == cv2.EVENT_RBUTTONDOWN:
    print(f"Right mouse button clicked at: ({x}, {y})")
    for idx, (pos_x, pos_y) in enumerate(position_car_list):
      if pos_x < x < pos_x + width and pos_y < y < pos_y + height:
        print(f"Clicked on parking slot {idx} at position ({pos_x}, {pos_y})")
        del position_car_list[idx]
        break
      
  ## Save the updated list of parking slot positions to a pickle file
  with open(f"{base_folder}/car_parking_slots.pkl", "wb") as f:
    pickle.dump(position_car_list, f)
    
if __name__ == "__main__":
  while True:
    ## read image via cv2
    img = cv2.imread(f"{base_folder}/assets/car_parking.png")
    for pos_x, pos_y in position_car_list:
      cv2.rectangle(img, (pos_x, pos_y), (pos_x + width, pos_y + height), (0, 255, 0), 2)

    # cv2.rectangle(img, (50, 143), (154, 192), (0, 255, 0), 2)
    cv2.imshow("Car Parking Slot", img)
    cv2.setMouseCallback("Car Parking Slot", mouse_callback)
    
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
      break
    
cv2.destroyAllWindows()