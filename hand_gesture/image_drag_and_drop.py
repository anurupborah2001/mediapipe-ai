from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
import cv2
from glob import glob
from pathlib import Path

from util import utility 

path, _= utility.get_calling_folder()

hand_detector = HandDetector()

class DrawingSpec:
    def __init__(self, path, origin, imgtype, target_size=(150, 150)):
      self.path = path
      self.origin = origin
      self.imgtype = imgtype
      self.target_size = target_size
      if imgtype == ".png":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
      else:
        img = cv2.imread(path)
        
      if img is None:
        print("Failed to load image:", path)
        raise ValueError(f"Failed to load image: {path}")
        
      # Resize here the image to the target size using cv2.INTER_AREA interpolation method for better quality when reducing the image size. This ensures that the image maintains its aspect ratio and visual quality when resized to fit within the specified dimensions.
      self.image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
      self.size = self.image.shape[1], self.image.shape[0]
    
    def update_position(self, finger_cursor_pos):
       img_x, img_y = self.origin
       img_w, img_h = self.size
      
      ##check of the finger cursor is within the image boundaries to update the position of the image accordingly. If the cursor is outside the boundaries, we can choose to either keep the image at its current position or move it to the edge of the screen.
       if img_x <= finger_cursor_pos[0] <= img_x + img_w and img_y <= finger_cursor_pos[1] <= img_y + img_h:
          self.origin = (finger_cursor_pos[0] - img_w // 2, finger_cursor_pos[1] - img_h // 2)


list_images = []
image_drag_n_drop_folder = f"{path}/images/drag_and_drop"
for idx, filepath in enumerate(glob(f"{image_drag_n_drop_folder}/*")):
  path = Path(filepath)
  list_images.append(DrawingSpec(filepath, [100 + idx * 200, 100], path.suffix))

def image_drag_and_drop_logic(frame):
    annotated_image, landmarks_list_arr = hand_detector.draw_landmarks(frame, to_draw_bounding_box=False, to_draw_center_point=False)
    if len(landmarks_list_arr) != 0:  
      hand_landmarks, _, _, _ = landmarks_list_arr[0]
      ## place the image on the screen
       ##Check if the index and middle fingers are joined (pinching gesture) to enable dragging and dropping of the image. The is_fingers_joined function checks if the distance between the tips of the index and middle fingers is below a certain threshold, indicating that they are joined together.
      is_joined = hand_detector.is_fingers_joined(hand_detector.fingerTips[1], hand_detector.fingerTips[2], annotated_image, hand_landmarks, threshold=0.45)
      if is_joined:
        print("Fingers joined - dragging image")
        index_co_ordinates = hand_landmarks[hand_detector.fingerTips[2]][1:]  # Get the x, y coordinates of the index finger tip
        for drawing_spec in list_images:
          drawing_spec.update_position(index_co_ordinates)  # Update the position of the image based on the index finger coordinates
             
    try:
      for drawing_spec in list_images:
        x, y = drawing_spec.origin
        w, h = drawing_spec.size
        H, W = annotated_image.shape[:2]
        if x + w > W or y + h > H:
            continue
        if drawing_spec.imgtype == ".png":
          annotated_image = utility.overlay_transparent(annotated_image, drawing_spec.image, (x, y))  
        else: 
          annotated_image[y:y+h, x:x+w] = drawing_spec.image   
    except Exception as e:
      print("Overlay error:", e)  
         
    return annotated_image

if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=False,      
        custom_logic=image_drag_and_drop_logic,
        window_name="Image Drag and Drop",
        resolution=(680, 320),
        center_window=True,
        draw_fps=True
    )