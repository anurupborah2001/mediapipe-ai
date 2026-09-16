from typing import List
import cv2
from module import hand_detector
from module.hair_segmentation import HairSegmentation
from module.hand_detector import HandDetector
from template.draw_object import DrawingObject
from openvisionkit.capture.video_template import video_capture_template

list_drawings = [
    DrawingObject(
        kind="circle",
        placement="top-left",
        size=(100, 100),
        color=(0, 255, 0),
    ),
    DrawingObject(
        kind="square",
        placement="top-right",
        size=(120, 120),
        color=(255, 0, 0),
    ),
    DrawingObject(
        kind="triangle",
        placement="bottom-left",
        size=(140, 140),
        color=(0, 0, 255),
    ),
    DrawingObject(
        kind="rectangle",
        placement="bottom-right",
        size=(180, 100),
        color=(255, 255, 0),
    ),
]


# Initialize only once
objects_initialized = False
drawing_objects = []
hand_gesture = HandDetector(max_hands=1, detection_confidence=0.7)
hair_segmentation = HairSegmentation()

def initialize_drawings(frame) -> List[DrawingObject]:
  """
  Initialize all DrawingSpec objects once
  based on the actual frame resolution.
  """
  global objects_initialized, drawing_objects
  if not objects_initialized:
    drawing_objects = DrawingObject.distribute_evenly(
        list_drawings,
        frame.shape,
        row="top",
        margin=20,
    )
    objects_initialized = True
    
  for drawing in list_drawings:
    frame = drawing.draw(frame)
    
  return drawing_objects
      
    

def hair_segmentation_color_option_logic(frame):

    # 1. Draw objects first (SAME FRAME SPACE)
    drawing_objects_local = initialize_drawings(frame)

    # 2. Run hand detection ON SAME FRAME
    hand_gesture.set_landmarks_image(frame)
    land_marks = hand_gesture.get_landmarks(frame)

    annotated_image = hand_gesture.draw_landmarks(frame)

    if len(land_marks) == 0:
        return annotated_image

    hand_landmarks = land_marks[0]["landmarks_list"]

    # 3. Finger tip (INDEX FINGER)
    px = int(hand_landmarks[hand_gesture.fingerTips[1]][1])
    py = int(hand_landmarks[hand_gesture.fingerTips[1]][2])

    # Draw finger for debugging (VERY IMPORTANT)
    cv2.circle(annotated_image, (px, py), 10, (0, 0, 255), -1)

    # ----------------------------
    # 4. CHECK EACH OBJECT
    # ----------------------------
    for draw_obj in drawing_objects_local:

        bounds = draw_obj.get_bounds()

        x1 = int(bounds["x1"])
        y1 = int(bounds["y1"])
        x2 = int(bounds["x2"])
        y2 = int(bounds["y2"])

        # convert to (x, y, w, h)
        rect = (x1, y1, x2 - x1, y2 - y1)

        # draw bbox (DEBUG VISUAL)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # print(f"[DEBUG] Finger=({px},{py}) Rect={rect}")

        # HIT TEST
        if hand_gesture.is_finger_point_inside_rect((px, py), rect):
            print("✅ Inside Object")
            ##Get the color of the object that is being hovered over. The code retrieves the color of the drawing object that the user's index finger is currently hovering over. This color information can be used to change the hair color in the hair segmentation application based on the user's interaction with the drawing objects.
            object_color = draw_obj.color
            annotated_image = hair_segmentation.detect(annotated_image, mask_color = object_color)

            
    return annotated_image


# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=True,
        custom_logic=hair_segmentation_color_option_logic,
        window_name="Hair Segmentation Color Option",
        resolution=(650, 500),
        center_window=True,
        draw_fps=True,
        enable_screenshot=True
    )