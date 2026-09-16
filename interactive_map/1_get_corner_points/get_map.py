import numpy as np
import cv2
from openvisionkit.capture.video_template import video_capture_template

resolution = (1920, 1080)
points = np.zeros((4, 2), dtype=np.int32)
counter=0
window_name = "Get Corner Points"
def mouse_points(event, x, y, flags, param):
  """
  Handles mouse click events to select corner points.

  Args:
      event: The type of mouse event (e.g., left button down).
      x: The x-coordinate of the mouse event.
      y: The y-coordinate of the mouse event.
      flags: Any relevant flags passed by OpenCV (not used here).
      param: Additional parameters (not used here).
  This function is called whenever a mouse event occurs in the OpenCV window.
  It records the coordinates of the mouse click in the 'points' array and increments the '
  """
  global counter
  if event == cv2.EVENT_LBUTTONDOWN:
    points[counter] = [x, y]
    counter += 1
    print(f"Point {counter}: ({x}, {y})")
    if counter == 4:
      print("All four points have been selected.")
      cv2.destroyAllWindows()

def get_corner_points_logic(frame):
  ## Draw the selected points on the frame
  for i in range(counter):
    cv2.circle(frame, tuple(points[i]), 5, (0, 255, 0), -1)  
  return frame

if __name__ == "__main__":
  video_capture_template(
    video_source=0,
    loop_forever=False,  
    resolution=resolution,
    mouse_callback=mouse_points,   
    custom_logic=get_corner_points_logic,
    window_name=window_name,
    draw_fps=False
  )
