from module.image_detector import ImageDetector
from openvisionkit.capture.image_template import image_template
from openvisionkit.capture.video_template import video_capture_template
from util import utility
import cv2
import numpy as np

path, _= utility.get_calling_folder()
detect_type = "IMAGE"

def object_distance_measurement(image,reference_width_cm=15,min_area=200):
    # Placeholder for object distance measurement logic
    # You can implement your distance measurement algorithm here using the input image
    # For demonstration, we will just return the original image
    # we neeed to get the contours of the object and then we can use the contours to measure the distance of the object from the camera using the known size of the object and the focal length of the camera
    # image_detector = ImageDetector(image)  ## We are setting preprocess to true because we want to use the preprocessed image for contour detection and distance measurement

    image_detector = ImageDetector(image)  ## We are setting preprocess to true because we want to use the preprocessed image for contour detection and distance measurement
    img= image.copy()
  
  
    contours_results, img_contours = image_detector.find_contours(img, min_area=min_area, get_canny_edges=True, debug=True, sort_countours_largest_to_smallest=True, sort_bbox_smaller_to_largest=True) ## This will give the sorted contours co ordinates based on the position in the image from top to bottom and left to right
    contours = [c["contour"] for c in contours_results]
    bbox = [c["bounding_box"] for c in contours_results]
    print("Number of contours detected:", len(contours))
    if len(contours_results) != 0:
      
      ## STEP 1: Use largest contour as reference
      biggest_contour = contours_results[0]
      ##  cv2.minAreaRect returns a Box2D structure which contains following details - ( center (x,y), (width, height), angle of rotation ).
      ref_rect = cv2.minAreaRect(biggest_contour["contour"])
      ## (width, height)
      ref_w, ref_h = ref_rect[1]
      ref_pixels = max(ref_w, ref_h)
      ## Calculate pixels per cm using the reference object size and the known real-world width of the reference object
      pixels_per_cm = ref_pixels / reference_width_cm 
      print(f"Reference pixels: {ref_pixels}, pixels/cm: {pixels_per_cm}")
      
      # STEP 2: Measure all contours
      for cnt in contours:
          #https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
          rect = cv2.minAreaRect(cnt)
          (cx, cy), (w, h), angle = rect

          if w == 0 or h == 0:
              continue
          # Get rotated box
          box = cv2.boxPoints(rect)
          box = np.int32(box)
          # cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

          # Draw contour box
          cv2.polylines(img, [box], True, (0, 255, 0), 2, cv2.LINE_AA)

          # Order points consistently (important!)
          # top-left, top-right, bottom-right, bottom-left
          box = sorted(box, key=lambda x: (x[1], x[0]))
          top_points = sorted(box[:2], key=lambda x: x[0])
          bottom_points = sorted(box[2:], key=lambda x: x[0])
          print(f"Top points: {top_points}, Bottom points: {bottom_points}")
          (tl, tr) = top_points
          (bl, br) = bottom_points
          print(tuple(tl), tuple(tr), tuple(bl), tuple(br))
          # -------- WIDTH (top edge) --------
          cv2.line(img, tuple(tl), tuple(tr), (255, 0, 0), 2, cv2.LINE_AA)

          width_px = np.linalg.norm(np.array(tr) - np.array(tl))
          ## Calculate width in cm using the pixels per cm ratio derived from the reference object
          width_cm = width_px / pixels_per_cm

          # midpoint for label
          mid_top = ((tl[0] + tr[0]) // 2, (tl[1] + tr[1]) // 2)

          # -------- HEIGHT (left edge) --------
          cv2.line(img, tuple(tl), tuple(bl), (255, 0, 0), 2, cv2.LINE_AA)

          height_px = np.linalg.norm(np.array(bl) - np.array(tl))
          height_cm = height_px / pixels_per_cm

          # midpoint for label
          mid_left = ((tl[0] + bl[0]) // 2, (tl[1] + bl[1]) // 2)

          # -------- LABELS --------
          def draw_label(img, text, position):
              font = cv2.FONT_HERSHEY_SIMPLEX
              scale = 0.5
              thickness = 1

              (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

              x, y = int(position[0]), int(position[1])

              # background box
              cv2.rectangle(
                  img,
                  (x - tw // 2 - 3, y - th - 3),
                  (x + tw // 2 + 3, y + 3),
                  (0, 0, 0),
                  -1
              )

              # text
              cv2.putText(
                  img,
                  text,
                  (x - tw // 2, y),
                  font,
                  scale,
                  (0, 255, 255),
                  thickness,
                  cv2.LINE_AA
              )

          draw_label(img, f"{width_cm:.1f} cm", mid_top)
          draw_label(img, f"{height_cm:.1f} cm", mid_left)

    return img
  
if __name__ == "__main__":
    if detect_type == "IMAGE":
      image_template(
        image_path=f"{path}/assets/images/distance_measurement/image_distance2.jpeg",
        custom_logic=object_distance_measurement,
        window_name="Object Distance Measurement - Image",
        show_window=True,  # We will show the window in the object_distance_measuremen function
        center_window=True,
        resolution=(640, 500)
      )
    else:
      video_capture_template(
        video_source=0,
        custom_logic=object_distance_measurement,
        window_name="Object Distance Measurement - Video",
        show_window=True  # We will show the window in the object_distance_measuremen function
      )
