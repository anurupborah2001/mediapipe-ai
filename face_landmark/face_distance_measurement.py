from module.face_mesh_detector import FaceMeshDetector
import numpy as np
import cv2
from openvisionkit.capture.video_template import video_capture_template
from util import utility

face_mesh_detector = FaceMeshDetector(num_faces=1, min_face_detection_confidence=0.8)
text_to_display = "Learning OpenCV is a journey that often begins with intimidating, \n low-level pointer arithmetic and complex documentation,\n  but it rewards persistent effort with the ability to build sophisticated,\n real-time vision applications"
sensitivity_of_scale = 20
def calculate_focal_length(known_distance, real_width, pixel_width):
    """
    Calculate the focal length of the camera using the known distance, real width, and pixel width.
    Args:
        known_distance (float): The known distance from the camera to the object (in cm).
        real_width (float): The real width of the object (in cm).
        pixel_width (float): The width of the object in pixels as detected by the camera.
    Returns:
        float: The calculated focal length of the camera.
    """
    if real_width == 0:
        return 0
    return (pixel_width * known_distance) / real_width

def face_distance_measurement_logic(frame):
    # Placeholder for face distance measurement logic
    img_palate = np.zeros_like(frame)
    
    annotated_img, faces, _, _, _  = face_mesh_detector.face_mesh_detection(frame, drawLandMarks=False)
    if faces:
      face = faces[0]  # Assuming we're only interested in the first detected face
      known_distance = 50  # Known distance from the camera to the face in cm (adjust as needed)
      pixel_width, _  = face_mesh_detector.distance_between_landmarks(face[face_mesh_detector.LEFT_IRIS_CENTER], face[face_mesh_detector.RIGHT_IRIS_CENTER])  # Real width of the face in cm (adjust as needed)
      real_length = 6.3 # Average width between the iris centers in cm (adjust as needed) in cm
      # focal_length = calculate_focal_length(known_distance, real_length, pixel_width)
      focal_length = 860
      distance = (real_length * focal_length) / pixel_width if pixel_width != 0 else 0
    
      forehead_x, forehead_y = face[face_mesh_detector.FOREHEAD_CENTER]
      utility.put_text_rect(annotated_img, f"Distance: {distance:.2f} cm", (forehead_x - 80, forehead_y - 100), scale=2, thickness=2, colorT=(0,255,0))
      
      ## Draw text in the palate
      scale = 0.8 + (int(distance/sensitivity_of_scale)*sensitivity_of_scale)/50
      print("Distance:", distance, "Scale:", scale)
      utility.draw_wrapped_text(img_palate, text_to_display, (30, 30), cv2.FONT_HERSHEY_PLAIN, scale, (255,255,255), 2, max_width=40)
      
    stack_img = utility.stack_images_grid([annotated_img, img_palate], cols=2,scale=1)
    return stack_img

if __name__ == "__main__":
  video_capture_template(
        video_source=0,
        loop_forever=False,      
        custom_logic=face_distance_measurement_logic,
        window_name="Face Distance Measurement",
        resolution=(680, 320),
        draw_fps=False
    )