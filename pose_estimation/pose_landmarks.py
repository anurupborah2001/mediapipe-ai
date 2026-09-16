from openvisionkit.capture.video_template import video_capture_template

from module.pose_detector import PoseDetector

pose_detector = PoseDetector(output_segmentation_masks=True)

def pose_landmark(frame):
    annotated_frame, pose_landmarks = pose_detector.detect(frame, draw_landmarks=True)
    # list_landmarks = pose_detector.get_all_postion(annotated_frame, pose_landmarks, to_draw_landmarks=False)
    # print("List of landmarks with pixel coordinates and visibility:", list_landmarks)
     

    ## 1. Detecting the left and right shoulder landmarks (11 and 12) and drawing a line between them to visualize the shoulder alignment. The code retrieves the pixel coordinates of the left shoulder (landmark 11) and right shoulder (landmark 12) from the list of landmarks, and then uses OpenCV's line function to draw a line connecting these two points on the annotated image. This can help in analyzing the posture and alignment of the shoulders in the video feed.
    # annotated_frame = pose_detector.draw_landmarks_on_image(annotated_frame, pose_landmarks, [11, 12])
    
    ## 2. Draw the segmentation mask on the annotated image using the draw_segmentation_mask method of the pose_detector object. The method takes the annotated image, the detected pose landmarks, an alpha value for transparency, and a color for the mask as arguments. The resulting image with the segmentation mask applied is returned and can be displayed or further processed.
    ## annotated_frame = pose_detector.draw_segmentation_mask(annotated_frame, pose_landmarks,alpha=0.65, color=(255, 100, 0))
    

    return annotated_frame

if __name__ == "__main__":
    video_capture_template(
      video_source="pose_estimation/videos/video_2.mp4",
      custom_logic=pose_landmark,
      window_name="Post Landmarks", 
      resolution=(640, 720), 
      center_window=True, 
      draw_fps=False
    )