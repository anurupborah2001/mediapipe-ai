
import cv2
import time
import numpy as np
from openvisionkit.capture.video_template import video_capture_template

from module.pose_detector import PoseDetector

pose_detector = PoseDetector()


def draw_workout_bar(image, percent, count):
    h, w = image.shape[:2]

    # Convert percent → bar position
    bar = np.interp(percent, [0, 100], [h - 50, 100])

    # Draw bar background
    cv2.rectangle(image, (w - 80, 100), (w - 40, h - 50), (255,255,255), 2)

    # Draw filled bar
    cv2.rectangle(image, (w - 80, int(bar)), (w - 40, h - 50), (0,255,0), cv2.FILLED)

    # Percentage text
    cv2.putText(image, f'{int(percent)} %', (w - 120, 80),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    # Rep count
    cv2.putText(image, f'Count: {count}', (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    return image


def ai_personal_trainer(frame):
    annotated_frame, pose_landmarks = pose_detector.detect(frame, draw_landmarks=False)
    
    ## Calculate the angle at the left elbow (landmarks 11, 13, 15) and draw the angle value on the annotated frame. The calculate_angle method of the pose_detector object is called with the annotated frame, detected pose landmarks, and the landmark IDs for the left shoulder (11), left elbow (13), and left wrist (15). The method calculates the angle at the left elbow joint and returns the annotated frame with the angle drawn on it, as well as the calculated angle value.
    annotated_frame, _ = pose_detector.calculate_angle(image=annotated_frame, detection_result=pose_landmarks,p1=11, p2=13, p3=15, to_draw_landmarks=True)
    
    ## Calculate workout percentage and rep count based on the angle, and draw the workout bar on the annotated frame
    # angle, min_angle, max_angle, percent, rep_count = pose_detector.calculate_workout_percentage()
    angle, percent, rep_count = pose_detector.calculate_workout_percentage()
    
    print(f"Angle: {angle}, Percent: {percent}, Reps: {rep_count}")
    
    annotated_frame = draw_workout_bar(
        annotated_frame,
        percent,
        rep_count
    )
    
    exercise = pose_detector.detect_exercise(annotated_frame, pose_landmarks)
    print(f"Detected exercise: {exercise}")
    
    pose_detector.get_workout_stats(annotated_frame)

    
    # print(f"Angle: {angle}, Percent: {percent}, Reps: {rep_count}, Min Angle: {min_angle}, Max Angle: {max_angle}")
    ## Draw the workout bar on the annotated frame using the draw_workout_bar method of the pose_detector object. The method takes the annotated frame, calculated percentage, rep count, and current angle as arguments, and returns the annotated frame with the workout bar drawn on it.
    # annotated_frame = draw_workout_bar(image=annotated_frame, percent=percent, rep_count=rep_count, angle= angle, min_angle=min_angle, max_angle=max_angle)
    return annotated_frame

if __name__ == "__main__":
    video_capture_template(
      video_source="pose_estimation/videos/gym_video_2.mp4",
      loop_forever=False,
      custom_logic=ai_personal_trainer,
      window_name="AI Personal Trainer", 
      resolution=(640, 720), 
      center_window=True, 
      draw_fps=False
    )
    
