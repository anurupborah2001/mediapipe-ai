import time

import cv2

from module.face_mesh_detector import FaceMeshDetector

detector = FaceMeshDetector(
    model_path='./models/face_landmarker_v2_with_blendshapes.task',
    num_faces=2
)

cap = cv2.VideoCapture(0)   # Change to 0 for webcam
previous_time = 0
glasses = None  # Path to your AR filter image (e.g., glasses.png)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get all extended data from the detector
    annotated_frame, faces, blendshapes, transformation_matrices, bboxes = \
        detector.face_mesh_detection(frame, drawLandMarks=True)

    # ====================== NEW FEATURES USING FACE LANDMARKS ======================
    num_faces_detected = len(faces)
    
    for i, (face, bbox, blend, matrix) in enumerate(zip(faces, bboxes, blendshapes, transformation_matrices)):
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        mouth_ratio   = detector.get_mouth_openness_ratio(face)
        left_gaze     = detector.get_eye_gaze_direction(face, is_left_eye=True)
        right_gaze    = detector.get_eye_gaze_direction(face, is_left_eye=False)
        ipd_pixels    = detector.get_inter_pupillary_distance(face)
        ipd_norm      = detector.get_inter_pupillary_distance(face, normalized=True)
            
        # === HEAD POSE (Yaw, Pitch, Roll) ===
        yaw, pitch, roll = detector.get_head_pose_angles(matrix)
        
        # === EMOTION DETECTION ===
        emotion = detector.get_emotion(blend)

        ## AR Filter 
        # annotated_frame = detector.overlay_ar_filter(annotated_frame, face, glasses, filter_type="glasses")

        # === BLENDSHAPE INFO (smile + blink) ===
        if blend:
            smile_score = blend.get('mouthSmileLeft', 0) + blend.get('mouthSmileRight', 0)
            eye_left = blend.get('eyeBlinkLeft', 0)
            eye_right = blend.get('eyeBlinkRight', 0)
        else:
            smile_score = eye_left = eye_right = 0.0

        # Build rich info text (multi-line)
        info_text = [
            f"Face {i} | {emotion}",
            f"L Eye: {left_gaze}   R Eye: {right_gaze}",
            f"IPD: {ipd_pixels:.0f}px ({ipd_norm:.2f})"
            f"Smile: {smile_score:.2f}  Blink: L{eye_left:.2f} R{eye_right:.2f}",
            f"Yaw: {yaw:+.0f}°  Pitch: {pitch:+.0f}°  Roll: {roll:+.0f}°"
        ]

        # Draw each line
        for line_idx, text in enumerate(info_text):
            y_pos = y1 - 10 - (line_idx * 28)
            cv2.putText(annotated_frame, text, (x1, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # ====================== FPS + TOTAL FACES ======================
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
    previous_time = current_time

    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    cv2.putText(annotated_frame, f'Faces: {num_faces_detected}', (20, 100),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    cv2.imshow("Face Mesh + Advanced Features (Pose + Emotion)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()