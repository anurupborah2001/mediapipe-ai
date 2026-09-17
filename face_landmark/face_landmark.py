import time

import cv2
from module.face_mesh_detector import FaceMeshDetector ## Make sure to adjust the import path based on your project structure

# Initialize the detector with useful options
detector = FaceMeshDetector(
    model_path='./models/face_landmarker_v2_with_blendshapes.task',  # change if your model is elsewhere
    num_faces=3
)
cap = cv2.VideoCapture(0)   # Change to 0 for webcam
previous_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pass the original BGR frame directly (the class handles RGB conversion internally)
    annotated_frame, faces, blendshapes, transformation_matrices, bboxes = \
        detector.face_mesh_detection(frame, drawLandMarks=True)

    ## Display extra information on the frame
    for i, (face, bbox, blend, matrix) in enumerate(zip(faces, bboxes, blendshapes, transformation_matrices)):
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show face index and some blendshape info (example: smile / eye blink)
        if blend:
            smile_score = blend.get('mouthSmileLeft', 0) + blend.get('mouthSmileRight', 0)
            eye_left = blend.get('eyeBlinkLeft', 0)
            eye_right = blend.get('eyeBlinkRight', 0)
            
            info_text = f"Face {i}: Smile={smile_score:.2f}  Blink=({eye_left:.2f},{eye_right:.2f})"
            cv2.putText(annotated_frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


      ## Display the FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
    previous_time = current_time
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    

    cv2.imshow("Face Mesh + Blendshapes + Bounding Boxes", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()