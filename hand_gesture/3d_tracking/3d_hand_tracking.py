import socket
import json

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template

width, height = 1280, 480

hand_detector = HandDetector(max_hands=1, detection_confidence=0.7)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)


def hand_gesture_tracking(frame):
    h, w, _ = frame.shape

    annotated_image, landmarks = hand_detector.draw_landmarks(frame)
  
    if len(landmarks) > 0:
        landmarks_list, _, _, _ = landmarks[0]
        hand_landmarks = landmarks_list

        # Fresh list per frame (IMPORTANT)
        data = []

        for point in hand_landmarks:
            # Use extend instead of append
            data.extend([
                point[1],        # x
                h - point[2],    # y (flip)
                point[3]         # z
            ])

        # Send only flattened landmarks
        message = json.dumps(data).encode("utf-8")

        # Safety check (UDP limit)
        if len(message) < 60000:
            sock.sendto(message, serverAddressPort)

    return annotated_image


if __name__ == "__main__":
    try:
        video_capture_template(
            video_source=0,
            loop_forever=False,
            custom_logic=hand_gesture_tracking,
            window_name="3D Hand Tracking",
            resolution=(width, height),
            draw_fps=True
        )
    finally:
        sock.close()