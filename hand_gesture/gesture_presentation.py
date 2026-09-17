import os
import time
import cv2
import numpy as np

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
from util import utility


width, height = 1280, 720

path, _ = utility.get_calling_folder()

overlay_drag = {
    "dragging": False,
    "x": 10,
    "y": 10,
    "offset_x": 0,
    "offset_y": 0,
}

gestureThresholdLine = 300

presentation_slides = f"{path}/images/presentation/"
list_slides = sorted(os.listdir(presentation_slides))
number_of_slides = len(list_slides)

slide_index = 0

scale = 1.0
width_small, height_small = int(300 * scale), int(250 * scale)

hand_detector = HandDetector(max_hands=1, detection_confidence=0.6)

last_slide_change_time = 0
SLIDE_DELAY = 1.0

# Each inner list is one continuous annotation stroke
annotations = [[]]

prev_point = None
smooth_point = None

SMOOTHING = 0.35
MIN_DRAW_DISTANCE = 4
DRAW_THICKNESS = 8
POINTER_RADIUS = 12


def smooth_position(prev, current, alpha=0.35):
    if prev is None:
        return current

    x = int(prev[0] * (1 - alpha) + current[0] * alpha)
    y = int(prev[1] * (1 - alpha) + current[1] * alpha)

    return (x, y)


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def draw_smooth_stroke(img, stroke, color=(0, 0, 255), thickness=8):
    if len(stroke) < 2:
        return

    for i in range(1, len(stroke)):
        cv2.line(
            img,
            stroke[i - 1],
            stroke[i],
            color,
            thickness,
            cv2.LINE_AA,
        )


def redraw_annotations(img):
    for stroke in annotations:
        draw_smooth_stroke(
            img,
            stroke,
            color=(0, 0, 255),
            thickness=DRAW_THICKNESS,
        )


def start_new_stroke_if_needed():
    global annotations

    if annotations and len(annotations[-1]) > 0:
        annotations.append([])

    if len(annotations) == 0:
        annotations.append([])


def undo_last_stroke():
    global annotations

    if not annotations:
        annotations.append([])
        return

    if len(annotations[-1]) == 0:
        annotations.pop()

    if annotations:
        annotations.pop()

    if len(annotations) == 0:
        annotations.append([])


def reset_annotation_state():
    global prev_point, smooth_point

    prev_point = None
    smooth_point = None


def clear_annotations():
    global annotations

    annotations = [[]]
    reset_annotation_state()


def gesture_presentaion_logic(frame):
    global slide_index
    global overlay_drag
    global last_slide_change_time
    global prev_point
    global smooth_point
    global annotations

    first_slide_page = list_slides[slide_index]
    pptImage = cv2.imread(f"{presentation_slides}/{first_slide_page}")

    if pptImage is None:
        raise FileNotFoundError(f"Could not read slide image: {first_slide_page}")

    annotated_img, landmark_result = hand_detector.draw_landmarks(
        frame,
        to_draw_center_point=False,
        to_draw_bounding_box=False,
    )

    cv2.line(
        annotated_img,
        (0, gestureThresholdLine),
        (annotated_img.shape[1], gestureThresholdLine),
        (241, 246, 241),
        2,
    )

    current_time = time.time()

    # Redraw old strokes first
    redraw_annotations(pptImage)

    if len(landmark_result) > 0:
   
        landmarks, _, landmark_params, _ = landmark_result[0]
        center_point_x, center_point_y = landmark_params["center_point"]

        finger_up_arr = hand_detector.fingers_up(landmarks)

        index_finger_x, index_finger_y = landmarks[hand_detector.fingerTips[1]][1:3]

        index_finger_x = int(
            np.interp(index_finger_x, [width // 2, width], [0, width])
        )
        index_finger_y = int(
            np.interp(index_finger_y, [150, height - 150], [0, height])
        )

        index_finger_x = max(0, min(index_finger_x, pptImage.shape[1] - 1))
        index_finger_y = max(0, min(index_finger_y, pptImage.shape[0] - 1))

        raw_point = (int(index_finger_x), int(index_finger_y))

        # Slide navigation only above threshold line
        if center_point_y <= gestureThresholdLine:
            # Previous slide: only thumb up
            if finger_up_arr == [1, 0, 0, 0, 0]:
                if slide_index > 0 and current_time - last_slide_change_time > SLIDE_DELAY:
                    slide_index -= 1
                    clear_annotations()
                    last_slide_change_time = current_time

            # Next slide: only pinky up
            elif finger_up_arr == [0, 0, 0, 0, 1]:
                if current_time - last_slide_change_time > SLIDE_DELAY:
                    slide_index += 1

                    if slide_index >= number_of_slides:
                        slide_index = 0

                    clear_annotations()
                    last_slide_change_time = current_time

        ## Pointer mode: index + middle fingers up (thumb, ring, pinky down)
        if finger_up_arr == [0, 1, 1, 0, 0]:
            reset_annotation_state()
            start_new_stroke_if_needed()

            cv2.circle(
                pptImage,
                raw_point,
                POINTER_RADIUS,
                (0, 0, 255),
                cv2.FILLED,
                cv2.LINE_AA,
            )

        ## Writing mode: only index finger up (thumb, middle, ring, pinky down)
        elif finger_up_arr == [0, 1, 0, 0, 0]:
            current_point = smooth_position(
                smooth_point,
                raw_point,
                SMOOTHING,
            )

            smooth_point = current_point

            ## Live visible pointer while writing and drawing line from previous point to current point
            cv2.circle(
                pptImage,
                current_point,
                POINTER_RADIUS,
                (0, 0, 255),
                cv2.FILLED,
                cv2.LINE_AA,
            )

            if len(annotations) == 0:
                annotations.append([])

            if (
                prev_point is None
                or distance(prev_point, current_point) > MIN_DRAW_DISTANCE
            ):
                annotations[-1].append(current_point)

                if prev_point is not None:
                    cv2.line(
                        pptImage,
                        prev_point,
                        current_point,
                        (0, 0, 255),
                        DRAW_THICKNESS,
                        cv2.LINE_AA,
                    )

                prev_point = current_point

        # Undo last stroke: index + middle + ring up
        elif finger_up_arr == [0, 1, 1, 1, 0]:
            undo_last_stroke()
            reset_annotation_state()

        else:
            reset_annotation_state()
            start_new_stroke_if_needed()

    camera_image = cv2.resize(annotated_img, (width_small, height_small))

    pptImage = utility.overlay_frame(
        pptImage,
        camera_image,
        position="custom",
        padding=10,
        draggable=True,
        drag_position=overlay_drag,
    )

    return pptImage


if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=False,
        custom_logic=gesture_presentaion_logic,
        window_name="Gesture Presentation",
        show_window=True,
        resolution=(width, height),
        draw_fps=False,
        enable_screenshot=True,
        mouse_callback=utility.mouse_drag_overlay,
        mouse_callback_params={
            "drag_position": overlay_drag,
            "overlay_w": width_small,
            "overlay_h": height_small,
        },
    )