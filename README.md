# MediaPipe AI — Computer Vision Project Collection

A collection of real-time computer vision demos built on **MediaPipe Tasks**, **OpenCV**, and **Tesseract OCR**.
Each top-level folder is a self-contained project (hand gestures, face landmarks, pose estimation, segmentation,
OCR/form extraction, and more). All projects share these reusable layers:

| Layer | Purpose |
| --- | --- |
| [`module/`](#module--detector-libraries) | Detector/estimator libraries that wrap MediaPipe, OpenCV, and Tesseract |
| [`util/`](#util--drawing--geometry-helpers) | Drawing, geometry, layout, and image helpers |
| [`openvisionkit.capture`](#openvisionkitcapture--application-scaffolding) | Video/image loop scaffolding, recording, screenshots — external `openvisionkit` package |
| [`template/`](#template--superseded-local-scaffolding) | Superseded local copy of that scaffolding; still provides `DrawingObject` |
| [`models/`](#models--model-files) | `.task` / `.tflite` / frozen graph model files used by `module/` |

---

## Table of Contents

- [Getting Started](#getting-started)
- [Detection Types](#detection-types)
- [Architecture](#architecture)
- [Shared Libraries](#shared-libraries)
  - [`module/` — Detector Libraries](#module--detector-libraries)
  - [`util/` — Drawing & Geometry Helpers](#util--drawing--geometry-helpers)
  - [`openvisionkit.capture` — Application Scaffolding](#openvisionkitcapture--application-scaffolding)
  - [`template/` — Superseded Local Scaffolding](#template--superseded-local-scaffolding)
  - [`models/` — Model Files](#models--model-files)
- [Projects](#projects)
  - [hand_gesture](#hand_gesture)
  - [face_detection](#face_detection)
  - [face_landmark](#face_landmark)
  - [pose_estimation](#pose_estimation)
  - [hair_segmentation](#hair_segmentation)
  - [selfie_segmentation](#selfie_segmentation)
  - [object_detection](#object_detection)
  - [image_detection](#image_detection)
  - [text_detection](#text_detection)
  - [car_parking_slot](#car_parking_slot)
  - [interactive_map](#interactive_map)
- [Image Gallery Index](#image-gallery-index)
- [Notes](#notes)

---

## Getting Started

### Requirements

Python 3.12+. Dependencies are managed with [uv](https://docs.astral.sh/uv/) — `pyproject.toml` and
`uv.lock` are committed, so one command installs everything into `.venv`:

```bash
uv sync
```

Optional extras, per project:

```bash
uv add tensorflow            # hand_gesture/hand_sign_detection (Teachable Machine .h5)
uv add spacy && uv run python -m spacy download en_core_web_sm   # text_detection NLP features
uv add pycaw comtypes        # volume control on Windows only
```

`mediapipe` is intentionally pinned to `>=0.10,<1.0`; on mediapipe 1.0.1 the MediaPipe Tasks vision
graphs abort at construction on macOS (`Check failed: service_ Service is unavailable.`).

Tesseract OCR must be installed natively for every OCR-based project
(`text_detection`, `image_detection`, `module/form_*`):

```bash
brew install tesseract            # macOS
```

### Running a project

Every script imports the shared packages with absolute paths (`from module...`, `from util...`) and
every detector loads its model from `./models/...`, so **run scripts from the repository root** so
that Python and the model paths both resolve:

```bash
cd /path/to/mediapipe-ai

uv run python -m hand_gesture.finger_counter
uv run python -m face_landmark.blink_counter
uv run python -m pose_estimation.ai_personal_trainer
```

Common runtime controls provided by `openvisionkit.capture.video_template`:

| Key | Action |
| --- | --- |
| `Esc` | Quit the demo |
| `s` | Save a screenshot to `screenshots/` (when `enable_screenshot=True`) |
| `r` | Start/stop recording to `recordings/` (when `enable_manual_recording=True`) |
| `p` | Pause/resume recording |

---

## Detection Types

Eleven detection and estimation families are demonstrated in this repository. Nine of them are
**MediaPipe Tasks** graphs; the remaining two are classic OpenCV pipelines (DNN and thresholding)
kept for comparison.

| # | Detection type | Backend | Model | Wrapper | Projects |
| --- | --- | --- | --- | --- | --- |
| 1 | **Hand Landmarker** — 21 hand landmarks, handedness, world coordinates | MediaPipe Tasks | `hand_landmarker.task`, `gesture_recognizer.task` | `module/hand_detector.py` | [`hand_gesture`](#hand_gesture) (18 demos) |
| 2 | **Face Detector** — face bounding boxes + 6 keypoints (BlazeFace) | MediaPipe Tasks | `face_detector.tflite`, `blaze_face_full_range.tflite` | `module/face_detector.py` | [`face_detection`](#face_detection) |
| 3 | **Face Landmarker** — 478 face-mesh landmarks, 52 blendshapes, 4×4 transform matrix | MediaPipe Tasks | `face_landmarker_v2_with_blendshapes.task` | `module/face_mesh_detector.py` | [`face_landmark`](#face_landmark) |
| 4 | **Pose Landmarker** — 33 body landmarks in pixel + world space, segmentation mask | MediaPipe Tasks | `pose_landmarker.task` | `module/pose_detector.py` | [`pose_estimation`](#pose_estimation) |
| 5 | **Object Detector** — multi-class bounding boxes with labels and scores | MediaPipe Tasks | `efficientdet_lite.tflite`, `ssd_mobilenet_v2_coco.tflite` | `module/object_detector.py` | [`object_detection`](#object_detection) |
| 6 | **Image Segmenter (selfie)** — person/background mask for virtual backgrounds | MediaPipe Tasks | `deeplab_v3.tflite`, `deeplabv3_plus_mobilenet.tflite`, `interactive_segmenter.tflite` | `module/selfie_segmentation.py` | [`selfie_segmentation`](#selfie_segmentation) |
| 7 | **Image Segmenter (hair)** — hair mask for recolouring | MediaPipe Tasks | `hair_segmenter.tflite` | `module/hair_segmentation.py` | [`hair_segmentation`](#hair_segmentation) |
| 8 | **Gesture / sign classification** — Teachable Machine Keras classifier over cropped hand ROIs | Keras (on top of Hand Landmarker) | user-trained `.h5` + labels | `module/classifier.py` | [`hand_gesture`](#hand_gesture) → `hand_sign_detection/` |
| 9 | **Text / OCR detection** — characters, words, digits, tables, layout, form ROIs | Tesseract + OpenCV | Tesseract native install | `module/text_detector.py`, `module/form_detector.py`, `module/form_roi_detector.py` | [`text_detection`](#text_detection), [`image_detection`](#image_detection) |
| 10 | **Classic OpenCV DNN object detection** — SSD MobileNet v3 over the COCO classes | OpenCV DNN | `frozen_inference_graph.pb` + `.pbtxt` + `coco.names` | *(inline in the script)* | [`object_detection`](#object_detection) → `coco_object_detection.py` |
| 11 | **Classic OpenCV occupancy / measurement** — thresholding, contours, ORB, perspective warp | OpenCV only | none | `module/image_detector.py` | [`car_parking_slot`](#car_parking_slot), [`image_detection`](#image_detection), [`interactive_map`](#interactive_map) |

Landmark index charts for the MediaPipe tasks are in [`models/`](#models--model-files).

---

## Architecture

### Layer dependency direction

Dependencies point in one direction only — project scripts import the shared layers, and no shared
layer imports a project.

```
<project>/*.py  --imports-->  openvisionkit.capture   (external package: window loop, FPS,
       |                                                recording, screenshots, key handling)
       |
       +---------imports-->  module/     (detectors; third-party libraries only)
       |
       +---------imports-->  util/       (drawing/geometry helpers; no internal imports)
       |
       +---------imports-->  models/     (loaded by path, not imported)
```

| Layer | Rule |
| --- | --- |
| `module/` | Detector and estimator classes. Depend on third-party libraries only — never on `util/`, `template/`, or a project folder. |
| `util/utility.py` | Flat function library (drawing, geometry, HSV masks, layout, path helpers). No internal dependencies. Imported as `from util import utility`. |
| `openvisionkit.capture` | Application scaffolding, installed as the external **`openvisionkit`** package. Fixes belong in that project, not here. |
| `template/` | Superseded local copy of the scaffolding, kept for reference. Only `template/draw_object.py` is still imported by a project (`hair_segmentation/hair_segmentation_color_option.py`). |
| `<project>/` | Thin scripts wiring a detector to a per-frame callback. Anything reusable belongs in `module/` or `util/`. |

### The canonical demo pattern

Nearly every one of the 69 Python files follows this shape:

```python
from openvisionkit.capture.video_template import video_capture_template
from module.hand_detector import HandDetector
from util import utility

path, _ = utility.get_calling_folder()      # this script's own folder, for assets
hand_detector = HandDetector()              # constructed ONCE at import time

def my_logic(frame):                        # called per frame, returns the frame to display
    annotated_image, hands = hand_detector.draw_landmarks(frame)
    ...
    return annotated_image

if __name__ == "__main__":
    video_capture_template(
        video_source=0,                     # int = camera index, str = video file path
        custom_logic=my_logic,
        window_name="My Demo",
        resolution=(1280, 720),
    )
```

Detectors are constructed at module level, not inside `custom_logic` — building one loads a
MediaPipe model, which is far too slow to repeat per frame.

### Two path conventions

| What | Resolved against | How |
| --- | --- | --- |
| Model files (`./models/...`) | The current working directory | Detector default arguments — hence "run from the repository root" |
| Project assets (`images/`, `assets/`, `videos/`, `audio/`) | The script's own folder | `path, _ = utility.get_calling_folder()` then an f-string |

### Running modes and state

Detectors take `running_mode="IMAGE"` or `"VIDEO"`; in VIDEO mode MediaPipe needs a monotonically
increasing millisecond timestamp (`PoseDetector` generates one, `FaceDetector` and `HandDetector`
expect the caller to pass `timestamp_ms`). Frames are passed as plain **BGR** — every detector
converts to RGB internally.

Cross-frame state lives in a plain dict handed to `video_capture_template(state=...)`; keyboard
actions are registered on a `KeyEventManager` and passed as `key_manager=`. See
`hand_gesture/kids_game/ping_pong_game.py` for the reference example.

For contributor-facing detail — detector return shapes, known quirks, platform notes — see
[`CLAUDE.md`](CLAUDE.md).

---

## Shared Libraries

### `module/` — Detector Libraries

Reusable detection and estimation classes. Projects import these instead of talking to MediaPipe directly.

| File | Description |
| --- | --- |
| `module/hand_detector.py` | `HandDetector` — the largest and most-used library in the repo. Wraps the MediaPipe **Hand Landmarker** task and adds: landmark extraction with pixel coordinates, bounding boxes and handedness labels, landmark drawing, `fingers_up()` state, pinch/join detection (`is_fingers_joined`, `finger_joined`, `joined_fingers`), distance measurement between landmarks, palm-width based **camera distance estimation** (focal-length calibration + polynomial fitting), point-in-rectangle hit testing, and gesture shortcuts (`is_fist`, `is_open_hand`, `is_thumbs_up`, `is_peace_sign`). |
| `module/face_detector.py` | `FaceDetector` — MediaPipe **Face Detector** task. Detects faces in image/video/live-stream modes, parses bounding boxes and keypoints, draws detections, filters by confidence, returns the largest face, crops faces with margin, sorts faces, and computes IoU between boxes. |
| `module/face_mesh_detector.py` | `FaceMeshDetector` — MediaPipe **Face Landmarker** (478 landmarks + blendshapes + transform matrix). Provides eye-aspect-ratio/blink inputs, mouth-openness ratio, iris centers and gaze direction, inter-pupillary distance, head-pose angles (yaw/pitch/roll) from the transform matrix, blendshape-driven emotion estimation, AR filter overlay (e.g. glasses), and landmark drawing/measurement helpers. |
| `module/pose_detector.py` | `PoseDetector` — MediaPipe **Pose Landmarker**. Returns 33 landmarks in pixel and world space, computes joint angles, derived neck landmark and neck-to-shoulder distance, distance measurements between landmarks, exercise/rep detection with workout percentage, active-arm selection, segmentation-mask drawing, and landmark rendering. |
| `module/object_detector.py` | `ObjectDetector` — MediaPipe **EfficientDet-Lite** object detection with bounding-box/label visualisation, for image and video running modes. |
| `module/selfie_segmentation.py` | `SelfieSegmentation` — MediaPipe **Image Segmenter** for people. Background removal, blur, solid colour, image replacement, foreground extraction, alpha/confidence blending, mask overlay and thresholding, morphological cleanup, layered backgrounds, plus downscaled `fast_*` variants and an optimised virtual-background path. |
| `module/hair_segmentation.py` | `HairSegmentation` — MediaPipe **hair segmenter** (`hair_segmenter.tflite`). Produces a smoothed hair mask and recolours hair with any BGR colour. |
| `module/text_detector.py` | `TextDetector` — the OCR/NLP workhorse. Tesseract-based character/word/digit detection with confidence, preprocessing, language switching, OSD, hOCR/PDF/ALTO export, table detection, layout analysis, deskewing, auto-Canny, rotation/resize (via `imutils`), cursive-text extraction, noisy-image OCR, ORB feature matching for form alignment (`compare_matches_bf_matcher`, `compare_matches_knn_matcher`), SSIM fallback comparison, and optional **spaCy** NLP (entities, keywords, relations, summarisation). |
| `module/image_detector.py` | `ImageDetector` — OpenCV image analysis. Highlighted-text detection via HSV masks, dominant HSV colour clustering, mask refinement, Canny edges, contour finding/sorting, reference-object detection and real-world measurement export, grid drawing, text-region extraction, and ORB/SSIM image comparison. |
| `module/form_detector.py` | `FormROIDetector` (compact version) — detects form field ROIs from table/cell structure, groups them into rows, extracts key/value pairs via OCR, detects checkbox/radio selection state, and visualises results. |
| `module/form_roi_detector.py` | `FormROIDetector` (enhanced version) — a richer form-field detector built around the `ROIRegion` dataclass. Detects tables, checkboxes, radio buttons, general text fields, dropdowns and signature areas; classifies field types; assigns labels by OCR proximity; detects fill state; deduplicates by IoU; groups rows and extracts key/value pairs. |
| `module/form_roi_annotator.py` | `FormROIAnnotator` — an interactive OpenCV tool to hand-label form ROIs. Click two diagonal corners, pick a field type and label, then undo/delete/edit/clear/list via keyboard shortcuts and save to JSON (`annotated_rois_latest.json` plus timestamped copies). |
| `module/classifier.py` | `Classifier` — loads a **Teachable Machine** `.h5` Keras model plus its labels file and returns predictions with an optional on-frame label. Used by the hand-sign project. |
| `module/fps_counter.py` | `FPSCounter` — computes frames-per-second from inter-frame time and overlays it on the frame. Superseded by the `FPSCounter` bundled in `openvisionkit.capture.video_template`. |
| `module/live_plot.py` | `LivePlot` — real-time scrolling line plot rendered as an OpenCV image (derived from cvzone's `PlotModule`). Used for blink/EAR graphs. |
| `module/image_hsv_detector.py` | Standalone HSV colour-picker tool: manual RGB→HSV conversion, trackbar-driven range tuning, and a mouse callback that prints the HSV value under the cursor. Useful for calibrating the highlight/mask detectors. |

### `util/` — Drawing & Geometry Helpers

`util/utility.py` is a flat function library imported almost everywhere as `from util import utility`.

| Function | Description |
| --- | --- |
| `rectangle_corners` | Draws a rounded/cornered bounding box (the "L-corner" style used across demos). |
| `put_text_rect`, `put_text_think_corners`, `draw_wrapped_text` | Text with filled background rectangle, corner-styled labels, and word-wrapped multi-line text. |
| `draw_rounded_rect`, `highlight_image`, `zoom_image` | Rounded rectangles, selection highlighting, and image zoom. |
| `detect_highlighted_text`, `detect_single_highlighted_text`, `get_dominant_hsv_colors`, `refine_mask`, `find_contours` | Standalone HSV-highlight and contour pipeline (the functional twin of `ImageDetector`). |
| `load_image`, `get_valid_images`, `resize_with_padding`, `overlay_transparent` | Image loading with alpha, folder scanning, aspect-preserving resize with padding, and alpha-aware overlay. |
| `is_hovering`, `create_centered_grid_buttons`, `move_image`, `mouse_drag_overlay` | Hover hit-testing, auto-laid-out button grids, and image drag/move logic for gesture and mouse interaction. |
| `stack_images_grid`, `auto_layout`, `overlay_frame` | Multi-image grid stacking with labels, automatic panel layout, and picture-in-picture overlay. |
| `get_currect_path`, `get_calling_folder`, `find_project_root` | Path resolution so scripts can locate their own `assets/`, `images/` and `models/` folders regardless of the working directory. |

### `openvisionkit.capture` — Application Scaffolding

| File | Description |
| --- | --- |
| `openvisionkit.capture.video_template` | `video_capture_template(...)` — **the entry point for nearly every project**. Opens a webcam index or a video file, runs your `custom_logic(frame)` callback per frame, and handles FPS overlay, window centring and sizing, `Esc` to quit, looping video files, a shared mutable `state` dict, mouse callbacks, screenshots (manual and timed), and video/GIF recording. Also defines `KeyEventManager` for registering per-key handlers and `save_screenshot()`. |
| `openvisionkit.capture.image_template` | `image_template(...)` — the still-image counterpart: loads an image, applies your `custom_logic(image)` callback, and displays it in a centred window. |
| `openvisionkit.capture.video_recorder` | `VideoRecorder` dataclass — start/write/pause/resume/stop recording to MP4 or animated GIF (via `imageio`), elapsed-time tracking, and optional audio attachment. |
| `openvisionkit.capture.screen_capture` | `ScreenCapture` — grabs monitor frames so a demo can process the screen instead of a camera. |
Import from the submodule path — `openvisionkit.capture.__init__` is empty:

```python
from openvisionkit.capture.video_template import video_capture_template, KeyEventManager
from openvisionkit.capture.image_template import image_template
```

### `template/` — Superseded Local Scaffolding

`template/video_template.py`, `template/image_template.py`, `template/video_recorder.py`, and
`template/screen_capture.py` are the original in-repo versions of the scaffolding above. Every
project script now imports from `openvisionkit.capture` instead; these files are kept for reference
and rollback. One file in the folder is still live:

| File | Description |
| --- | --- |
| `template/draw_object.py` | `DrawingObject` — an interactive on-canvas shape (circle/rectangle/etc.) with hover and selected states, hit testing, movement, even distribution across the frame, and dict serialisation. Used for gesture-driven UI controls such as the hair-colour picker. |

### `models/` — Model Files

| File | Used by |
| --- | --- |
| `hand_landmarker.task`, `gesture_recognizer.task` | `module/hand_detector.py` |
| `face_detector.tflite`, `blaze_face_full_range.tflite` | `module/face_detector.py` |
| `face_landmarker_v2_with_blendshapes.task` | `module/face_mesh_detector.py` |
| `pose_landmarker.task` | `module/pose_detector.py` |
| `efficientdet_lite.tflite`, `ssd_mobilenet_v2_coco.tflite` | `module/object_detector.py` |
| `deeplab_v3.tflite`, `deeplabv3_plus_mobilenet.tflite`, `interactive_segmenter.tflite` | `module/selfie_segmentation.py` |
| `hair_segmenter.tflite` | `module/hair_segmentation.py` |
| `frozen_inference_graph.pb`, `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`, `coco.names` | `object_detection/coco_object_detection.py` (OpenCV DNN) |

MediaPipe landmark index references are kept for convenience:

| Palm keypoints | Face landmark keypoints | Pose keypoints |
| --- | --- | --- |
| ![Palm keypoints](images/github/mediapipe/palm_keypoints.png) | ![Face landmark keypoints](images/github/mediapipe/face_landmark_keypoints.png) | ![Pose keypoints](images/github/mediapipe/pose_keypoints.png) |

---

## Projects

### hand_gesture

Hand-tracking applications built on `module/hand_detector.py` + `openvisionkit.capture.video_template`.

| File | Description |
| --- | --- |
| `multiple_hand_gesture.py` | Baseline two-hand demo: detects both hands, draws landmarks, bounding boxes and handedness labels, and connects the two palm centres with a line. Good starting point for reading `HandDetector` output. |
| `finger_counter.py` | Counts raised fingers with `fingers_up()` and overlays the matching finger-count image from `hand_gesture/images/`. |
| `hand_distance_measurement.py` | Estimates camera-to-hand distance in centimetres from palm width in pixels (INDEX_MCP↔PINKY_MCP is used because knuckles are more stable than fingertips), smoothed with a `deque` rolling average. |
| `volume_control_by_hand.py` | Thumb–index pinch distance drives system volume, with auto-calibration, smoothing and an on-screen volume bar. Uses `pycaw` on Windows and `osascript` via `subprocess` on macOS. |
| `zoom_gesture.py` | Two-handed pinch-to-zoom: the distance between both hands scales an overlaid image, clamped to a safe ROI inside the frame. |
| `drag_and_drop.py` | `DraggingRectangle` objects follow the index fingertip while index and middle finger are pinched together; includes transparency rendering for multiple rectangles. |
| `image_drag_and_drop.py` | Same interaction applied to real images (`DrawingSpec` class): images are loaded from a folder, resized, alpha-composited, and dragged with a pinch gesture. |
| `ai_virtual_mouse.py` | Turns the hand into a mouse via `pyautogui`: index finger up moves the cursor (frame coordinates interpolated to screen coordinates inside a control box), index + middle up clicks, and pinch distance normalised by palm width triggers scroll mode. |
| `ai_virtual_keyboard.py` | On-screen QWERTY keyboard with a transparent backdrop. `Button` objects highlight on hover; a normalised pinch registers a keypress through `pynput`, with a `pygame` click sound from `hand_gesture/audio/click.wav`. |
| `ai_virtual_painter.py` | Air-drawing canvas: a header of tool images selects brush colour or eraser, and index-finger strokes are drawn onto a persistent canvas that is blended back over the camera frame. |
| `calculator.py` | Gesture-driven calculator with `DisplayBox` and `Button` classes, hover/click feedback, mouse-click support, and safe expression evaluation using `ast` + `operator` (no `eval`). |
| `gesture_presentation.py` | Slide-deck controller: thumb-only for previous slide, pinky-only for next slide (only above a threshold line), index + middle for a pointer, index-only for freehand annotation strokes, and index + middle + ring to undo the last stroke. Includes stroke smoothing. |
| `hand_sign_detection/hand_sign_detection_capture.py` | Dataset capture tool: crops the hand region onto a square white canvas (aspect-preserving) and saves training images per letter on the `s` key. |
| `hand_sign_detection/hand_sign_detection_test_model.py` | Runs the captured dataset's Teachable Machine model through `module/classifier.py` and predicts the American Sign Language letter live. |
| `kids_game/ping_pong_game.py` | Two-player hand-controlled ping pong. Hand bounding boxes act as paddles, ball physics and per-player scores live in the template `state` dict, and `r` resets the game through `KeyEventManager`. |
| `3d_tracking/3d_hand_tracking.py` | Streams flattened hand landmarks over **UDP** (`socket` + `json`) to Unity, with a payload-size guard. |
| `3d_tracking/HandTracking.cs`, `UDPReceive.cs`, `LineCode.cs` | Unity C# receivers that consume the UDP stream and render a 3D hand skeleton. |

**Screenshots**

| Hand landmarks (left / right) | Finger counter |
| --- | --- |
| ![Left hand landmarks](images/github/left-hand-landmarks.png) ![Right hand landmarks](images/github/right-hand-landmarks.png) | ![Finger counter 1](images/github/finger_couner1.png) ![Finger counter 2](images/github/finger_couner2.png) ![Finger counter 3](images/github/finger_couner3.png) ![Finger counter 4](images/github/finger_couner4.png) |

| Hand distance measurement | Volume control |
| --- | --- |
| ![Hand distance calculation](images/github/hand_distance_calculation.png) | ![Volume control](images/github/volume_control.png) ![Volume control gesture](images/github/volume-control-hand-gesture.png) |

| Zoom gesture | Drag and drop |
| --- | --- |
| ![Zoom gesture animation](images/github/zoom_gesture.gif) ![Zoom gesture 1](images/github/zoom_gesture1.png) ![Zoom gesture 2](images/github/zoom_gesture2.png) ![Zoom gesture 3](images/github/zoom_gesture3.png) | ![Drag and drop 1](images/github/drag_and_drop1.png) ![Drag and drop 2](images/github/drag_and_drop2.png) |

| Image drag and drop | AI virtual mouse |
| --- | --- |
| ![Image drag and drop 1](images/github/img_drag_n_drop_1.png) ![Image drag and drop 2](images/github/img_drag_n_drop_2.png) ![Image drag and drop 3](images/github/img_drag_n_drop_3.png) | ![Virtual mouse 1](images/github/ai_virtual_mouse1.png) ![Virtual mouse 2](images/github/ai_virtual_mouse2.png) ![Virtual mouse 3](images/github/ai_virtual_mouse3.png) |

| AI virtual keyboard | AI virtual painter |
| --- | --- |
| ![Virtual keyboard 1](images/github/ai_virtual_keyboard_1.png) ![Virtual keyboard 2](images/github/ai_virtual_keyboard_2.png) ![Virtual keyboard 3](images/github/ai_virtual_keyboard_3.png) | ![Painter 1](images/github/painter1.png) ![Painter 2](images/github/painter2.png) ![Painter 3](images/github/painter3.png) ![Painter 4](images/github/painter4.png) ![Painter 5](images/github/painter5.png) |

| Calculator | Gesture presentation |
| --- | --- |
| ![Calculator hand gesture](images/github/calculator_handgesture.png) ![Calculator mouse click](images/github/calculator_mouse_click.png) | ![Pointer mode](images/github/gesture_presentation_pointer_mode.png) ![Show pointer](images/github/gesture_presentation_show_pointer.png) ![Draw mode](images/github/gesture_presentation_draw_mode.png) ![Next page](images/github/gesture_presentation_next_page.png) ![Previous page](images/github/gesture_presentation_previous_page.png) |

| Hand sign detection (ASL) | 3D hand tracking (Unity via UDP) |
| --- | --- |
| ![Sign A](images/github/americal_sign_A.png) ![Sign B](images/github/americal_sign_B.png) ![Sign C](images/github/americal_sign_C.png) ![Sign D](images/github/americal_sign_D.png) | ![3D hand tracking](images/github/3D_handtracking_1.png) ![3D hand tracking config](images/github/3D_handtracking_config.png) ![3D hand tracking move object](images/github/3D_handtracking_move_object.png) |

---

### face_detection

Bounding-box face detection using `module/face_detector.py`.

| File | Description |
| --- | --- |
| `detect_faces.py` | Detects faces in both image and video modes, draws corner-styled bounding boxes with `util.utility.put_text_think_corners`, and passes a millisecond timestamp for accurate video-mode results. |
| `face_blur.py` | Crops each detected face region and applies a blur/pixelation before writing it back into the frame — a simple privacy filter. |

| Face detection |
| --- |
| ![Face detection 1](images/github/face_detection1.png) ![Face detection 2](images/github/face_detection2.png) |

---

### face_landmark

478-point face-mesh applications using `module/face_mesh_detector.py`.

| File | Description |
| --- | --- |
| `face_landmark.py` | Baseline mesh demo: draws all landmarks and the face bounding box, prints the face index and sample blendshape values (smile, blink), and shows FPS. |
| `face_landmark_more_features.py` | Extended demo layering head pose (yaw/pitch/roll) from the transform matrix, blendshape-derived emotion, AR filter overlay, and a multi-line information panel. |
| `blink_counter.py` | Blink counter driven by the eye-aspect ratio: draws eye landmarks, streams the ratio into `module/live_plot.py`, and shows the camera feed and live graph side by side with `util.utility.stack_images_grid`. |
| `face_distance_measurement.py` | Estimates camera distance from the inter-pupillary distance using a calibrated focal length (`calculate_focal_length`). |
| `kids_game/kids_game_eatable.py` | Kids' game: fruit items fall down the screen and are "eaten" when the mouth-openness ratio is high and the mouth centre is close enough to the item; tracks a success count and game-over state, with `r` to reset. |

| Face mesh | Extended features |
| --- | --- |
| ![Face landmarks 1](images/github/face_landmarks_1.png) ![Face landmarks 2](images/github/face_landmarks_2.png) | ![Face landmark more features](images/github/face_landmark_more_features_1.png) |

| Blink counter | Face distance measurement | Kids game (eat the fruit) |
| --- | --- | --- |
| ![Blink counter](images/github/blink_counter_1.png) | ![Face distance 1](images/github/face_distance_measurement_1.png) ![Face distance 2](images/github/face_distance_measurement_2.png) | ![Kids game 1](images/github/kids_game_1.png) ![Kids game 2](images/github/kids_game_2.png) ![Kids game 3](images/github/kids_game_3.png) |

---

### pose_estimation

Body-pose applications using `module/pose_detector.py`. Sample clips live in `pose_estimation/videos/`.

| File | Description |
| --- | --- |
| `pose_landmarks.py` | Baseline pose demo: detects the 33 landmarks, draws a shoulder-alignment line between landmarks 11 and 12, and optionally renders the segmentation mask. |
| `ai_personal_trainer.py` | Rep counter and form tracker: computes the elbow angle (landmarks 11-13-15), converts it to a workout percentage, counts reps, and draws a progress bar with percentage and count. Runs against `pose_estimation/videos/gym_video_2.mp4`. |
| `shirt_try_on.py` | Virtual try-on: scales a transparent shirt PNG from `pose_estimation/images/shirt_try_on/shirts/` to the detected shoulder width, aligns the collar (~15% from the top of the image) to the neck landmark, and overlays it. Left/right button images switch shirts. |

| AI personal trainer |
| --- |
| ![Personal trainer 1](images/github/ai_personal_trainer_1.png) ![Personal trainer 2](images/github/ai_personal_trainer_2.png) ![Personal trainer 3](images/github/ai_personal_trainer_3.png) ![Personal trainer 4](images/github/ai_personal_trainer_4.png) |

---

### hair_segmentation

Hair recolouring using `module/hair_segmentation.py`.

| File | Description |
| --- | --- |
| `hair_segment.py` | Minimal demo: segments hair and tints it with a fixed colour. |
| `hair_segmentation_color_option.py` | Interactive colour picker: `template/draw_object.py` renders colour swatches on the frame, `module/hand_detector.py` tracks the index fingertip, and touching a swatch changes the hair colour live. |

| Hair segmentation | Colour selection |
| --- | --- |
| ![Hair segmentation](images/github/hair_segmentation.png) | ![Hair red](images/github/hair_segmentation_choose_color_red.png) ![Hair blue](images/github/hair_segmentation_choose_color_blue2.png) ![Hair cyan](images/github/hair_segmentation_choose_color_cyan.png) |

---

### selfie_segmentation

Virtual-background effects using `module/selfie_segmentation.py`. Backgrounds live in
`selfie_segmentation/images/background/`.

| File | Description |
| --- | --- |
| `remove_background.py` | Removes the background, leaving the person on a transparent/empty background. |
| `blur_background.py` | Keeps the person sharp and applies a Gaussian blur to the background (video-call style). |
| `color_background.py` | Replaces the background with a solid colour (green-screen style). |
| `replace_background.py` | Replaces the background with an image, demonstrating three approaches: the plain `replace_background()` call, the optimised virtual-background path, and confidence-based alpha blending. |
| `mask_images.py` | Mask inspection demo: visualises the raw segmentation mask and layered/multi-background compositing. |

---

### object_detection

| File | Description |
| --- | --- |
| `detect_objects.py` | MediaPipe **EfficientDet-Lite** detection via `module/object_detector.py`, for both `assets/image/coco_test_image.jpg` and live video. |
| `coco_object_detection.py` | The OpenCV DNN alternative: loads `models/frozen_inference_graph.pb` + the SSD MobileNet v3 config and `models/coco.names`, then draws class labels with confidence scores. Includes separate image and video logic functions. |

---

### image_detection

Classic OpenCV measurement and highlight extraction using `module/image_detector.py`.

| File | Description |
| --- | --- |
| `image_highlight_extraction.py` | Extracts highlighter-marked text from a document: finds dominant HSV colours, builds and combines per-colour masks, refines them, finds contours of the highlighted regions, OCRs each region, writes the text to a file, and saves an annotated image. |
| `object_distance_measurement.py` | Measures real-world object dimensions: takes the largest contour as a reference object of known width, derives a pixels-per-centimetre scale with `cv2.minAreaRect`, then labels every other contour with its measured size. |

| Highlight extraction | Object measurement |
| --- | --- |
| ![Combined mask](images/github/highlight_image_combined_mask.png) ![Combined mask regions](images/github/highlight_combined_mask_regions.png) ![Mask with Gaussian blur](images/github/highlight_mask_ith_gaussian_blur.png) | ![Object measurement 1](images/github/object_measurement1.png) ![Object measurement 2](images/github/object_measurement2.png) ![Object measurement 3](images/github/object_measurement3.png) |

---

### text_detection

OCR and form-understanding pipeline using `module/text_detector.py`, `module/form_detector.py`,
`module/form_roi_detector.py` and `module/form_roi_annotator.py`.
Test assets live in `text_detection/assets/` (cards, a blank `main_form.jpg`, and filled form scans);
outputs are written to `text_detection/results/`.

| File | Description |
| --- | --- |
| `detect_text.py` | Character-level and word-level OCR on card images, with commented examples for spaCy entity extraction and hOCR/PDF export. |
| `form_extraction.py` | End-to-end form data extraction. Aligns a filled scan against the blank template with ORB feature matching (`compare_matches`), loads hand-annotated ROIs from `annotated_rois_latest.json`, then handles each field by type — textbox/dropdown via OCR, radio and checkbox via fill-state detection, and paragraphs via the cursive-text path — before writing an annotated image and a CSV of the extracted values. |

| ORB feature matching / alignment | Form keypoints |
| --- | --- |
| ![Form ORB](images/github/form_orb.png) ![Form ORB 2](images/github/form_orb_2.png) ![OpenCV ORB 0](images/github/open_cv_orb_0.png) ![OpenCV ORB 1](images/github/open_cv_orb_1.png) ![OpenCV ORB 2](images/github/open_cv_orb_2.png) | ![Form detection keypoints](images/github/form_detection_keypoints.png) |

| ROI annotation (`module/form_roi_annotator.py`) | Extracted data |
| --- | --- |
| ![Annotated form ROIs](images/github/annotated_image_20260418_002406.png) ![Annotate selected](images/github/form_annotate_selected.png) | ![Annotated form](images/github/annotated_image_form_20260419_215216.png) ![Registration form extraction](images/github/registration_form_extraction.png) |

Reference ROI definitions used by these screenshots: [`images/github/annotated_rois_latest.json`](images/github/annotated_rois_latest.json).

---

### car_parking_slot

Classic (non-neural) parking-occupancy detection.

| File | Description |
| --- | --- |
| `car_parking_pickle_download.py` | Setup tool: downloads the reference parking-lot image if needed and lets you click parking slots to add or remove them, persisting the slot coordinates to `car_parking_slots.pkl`. |
| `car_parking_slot_finder.py` | Occupancy detector: loads the slot list from the pickle, thresholds and dilates each frame, then counts non-zero pixels per slot to decide free vs occupied, labelling each slot with `util.utility.put_text_rect` and showing a free-slot count. |

| Parking slot detection |
| --- |
| ![Car parking 1](images/github/car_parking_1.png) ![Car parking 2](images/github/car_parking_2.png) |

---

### interactive_map

| File | Description |
| --- | --- |
| `1_get_corner_points/get_map.py` | Step 1 of a perspective-warp map project: a mouse callback collects four corner points from the video frame and draws them, producing the source quad for a later `cv2.warpPerspective` transform. |

---

## Image Gallery Index

All README screenshots live in [`images/github/`](images/github). MediaPipe landmark reference charts are in
[`images/github/mediapipe/`](images/github/mediapipe).

| Project | Images |
| --- | --- |
| hand_gesture — landmarks | `left-hand-landmarks.png`, `right-hand-landmarks.png` |
| hand_gesture — finger counter | `finger_couner1.png` … `finger_couner4.png` |
| hand_gesture — hand distance | `hand_distance_calculation.png` |
| hand_gesture — volume control | `volume_control.png`, `volume-control-hand-gesture.png` |
| hand_gesture — zoom gesture | `zoom_gesture.gif`, `zoom_gesture1.png` … `zoom_gesture3.png` |
| hand_gesture — drag and drop | `drag_and_drop1.png`, `drag_and_drop2.png` |
| hand_gesture — image drag and drop | `img_drag_n_drop_1.png` … `img_drag_n_drop_3.png` |
| hand_gesture — virtual mouse | `ai_virtual_mouse1.png` … `ai_virtual_mouse3.png` |
| hand_gesture — virtual keyboard | `ai_virtual_keyboard_1.png` … `ai_virtual_keyboard_3.png` |
| hand_gesture — virtual painter | `painter1.png` … `painter5.png` |
| hand_gesture — calculator | `calculator_handgesture.png`, `calculator_mouse_click.png` |
| hand_gesture — gesture presentation | `gesture_presentation_pointer_mode.png`, `gesture_presentation_show_pointer.png`, `gesture_presentation_draw_mode.png`, `gesture_presentation_next_page.png`, `gesture_presentation_previous_page.png` |
| hand_gesture — hand sign detection | `americal_sign_A.png` … `americal_sign_D.png` |
| hand_gesture — 3D tracking | `3D_handtracking_1.png`, `3D_handtracking_config.png`, `3D_handtracking_move_object.png` |
| face_detection | `face_detection1.png`, `face_detection2.png` |
| face_landmark | `face_landmarks_1.png`, `face_landmarks_2.png`, `face_landmark_more_features_1.png`, `blink_counter_1.png`, `face_distance_measurement_1.png`, `face_distance_measurement_2.png`, `kids_game_1.png` … `kids_game_3.png` |
| pose_estimation | `ai_personal_trainer_1.png` … `ai_personal_trainer_4.png` |
| hair_segmentation | `hair_segmentation.png`, `hair_segmentation_choose_color_red.png`, `hair_segmentation_choose_color_blue2.png`, `hair_segmentation_choose_color_cyan.png` |
| image_detection | `highlight_image_combined_mask.png`, `highlight_combined_mask_regions.png`, `highlight_mask_ith_gaussian_blur.png`, `object_measurement1.png` … `object_measurement3.png` |
| text_detection | `form_orb.png`, `form_orb_2.png`, `open_cv_orb_0.png` … `open_cv_orb_2.png`, `form_detection_keypoints.png`, `form_annotate_selected.png`, `annotated_image_20260418_002406.png`, `annotated_image_form_20260419_215216.png`, `registration_form_extraction.png`, `annotated_rois_latest.json` |
| car_parking_slot | `car_parking_1.png`, `car_parking_2.png` |
| MediaPipe references | `mediapipe/palm_keypoints.png`, `mediapipe/face_landmark_keypoints.png`, `mediapipe/face_landmark_keypoints.jpg`, `mediapipe/pose_keypoints.png` |

`selfie_segmentation`, `object_detection` and `interactive_map` currently have no screenshots in `images/github/`.

---

## Notes

- Some demos expect per-project asset folders that are not committed (for example
  `hand_gesture/images/`, `hand_gesture/audio/click.wav`, `face_detection/assets/`,
  `car_parking_slot/assets/car_parking.mp4`). Add your own assets at those paths, or point the
  script at a different file, before running.
- Output folders: `screenshots/` for captured frames, `recordings/` for MP4/GIF recordings,
  `text_detection/results/` for OCR and form-extraction artifacts.
- `module/form_detector.py` and `module/form_roi_detector.py` both define a `FormROIDetector`.
  The `form_roi_detector.py` version is the newer, more capable one; `form_extraction.py`
  currently imports the one from `form_detector.py`.
- `text_detection/form_extraction.py` also needs
  `text_detection/assets/annotated_form/annotated_rois_latest.json`, which is not committed; generate
  it by annotating a blank form with `module/form_roi_annotator.py` first.
- External research links and experiment notes are kept in [`NOTES.md`](NOTES.md).
