# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A collection of standalone real-time computer vision demos (hand gestures, face landmarks, pose
estimation, segmentation, OCR/form extraction) built on **MediaPipe Tasks**, **OpenCV**, and
**Tesseract OCR**. Each top-level folder is one project; two shared packages
(`module/`, `util/`) plus the external `openvisionkit` package are imported by all of them.

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`). There is **no build step, no
test suite, and no linter config**. There is nothing to build and nothing to run tests against —
verification means launching a demo and watching the OpenCV window. Do not invent test/build
commands or add a test framework unless asked.

The repository is tracked in git (branch `master`, remote `origin`). Binary assets — models,
images, video, audio — are stored in **Git LFS**; source and text files deliberately are not, see
"Git LFS" below.

See `README.md` for the project description, the detection-type table, the architecture overview,
the per-file description of all 69 Python files, and the screenshot gallery.

## Running demos

Every script uses absolute imports (`from module...`, `from util...`) and every
detector defaults its `model_path` to `./models/*` — both are resolved relative to the **current
working directory**, so scripts must be run from the repository root:

```bash
uv run python -m hand_gesture.finger_counter
uv run python -m face_landmark.blink_counter
uv run python -m pose_estimation.ai_personal_trainer
```

Running a script from inside its own folder (`cd hand_gesture && python finger_counter.py`) fails on
both imports and model loading.

For a headless check that a script at least imports and processes frames, pass
`show_window=False` to `video_capture_template`, or import the module and call its
`custom_logic(frame)` function directly with a `cv2.imread` frame.

Runtime keys handled by `openvisionkit.capture.video_template`: `Esc` quit, `s` screenshot
(`screenshots/`), `r` start/stop recording (`recordings/`), `p` pause/resume recording.

## Dependencies

`pyproject.toml` and `uv.lock` hold the Python dependencies; `uv sync` creates `.venv` and installs
them. Run every script through `uv run` so the project environment is used.

```bash
uv sync                                                        # create .venv from the lockfile
uv add <package>                                               # add a new dependency
brew install tesseract                                         # required for all OCR projects
```

Not in the lockfile — install only when working on the feature that needs it:

```bash
uv add tensorflow tf-keras                                     # hand_gesture/hand_sign_detection only
uv add spacy && uv run python -m spacy download en_core_web_sm # TextDetector NLP methods only
uv add pycaw comtypes                                          # Windows volume control only
```

`mediapipe` is pinned to `>=0.10,<1.0`. On mediapipe 1.0.1 every MediaPipe Tasks vision graph aborts
at construction on macOS with `graph_service.h: Check failed: service_ Service is unavailable.`,
which kills the interpreter as soon as any detector in `module/` is created.

The video and image scaffolding comes from the published **`openvisionkit`** package, not from the
local `template/` folder — see "The capture templates live in openvisionkit" below.

## Architecture

### Layer dependency direction

```
<project>/*.py  --imports-->  openvisionkit.capture   (external package: window loop, FPS,
       |                                                recording, screenshots, key handling)
       +---------imports-->  module/ , util/            (util imports nothing internal)
```

- `module/` — detector/estimator classes. Depend only on third-party libraries, never on
  `util/`, `template/`, or a project folder.
- `util/utility.py` — flat function library (drawing, geometry, HSV masks, layout, path helpers).
  No internal dependencies. Imported as `from util import utility`.
- `openvisionkit.capture` — application scaffolding (`video_template`, `image_template`,
  `video_recorder`, `screen_capture`). An external dependency, so fixes to it belong in the
  `openvisionkit` project, not in this repository.
- `template/` — the superseded local copy of that scaffolding, kept for reference and rollback.
  Only `template/draw_object.py` is still imported by a project script
  (`hair_segmentation/hair_segmentation_color_option.py`).
- Project folders — thin scripts that wire a detector to a frame callback. Business logic lives
  here; anything reusable belongs in `module/` or `util/`.

### The canonical demo pattern

Nearly every project file follows this shape. Preserve it when adding a demo:

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

Detectors must be built at module level, not inside `custom_logic` — constructing one loads a
MediaPipe model, which is far too slow to do per frame.

### The capture templates live in openvisionkit

Every project script imports the window loop from the external package:

```python
from openvisionkit.capture.video_template import video_capture_template, KeyEventManager
from openvisionkit.capture.image_template import image_template
```

`openvisionkit.capture.__init__` is empty, so import from the submodule path — `from
openvisionkit.capture import video_capture_template` fails. Both functions take exactly the same
parameter names and defaults as the old local `template/` copies, so no call site changed; the
package version bundles its own `FPSCounter` and imports `pyautogui` lazily. New demos must import
from `openvisionkit.capture`, never from `template/`.

### Two different path conventions (easy to get wrong)

| What | Resolved against | How |
| --- | --- | --- |
| Model files (`./models/...`) | Current working directory | Detector default arguments |
| Project assets (`images/`, `assets/`, `videos/`, `audio/`) | The script's own folder | `path, _ = utility.get_calling_folder()` then f-string |

`utility.get_calling_folder()` uses `inspect.stack()[1]`, so it must be called **directly from the
script that wants its own directory** — wrapping it in a helper returns the wrapper's folder instead.
`utility.get_currect_path()` returns `(path, filename, full_path)` for `util/utility.py` itself, not
for the caller. `utility.find_project_root(start, marker="module")` walks upward looking for a
marker folder.

### MediaPipe running modes and timestamps

Detectors take `running_mode="IMAGE"` or `"VIDEO"` (`PoseDetector` takes the
`vision.RunningMode` enum directly). In VIDEO mode MediaPipe requires a monotonically increasing
millisecond timestamp:

- `PoseDetector.detect()` auto-generates one (`frame_count * 33`, ~30 FPS) when `timestamp_ms` is
  omitted.
- `FaceDetector` and `HandDetector` expect the caller to pass `timestamp_ms` in VIDEO mode
  (see `face_detection/detect_faces.py`).

Pass plain **BGR** frames from OpenCV; every detector converts to RGB internally.

### Detector return shapes differ — check before destructuring

| Call | Returns |
| --- | --- |
| `HandDetector.draw_landmarks(frame, ...)` | `(annotated_image, [(landmarks_list, bounding_box, landmark_params, hand_type), ...])` |
| `HandDetector.get_landmarks(frame)` | `[{"landmarks_list", "bounding_box", "center_point", "hand_type"}, ...]` — a dict per hand, **not** the tuple above |
| `PoseDetector.detect(frame)` | `(annotated_image, PoseLandmarkerResult)` — the raw MediaPipe result; use `get_all_postion()` for pixel coordinates |
| `FaceMeshDetector.face_mesh_detection(frame)` | `(annotated_image, faces, blendshapes, transformation_matrices, bboxes)` |
| `FaceDetector.detect_faces(frame, ...)` | `(annotated_image, detections)` with parsed boxes and keypoints |

`landmarks_list` entries are `[landmark_id, x, y, z]` in **pixel** coordinates. Hand landmark index
constants are attributes of `HandDetector` (`fingerTips`, `fingerPips`, `fingerDips`, `finger_mcp`,
`wrist`).

### Stateful and interactive demos

Cross-frame state goes in a plain dict passed as `state=` to `video_capture_template`; keyboard
actions are registered on a `KeyEventManager` and passed as `key_manager=`. Handlers receive
`(frame, state)`. `hand_gesture/kids_game/ping_pong_game.py` is the reference example. Mouse
interaction uses `mouse_callback=` / `mouse_callback_params=`.

## Known quirks in the existing code

- `module/classifier.py` imports its loader from `tf_keras`, not `tensorflow.keras`. The Teachable
  Machine `.h5` files are Keras 2; Keras 3 (TensorFlow 2.16 and newer) cannot read them. The
  `TF_USE_LEGACY_KERAS` environment variable is not a workaround here — it is only read when
  TensorFlow is first imported, and every script that uses `Classifier` imports a `module/` detector
  first, which pulls in mediapipe and Keras 3 before `classifier.py` runs.
- `module/hand_detector.py:11` opens `cv2.VideoCapture(0)` at module scope, so **importing
  `HandDetector` grabs the camera** on some platforms. It exists only for the `main()` demo at the
  bottom of that file. Several project scripts also create an unused module-level `cap`.
- `HandDetector.draw_landmarks()` ends with a `cv2.cvtColor(..., COLOR_RGB2BGR)`; its returned image
  is already display-ready, so do not convert it again.
- `module/form_detector.py` and `module/form_roi_detector.py` both define a class named
  `FormROIDetector`. `form_roi_detector.py` is the newer, more capable one, but
  `text_detection/form_extraction.py` imports the version from `form_detector.py`.
- `util/utility.py` duplicates several `ImageDetector` methods (`detect_highlighted_text`,
  `get_dominant_hsv_colors`, `refine_mask`, `find_contours`) as free functions. Both are live.
- `text_detection/form_extraction.py` reads its ROI definitions from
  `text_detection/assets/annotated_form/annotated_rois_latest.json` at import time, so that file
  must exist before the module can even be imported. It is committed and matches
  `assets/form/main_form.jpg`; re-annotating a different blank form with
  `module/form_roi_annotator.py` replaces it.
- Several asset folders referenced by scripts are not committed (`hand_gesture/images/`,
  `hand_gesture/audio/click.wav`, `face_detection/assets/`, `car_parking_slot/assets/`). Scripts
  that need them will fail on a fresh checkout until assets are supplied.
- Spelling of existing public names is inconsistent (`get_all_postion`, `get_currect_path`,
  `finger_couner*.png`). Do not silently rename them; call sites depend on the current spelling.

## Git LFS

`.gitattributes` tracks **binary assets only** in LFS: `*.task`, `*.tflite`, `*.pb`, `*.h5`,
`*.pkl`, `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.mp4`, `*.wav`.

Source and text files (`*.py`, `*.md`, `*.json`, `*.toml`, `*.lock`, `*.txt`, `*.cs`, `*.names`,
`*.csv`) are deliberately **not** in LFS. They were until the tracking rules were narrowed: LFS
sets `diff: lfs`, which suppresses textual diffs, and any clone made with `GIT_LFS_SKIP_SMUDGE=1`
leaves every one of them as a three-line pointer stub instead of source code:

```
version https://git-lfs.github.com/spec/v1
oid sha256:...
size 34374
```

Do not add text or source patterns back to `.gitattributes`.

If files in the working tree ever appear as those pointer stubs, the LFS objects are present but
the smudge filter did not run. Restore them without re-cloning:

```bash
git lfs checkout          # rewrite pointer stubs in the worktree from local LFS objects
git lfs fsck              # verify local object integrity
git lfs pull              # only if objects are missing locally
```

Commits made before the tracking rules were narrowed still hold text files as LFS pointers, so
`git show <old-sha>:file.py` needs LFS available to read them.

## Platform notes

- `hand_gesture/volume_control_by_hand.py` branches per OS: `pycaw` on Windows, `osascript` via
  `subprocess` on macOS.
- `pyautogui`-driven demos (virtual mouse, keyboard) need macOS Accessibility permission, and all
  camera demos need Camera permission, for the terminal running Python.
- `hand_gesture/3d_tracking/3d_hand_tracking.py` sends landmarks over UDP to the Unity C# scripts in
  the same folder; the payload is size-guarded against the UDP datagram limit.
