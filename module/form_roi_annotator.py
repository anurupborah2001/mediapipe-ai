import cv2
import sys
import json
from typing import List, Tuple
from datetime import datetime
import os

"""
Form ROI Annotator
- Click two diagonal corners to define a form field ROI
- Choose field type + label when prompted
- Keyboard shortcuts for undo, delete, edit, clear, save, list, help
- Auto-saves after every change + timestamped backups
- Output format: roi = [[(x1,y1), (x2,y2), "type", "label","category"], ...]

Undo last ROI => Press u
Delete specific ROI => Press d (shows numbered list)
Edit existing ROI => Press e (change type or label)
Auto-save with timestamp => Every change + final exit (JSON + overlaid PNG)
Always saves annotated_rois_latest.json (easy to continue)
Load previous session => Pass JSON as second argument
Right-click to cancel current selection
Clear all => Press c (with confirmation)
Manual save anytime => Press s
List ROIs => Press l
Help menu => Press h

Other features to consider:
Index numbers shown on image for easy identification
Custom field types supported
More field types pre-loaded
Annotated image export with all boxes/labels


Usage:
python form_roi_annotator.py your_form.jpg
# or continue from previous:
python form_roi_annotator.py your_form.jpg annotated_rois_latest.json

"""
class FormROIAnnotator:
    def __init__(self, image_path: str, load_path: str = None):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self.clone = self.image.copy()
        self.rois: List = []   # Now: [[(x1,y1), (x2,y2), "type", "label", "category"], ...]
        self.current_points: List[Tuple[int, int]] = []

        # Supported field types
        self.field_types = [
            "text", "textbox", "box", "checkbox", "radio",
            "daterange", "date_range", "table", "table_cell",
            "signature", "dropdown", "header", "footer", "paragraph",
            "logo", "line", "custom"
        ]

        # Load existing annotations if provided
        if load_path and os.path.exists(load_path):
            try:
                with open(load_path, "r") as f:
                    self.rois = json.load(f)
                print(f"✅ Loaded {len(self.rois)} existing ROIs from {load_path}")
                self._redraw_all_rois()
            except Exception as e:
                print(f"⚠️ Could not load existing ROIs: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            cv2.circle(self.clone, (x, y), 6, (0, 255, 255), -1)
            cv2.imshow("Form ROI Annotator", self.clone)

            if len(self.current_points) == 2:
                pt1 = self.current_points[0]
                pt2 = self.current_points[1]

                x1 = min(pt1[0], pt2[0])
                y1 = min(pt1[1], pt2[1])
                x2 = max(pt1[0], pt2[0])
                y2 = max(pt1[1], pt2[1])

                self._select_and_add_roi(x1, y1, x2, y2)

                self.current_points.clear()
                self.clone = self.image.copy()
                self._redraw_all_rois()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                self.current_points.clear()
                self.clone = self.image.copy()
                self._redraw_all_rois()
                print("🗑️ Current selection cancelled.")

    def _select_and_add_roi(self, x1: int, y1: int, x2: int, y2: int):
        print("\n" + "=" * 75)
        print("Select Form Field Type:")
        for i, t in enumerate(self.field_types, 1):
            print(f"  {i:2d}. {t}")
        print("   0. Custom type")

        while True:
            try:
                choice = int(input("\nEnter number for field type: "))
                if choice == 0:
                    field_type = input("Enter custom field type: ").strip() or "custom"
                    break
                if 1 <= choice <= len(self.field_types):
                    field_type = self.field_types[choice - 1]
                    break
                print("Invalid number.")
            except ValueError:
                print("Please enter a valid number.")

        # Label (what is written inside / on the field)
        label = input(f"Enter label/text visible on field (press Enter to skip): ").strip()

        # NEW: Category (semantic group / section the field belongs to)
        print("\nEnter Category (e.g., Personal Info, Contact, Meal Preference, Satisfaction, etc.)")
        category = input("Category: ").strip()
        if not category:
            category = "Uncategorized"

        # Store in new 5-element format
        self.rois.append([(x1, y1), (x2, y2), field_type, label, category])

        print(f"✅ ROI Added → Type: {field_type} | Label: '{label}' | Category: '{category}'")

        self._save_latest()

    def _redraw_all_rois(self):
        self.clone = self.image.copy()

        for idx, roi in enumerate(self.rois, 1):
            (x1, y1), (x2, y2), ftype, label, category = roi
            color = self._get_color(ftype)

            cv2.rectangle(self.clone, (x1, y1), (x2, y2), color, 3)

            # Main display text: Type + Label
            display_text = f"{ftype}: {label}" if label else ftype
            cv2.putText(self.clone, display_text, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            # Show Category below the box
            if category and category != "Uncategorized":
                cv2.putText(self.clone, f"[{category}]", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2, cv2.LINE_AA)

            # Index number
            cv2.putText(self.clone, str(idx), (x1 + 8, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(self.clone, str(idx), (x1 + 8, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 1)

        cv2.setWindowTitle("Form ROI Annotator", f"Form ROI Annotator — {len(self.rois)} ROIs")

    def _get_color(self, ftype: str):
        color_map = {
            "text": (0, 255, 0),
            "textbox": (0, 165, 255),
            "box": (255, 0, 0),
            "checkbox": (255, 0, 255),
            "radio": (0, 255, 255),
            "daterange": (255, 140, 0),
            "table": (128, 0, 255),
            "table_cell": (255, 0, 128),
            "signature": (0, 128, 255),
            "dropdown": (100, 200, 50),
            "header": (0, 100, 200),
            "footer": (200, 100, 0),
            "custom": (255, 255, 0),
        }
        return color_map.get(ftype, (255, 255, 0))

    def _undo_last(self):
        if not self.rois:
            print("⚠️ No ROIs to undo.")
            return
        removed = self.rois.pop()
        print(f"🗑️ Undid last ROI: {removed}")
        self._redraw_all_rois()
        self._save_latest()

    def _delete_specific(self):
        if not self.rois:
            print("⚠️ No ROIs.")
            return
        print("\nCurrent ROIs:")
        for i, r in enumerate(self.rois, 1):
            print(f"  {i:2d}. {r[2]:10} | Label: '{r[3]}' | Category: '{r[4]}'")
        try:
            idx = int(input("\nEnter ROI number to delete (0 to cancel): "))
            if 1 <= idx <= len(self.rois):
                removed = self.rois.pop(idx - 1)
                print(f"🗑️ Deleted ROI #{idx}")
                self._redraw_all_rois()
                self._save_latest()
        except ValueError:
            print("Invalid input.")

    def _edit_roi(self):
        if not self.rois:
            print("⚠️ No ROIs to edit.")
            return

        print("\nCurrent ROIs:")
        for i, r in enumerate(self.rois, 1):
            print(f"  {i:2d}. {r[2]:10} | Label: '{r[3]}' | Category: '{r[4]}'")

        try:
            idx = int(input("\nEnter ROI number to edit (0 to cancel): "))
            if idx == 0 or not (1 <= idx <= len(self.rois)):
                return

            roi = self.rois[idx - 1]
            (_, _, ftype, label, category) = roi

            print(f"\nEditing ROI #{idx} (Current: Type={ftype}, Label='{label}', Category='{category}')")

            # Re-select type
            choice_str = input("New type number (Enter to keep): ").strip()
            if choice_str:
                try:
                    ch = int(choice_str)
                    if ch == 0:
                        new_type = input("Custom type: ").strip() or "custom"
                    elif 1 <= ch <= len(self.field_types):
                        new_type = self.field_types[ch - 1]
                    else:
                        new_type = ftype
                except:
                    new_type = ftype
            else:
                new_type = ftype

            new_label = input(f"New label (current: '{label}'): ").strip()
            if not new_label:
                new_label = label

            new_category = input(f"New category (current: '{category}'): ").strip()
            if not new_category:
                new_category = category

            self.rois[idx - 1] = [roi[0], roi[1], new_type, new_label, new_category]
            print(f"✅ ROI #{idx} updated successfully.")
            self._redraw_all_rois()
            self._save_latest()
        except Exception as e:
            print(f"Error during edit: {e}")

    def _save_latest(self):
        with open("annotated_rois_latest.json", "w") as f:
            json.dump(self.rois, f, indent=2)

    def _save_with_timestamp(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"annotated_rois_{ts}.json"
        png_file = f"annotated_image_{ts}.png"

        with open(json_file, "w") as f:
            json.dump(self.rois, f, indent=2)

        # Save visual image
        vis = self.image.copy()
        for roi in self.rois:
            (x1, y1), (x2, y2), ftype, label, category = roi
            color = self._get_color(ftype)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            text = f"{ftype}: {label}" if label else ftype
            cv2.putText(vis, text, (x1, max(10, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if category and category != "Uncategorized":
                cv2.putText(vis, f"[{category}]", (x1, y2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)

        cv2.imwrite(png_file, vis)
        print(f"💾 Timestamped backup saved: {json_file} & {png_file}")

    def run(self):
        print("\n" + "="*95)
        print("🚀 FORM ROI ANNOTATOR with Category Support")
        print("="*95)
        print("Now each ROI includes: [ (x1,y1), (x2,y2), type, label, category ]")
        print("\nKeyboard Shortcuts:")
        print("  u = Undo last    |  d = Delete specific    |  e = Edit ROI")
        print("  c = Clear all    |  s = Manual save        |  l = List ROIs")
        print("  h = Help         |  ESC = Finish & Exit")
        print("="*95)

        cv2.namedWindow("Form ROI Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Form ROI Annotator", self.mouse_callback)

        while True:
            cv2.imshow("Form ROI Annotator", self.clone)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:        # ESC
                break
            elif key == ord('u'):
                self._undo_last()
            elif key == ord('d'):
                self._delete_specific()
            elif key == ord('e'):
                self._edit_roi()
            elif key == ord('c'):
                if input("Clear ALL ROIs? Type YES to confirm: ").strip().upper() == "YES":
                    self.rois.clear()
                    self.clone = self.image.copy()
                    self._redraw_all_rois()
                    self._save_latest()
            elif key == ord('s'):
                self._save_with_timestamp()
            elif key == ord('l'):
                print("\nCurrent ROIs:")
                for i, r in enumerate(self.rois, 1):
                    print(f"  {i:2d}. {r}")
            elif key == ord('h'):
                print("Use mouse + shortcuts as shown above.")

        cv2.destroyAllWindows()

        self._save_latest()
        self._save_with_timestamp()

        print("\n" + "="*95)
        print("ANNOTATION COMPLETED!")
        print("Final ROI (with category):")
        print("roi =", self.rois)
        print("\nSaved files: annotated_rois_latest.json + timestamped backups")
        print("   • Multiple timestamped backups + annotated images")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("   python form_roi_annotator.py <image_path> [existing_rois.json]")
        print("Example:")
        print("   python form_roi_annotator.py form_sample.jpg")
        print("   python form_roi_annotator.py form_sample.jpg annotated_rois_latest.json")
        sys.exit(1)

    image_path = sys.argv[1]
    load_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        annotator = FormROIAnnotator(image_path, load_path)
        annotator.run()
    except Exception as e:
        print(f"Error: {e}")