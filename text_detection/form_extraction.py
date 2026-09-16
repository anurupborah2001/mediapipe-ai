
import os
import json
import csv
from module.form_detector import FormROIDetector
from module.text_detector import TextDetector
from openvisionkit.capture.image_template import image_template
from openvisionkit.capture.video_template import video_capture_template
from util import utility
import numpy as np
import pytesseract
import pandas as pd
import cv2


detect_type = "IMAGE" 
pixel_threshold = 500
form_prefix="registration_form"
path, _= utility.get_calling_folder()
width, height = 480, 800
form_asset_path = f"{path}/assets/form"
main_form_path = f"{form_asset_path}/main_form.jpg"
filled_form_path = f"{form_asset_path}/filled"
list_filled_form = utility.get_valid_images(filled_form_path)

##Get the width , height and channels for the image
percentage_of_matches  = 25

form_roi_annotation_json = f"{path}/assets/annotated_form/annotated_rois_latest.json"
##Extract the roi
roi = []

with open(form_roi_annotation_json, "r") as f:
    roi = json.load(f)

def compare_matches(original_image, image2_path, form_name):
    # Read image2
    image2 = cv2.imread(image2_path)
    if image2 is None:
      raise ValueError("image2 could not be loaded")
    
    # Resize image2 to match image1 
    h, w = original_image.shape[:2]
    
    image2 = cv2.resize(image2, (w, h))
    
    text_form = TextDetector(original_image, preprocess=False)  
    transform_results = text_form.compare_matches_bf_matcher(image2, form_name=form_name, no_of_feature=2000, percentage_of_matches=percentage_of_matches, draw_matches=False, draw_aligned=False)

    if "aligned_image" in transform_results:
      scanned_image = transform_results["aligned_image"]
      scanned_image_copy = scanned_image.copy()
      mask_image = np.zeros_like(scanned_image)
      form_records = {} 
          
      for idx, roi_info in enumerate(roi):
        x, y, w, h = roi_info[0][0], roi_info[0][1], roi_info[1][0], roi_info[1][1]
        form_element = f"{roi_info[2]}"
        label_name =f"{roi_info[3]}"
        category_name = f"{roi_info[4]}"
        print("category_name", category_name)
        cv2.rectangle(mask_image, (x, y), (w, h), (0, 255, 0),  cv2.FILLED)
        scanned_image_copy = cv2.addWeighted(scanned_image_copy, 0.99, mask_image, 0.1, 0)
        crop_image = scanned_image[y:h, x:w]
        # cv2.imshow(f"{form_name} - Field {idx}: {text}", crop_image)
        extracted_text = ""
        
        ##TEXTBOX / DROPDOWN
        if form_element in ["textbox", "dropdown"]:
          extracted_text = pytesseract.image_to_string(crop_image)
          if extracted_text:
            form_records[category_name] = extracted_text
            
        ##RADIO BUTTON
        elif form_element == "radio":
          gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
          thresh = cv2.threshold(
              gray, 128, 255,
              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
          )[1]

          non_zero = cv2.countNonZero(thresh)
          is_checked = non_zero > pixel_threshold
          if is_checked:
              form_records[category_name] = label_name
              
        ##CHECKBOX
        elif form_element == "checkbox":
          gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
          thresh = cv2.threshold(
              gray, 128, 255,
              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
          )[1]

          non_zero = cv2.countNonZero(thresh)
          is_checked = non_zero > pixel_threshold

          if is_checked:
              if category_name not in form_records:
                  form_records[category_name] = []
              form_records[category_name].append(label_name)
          
        ##PARAGRAPH (Cursive)
        elif form_element == "paragraph":
          extracted_text = pytesseract.image_to_string(crop_image).strip()
          if extracted_text:
            form_records[category_name] = extracted_text
            
        ## Visualization
        display_text = form_records.get(category_name, "")
        cv2.putText(
            scanned_image,
            str(display_text),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

      ## Convert list values to string for CSV compatibility
      for key, value in form_records.items():
        if isinstance(value, list):
            form_records[key] = ", ".join(value)
            
      ## Add header to the csv file if it doesn't exist
      csv_path = f"{path}/{form_prefix}_extracted_data.csv"

      ## Load existing data (if any)
      rows = []
      existing_headers = []

      if os.path.exists(csv_path):
          with open(csv_path, "r", newline="") as f:
              reader = csv.DictReader(f)
              rows = list(reader)
              existing_headers = reader.fieldnames or []

      ## Merge headers
      final_headers = list(dict.fromkeys(existing_headers + list(form_records.keys())))

      ## Append new row
      rows.append(form_records)

      ## Rewrite CSV (simple + safe)
      with open(csv_path, "w", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=final_headers)
          writer.writeheader()
          writer.writerows(rows)
          
      scanned_image = cv2.resize(scanned_image, (w//2, h//2))
      cv2.imshow(f"{form_name} - Extracted Fields", scanned_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
  

def extract_form_fields(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for idx, filled_form in enumerate(list_filled_form):
    compare_matches(image, filled_form, form_name=f"Compare Image {idx}")    
  
  
  ## Find the keypoint of the main form and the filled form and match the keypoint to get the
  ## homography transformation matrix to align the filled form to the main form
  
  # text_detector = TextDetector(image)
  # keypoint, descriptors, image_with_keypoints = text_detector.detect_keypoints(features=600, draw_keypoints=True)
  # print(f"Extracted {len(keypoint)} keypoints from the form image.")
  # # print(keypoint)
  # # print(f"Keypoint descriptors: {descriptors}")
  
  # h,w,c = image.shape
  # image_with_keypoints = cv2.resize(image_with_keypoints, (w//4, h//4))
  # cv2.imshow("Keypoints", image_with_keypoints)
  
  return image

if __name__ == "__main__":
  if detect_type == "IMAGE":
    image_template(
        image_path=main_form_path,
        custom_logic=extract_form_fields,
        window_name="Text Character Detection - Image",
        show_window=True,
        center_window=False,
        resolution=(width, height),
    )
  else:
    # Example usage for video
    video_capture_template(
        video_source=0,  # Use 0 for webcam or provide path to video file
        custom_logic=extract_form_fields,
        window_name="Text Character Detection - Video"
    )