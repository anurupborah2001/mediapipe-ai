import cv2
from module.image_detector import ImageDetector
from openvisionkit.capture.image_template import image_template
from openvisionkit.capture.video_template import video_capture_template
from util import utility


path, _= utility.get_calling_folder()
detect_type = "IMAGE"

def save_results_and_annotate(image, results, output_txt="output.txt", output_img="annotated.png"):
    annotated = image.copy()
    output_txt_path = f"{path}/results/{output_txt}"
    output_img_path = f"{path}/results/{output_img}"
    
    # 1. Write all extracted text into a file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            x1, y1, x2, y2 = r["bbox"]
            text = r["text"]

            f.write(f"[{i}] ({x1},{y1},{x2},{y2})\n{text}\n\n")

            # 2. Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 3. Put text ABOVE the box
            label_y = max(20, y1 - 10)

            cv2.putText(
                annotated,
                text[:50],  # truncate long text
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    # 4. Save annotated image
    cv2.imwrite(output_img_path, annotated)

    return annotated
  
def extract_highlighted_text(image_detector):
  ##1. It will find the dominant HSV colors which will provide the list of highlighted hsv colors in the image and then we can use this list to find the highlighted areas in the image
  dominant_hsv_colors = image_detector.get_dominant_hsv_colors(k=4)
  print("Dominant HSV Colors in the image:", dominant_hsv_colors)

  ##This method will find different highlighted areas based on the list of hsv colors provided and combine the mask to find all the highlighted areas in the image
  _, combined_mask, _  = image_detector.detect_highlighted_text(hsv_colors=dominant_hsv_colors, h_tol=7,s_tol=60, v_tol=55) ## We are setting show_image_with_mask to False because we want to use the combined mask for contour detection and text extraction, and we don't need to visualize it at this stage.
  print("Combined Mask Shape:", combined_mask.shape)
  ## 3. we need to find the contours of the highlighted areas and extract the contours co ordinates
  contours_results,img_contours = image_detector.find_contours(combined_mask, min_area=30, debug=True,draw_contours=True) ## This will give the sorted contours co ordinates based on the position in the image from top to bottom and left to right
  all_bounding_box = [c["bounding_box"] for c in contours_results]
  all_contours = [c["contour"] for c in contours_results]
  
  # cv2.imshow("Contours", img_contours)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  # print("Contours Co-ordinates:", all_contours)
  print("Bounding Box Co-ordinates:", all_bounding_box)
  ## 4. We need to crop the hightlighted areas using the contours co ordinates 
  
  results = image_detector.extract_text_regions(all_bounding_box)
  
  return results, all_bounding_box, combined_mask
  

def detect_highlighted_text(image):
  image_detector = ImageDetector(image)  ## We are setting preprocess to false because we want to use the original image for highlight detection and text extraction
  results, _, _ = extract_highlighted_text(image_detector)

  # Draw boxes
  annotated = save_results_and_annotate(
    image,
    results,
    output_txt="highlighted_text.txt",
    output_img="highlighted_annotated.png"
  )

  cv2.imshow("Annotated", annotated)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ## 5. We need to pass the crop image to pytesseract to extract the text 
  # 
  # 
  ## save the extracted text in the txt files
  return image_detector.image
    
    
if __name__ == "__main__":
  print("This module is not meant to be run directly. Please import and use the functions in your main application.") 
  if detect_type == "IMAGE":
    image_template(
        image_path=f"{path}/assets/images/highlight/highlight_text.png",
        custom_logic=detect_highlighted_text,
        window_name="Highlight Text Detector - Image",
        show_window=False  # We will show the window in the detect_highlighted_text function after processing
    )
  else:
    # Example usage for video
    video_capture_template(
        video_source=0,  # Use 0 for webcam or provide path to video file
        custom_logic=detect_highlighted_text,
        window_name="Highlight Text Detector - Video"
    )