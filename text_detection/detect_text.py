from module.text_detector import TextDetector
from openvisionkit.capture.image_template import image_template
from openvisionkit.capture.video_template import video_capture_template
from util import utility

path, _= utility.get_calling_folder()

detect_type = "IMAGE"  # Change to "VIDEO" for video detection

def detect_characters(image):
  text_detector = TextDetector(image)
  bounding_boxes, annotated_image = text_detector.detect_characters(draw_boxes=True)
  return annotated_image

def detect_words(image):
  text_detector = TextDetector(image)
  bounding_boxes, annotated_image = text_detector.detect_words(draw_boxes=True)
  
  ## Entities extraction and PDF generation examples (uncomment if needed)
  # entities = text_detector.extract_entities()
  # pdf = text_detector.image_to_pdf_or_hocr("pdf")
  
  words = text_detector.get_words()
  print("Detected words:", words)
  return annotated_image  
    

if __name__ == "__main__":
    if detect_type == "IMAGE":
      image_template(
          image_path=f"{path}/assets/images/cards/card4.png",
          custom_logic=detect_words,
          window_name="Text Character Detection - Image"
      )
    else:
      # Example usage for video
      video_capture_template(
          video_source=0,  # Use 0 for webcam or provide path to video file
          custom_logic=detect_words,
          window_name="Text Character Detection - Video"
      )