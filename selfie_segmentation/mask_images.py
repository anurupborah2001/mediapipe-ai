import mediapipe as mp
import cv2
import numpy as np
from module.selfie_segmentation import SelfieSegmentation
from openvisionkit.capture.video_template import video_capture_template

panel_name = "Mask Images"
MASK=8

selfie_segmentation = SelfieSegmentation()
def mask_image_logic(frame):
    global panel_name
    if MASK == 0:
        panel_name = "Extract Foreground"
        return selfie_segmentation.extract_foreground(frame)
    elif MASK == 1:
        panel_name = "Overlay Mask"
        return selfie_segmentation.overlay_mask(frame)
    elif MASK == 2:
        panel_name = "Threshold Mask"
        return selfie_segmentation.threshold_mask(frame)
    elif MASK == 3:
        panel_name = "Alpha Blend"
        image =  cv2.imread("./selfie_segmentation/images/background/background-image2.jpg")
        return selfie_segmentation.alpha_blend(frame, image)
    elif MASK == 4:
      panel_name = "Using Confidence for Alpha Blend"
      image =  cv2.imread("./selfie_segmentation/images/background/background-image2.jpg")
      return selfie_segmentation.confidence_alpha_blend(frame, bg=image)  
    elif MASK == 5:
        panel_name = "Morphological Segmentation"
        image =  cv2.imread("./selfie_segmentation/images/background/background-image2.jpg")
        return selfie_segmentation.morphological_segmentation(frame, bg=image) 
    elif MASK == 6:
        panel_name = "Blur Background"
        return selfie_segmentation.blur_background2(frame)
    elif MASK == 7:
        panel_name = "Layered Background"
        bg1 = cv2.imread("./selfie_segmentation/images/background/background-image1.jpg")
        bg2 = cv2.imread("./selfie_segmentation/images/background/background-image2.jpg")
        return selfie_segmentation.layered_background(frame, bg1, bg2)  
    else:
        panel_name = "Optimize Virtual Background"
        image =  cv2.imread("./selfie_segmentation/images/background/background-image2.jpg")
        return selfie_segmentation.optimize_virtual_background(frame, bg=image)
    
if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=False,      
        custom_logic=mask_image_logic,
        window_name=panel_name,
        resolution=(680, 320),
        center_window=True,
        draw_fps=True
    )
