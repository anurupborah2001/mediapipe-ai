import mediapipe as mp
import cv2
import numpy as np
from module.selfie_segmentation import SelfieSegmentation
from openvisionkit.capture.video_template import video_capture_template


selfie_segmentation = SelfieSegmentation()
def replace_background_logic(frame):
    ## 1st Method to replace background using replace_background function
    # return selfie_segmentation.replace_background(frame, background_path="./selfie_segmentation/images/background/background-image1.jpg")
    
    ## 2nd Method method to replace background using optimized virtual background
    image =  cv2.imread("./selfie_segmentation/images/background/background-image2.jpg")
    return selfie_segmentation.optimize_virtual_background_improved(frame, bg=image)

    ## 3rd Method Using confidence for alpha blend
    image =  cv2.imread("./selfie_segmentation/images/background/background-image2.jpg")
    return selfie_segmentation.optimize_virtual_background(frame, bg=image)

if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=False,      
        custom_logic=replace_background_logic,
        window_name="Replace BackGround",
        resolution=(680, 320),
        center_window=True,
        draw_fps=True
    )
