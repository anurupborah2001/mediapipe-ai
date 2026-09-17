import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

class HairSegmentation:
  
  def __init__(
      self,
      model_path: str = "./models/hair_segmenter.tflite",
      output_category_mask: bool = True,
      output_confidence_masks: bool = False,
      running_mode: VisionTaskRunningMode = VisionTaskRunningMode.IMAGE,
  ):
    base_options = BaseOptions(model_asset_path=model_path)

    self.options = mp.tasks.vision.ImageSegmenterOptions(
          base_options=base_options,
          output_category_mask=output_category_mask,
          output_confidence_masks=output_confidence_masks,
          running_mode=running_mode
    )
    
    self.segmentor = vision.ImageSegmenter.create_from_options(self.options)
       
  def process(self, image: np.ndarray):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    return self.segmentor.segment(mp_image)
  
  def _get_mask(self, result, smooth=True):
    mask = result.category_mask.numpy_view()
    mask = (mask * 255).astype(np.uint8)
    return mask 
  
  def detect(self, image: np.ndarray, smooth=True, mask_color=(255, 0, 255)):
    result = self.process(image)
    mask =  self._get_mask(result, smooth)
    if smooth:
      _, mask = cv2.threshold(
          mask,
          1,
          255,
          cv2.THRESH_BINARY,
      )
    overlay = image.copy()
    overlay[mask > 0] = mask_color
    result_frame = cv2.addWeighted(
        overlay,
        0.4,
        image,
        0.6,
        0,
    )
    return result_frame
  