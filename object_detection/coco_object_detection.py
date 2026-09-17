import cv2
from openvisionkit.capture.image_template import image_template
from openvisionkit.capture.video_template import video_capture_template
from util import utility

path, _= utility.get_calling_folder()

coco_label_files = f"{path}/../models/coco.names"

detect_source="VIDEO"

with open(coco_label_files, "r", encoding="utf-8") as f:
    coco_labels = [line.strip() for line in f.readlines() if line.strip()]
print(f"COCO labels loaded | {len(coco_labels)} labels")

config_path = f"{path}/../models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = f"{path}/../models/frozen_inference_graph.pb"

object_detection_threshold = 0.5
nms_threshold = 0.2

cv2.dnn_DetectionModel
net = cv2.dnn_DetectionModel(weights_path,config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)  # Convert BGR to RGB


def coco_object_detection_image_logic(frame):
  class_ids, confidences, boxes = net.detect(frame, confThreshold=object_detection_threshold)
  if len(class_ids) != 0:
    for class_id, confidence, box in zip(class_ids, confidences, boxes):
      label = coco_labels[class_id - 1]
      cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
      cv2.putText(frame, f"{label.upper()}: {confidence:.2f}", (box[0], box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 2)
  return frame


def coco_object_detection_video_logic(frame):
  class_ids, confidences, boxes = net.detect(frame, confThreshold=object_detection_threshold)
  indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=object_detection_threshold, nms_threshold=nms_threshold)
  print(indices)
  for i in indices:
    box = boxes[i]
    confidence = confidences[i]
    class_id = class_ids[i]
    label = coco_labels[class_id - 1]
    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, f"{label.upper()}: {confidence:.2f}", (box[0], box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 2)  
  return frame

if __name__ == "__main__":
  if detect_source == "IMAGE":
    image_template(
      image_path=f"{path}/assets/image/coco_test_image.jpg",
      custom_logic=coco_object_detection_image_logic,
      window_name="COCO Object Detection"
    )
  else:
    video_capture_template(
      video_source=0,
      custom_logic=coco_object_detection_video_logic,
      window_name="COCO Object Detection",
    )
