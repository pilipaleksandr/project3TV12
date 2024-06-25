import cv2
import numpy as np

# config paths
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model_path = 'frozen_inference_graph.pb'
detection_model = cv2.dnn_DetectionModel(model_path, config_path)

# reading labels/
labels = []
labels_file = 'labels.txt'
with open(labels_file, 'rt') as file:
    labels = file.read().rstrip("\n").split("\n")

# setup model
detection_model.setInputSize(320, 320)
detection_model.setInputScale(1.0 / 127.5)
detection_model.setInputMean((127.5, 127.5, 127.5))
detection_model.setInputSwapRB(True)

def get_color_by_id(class_id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    return colors[class_id % len(colors)]

# connect to camera
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise IOError("Проблема с подключением к камере")

scale_factor = 3
font_style = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = video_capture.read()
    class_indices, confidences, bounding_boxes = detection_model.detect(frame, confThreshold=0.55)

    if isinstance(class_indices, np.ndarray):
        for class_idx, conf, box in zip(class_indices.flatten(), confidences.flatten(), bounding_boxes):
            if 0 < class_idx <= len(labels):
                color = get_color_by_id(class_idx)
                print(f'Label: {labels[class_idx - 1]}, Confidence: {conf}')
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, labels[class_idx - 1], (box[0] + 10, box[1] + 40), font_style, fontScale=scale_factor, color=(0, 255, 0), thickness=2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0xff == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
