
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from openvino.runtime import Core
# from IPython.display import display, clear_output

# # Initialize OpenVINO core and read the model
# core = Core()
# # detection_model_xml = "model_2022_3/person-detection-retail-0013.xml" 
# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
#  # Update the path accordingly
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Image path
# image_path = 'C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/videos/videos/image8.jpg'

# # Read the image
# frame = cv2.imread(image_path)
# if frame is None:
#     print("Error: Could not read the image.")
#     exit()

# #

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# # Preprocess the frame
# resized_image = cv2.resize(src=frame, dsize=(W, H))
# input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

# # Perform inference
# request = compiled_model.create_infer_request()
# request.infer(inputs={input_layer.any_name: input_data})
# result = request.get_output_tensor(output_layer.index).data

# # Process detection results
# frame_height, frame_width = frame.shape[:2]
# for detection in result[0][0]:
#     conf = float(detection[2])
#     if conf > 0.76:
#         xmin = int(detection[3] * frame_width)
#         ymin = int(detection[4] * frame_height)
#         xmax = int(detection[5] * frame_width)
#         ymax = int(detection[6] * frame_height)
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#         cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# # Display the frame
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.title('Person Detection Demo')
# plt.axis('off')
# plt.show()













# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from openvino.runtime import Core
# from IPython.display import display, clear_output

# # Initialize OpenVINO core and read the model
# core = Core()
# # detection_model_xml = "model_2022_3/person-detection-retail-0013.xml" 
# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
#  # Update the path accordingly
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Image path
# image_path = 'C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/videos/videos/image3.jpg'

# # Read the image
# frame = cv2.imread(image_path)
# if frame is None:
#     print("Error: Could not read the image.")
#     exit()

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# # Preprocess the frame
# resized_image = cv2.resize(src=frame, dsize=(W, H))
# input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

# # Perform inference
# request = compiled_model.create_infer_request()
# request.infer(inputs={input_layer.any_name: input_data})
# result = request.get_output_tensor(output_layer.index).data

# # Process detection results
# frame_height, frame_width = frame.shape[:2]
# num_people = 0
# for detection in result[0][0]:
#     conf = float(detection[2])
#     if conf > 0.76:
#         xmin = int(detection[3] * frame_width)
#         ymin = int(detection[4] * frame_height)
#         xmax = int(detection[5] * frame_width)
#         ymax = int(detection[6] * frame_height)
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#         cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#         num_people += 1

# # Display the frame with count
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.title(f'Person Detection Demo - Number of People: {num_people}')
# plt.axis('off')
# plt.show()











import cv2
import numpy as np
import os
from openvino.inference_engine import IECore

# Initialize OpenVINO core and read the model
ie = IECore()
detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
detection_model_bin = os.path.splitext(detection_model_xml)[0] + ".bin"
net = ie.read_network(model=detection_model_xml, weights=detection_model_bin)
exec_net = ie.load_network(network=net, device_name="CPU", num_requests=1)

# Open video file
video_path = ''  # Update with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize variables
num_people_total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
    input_data = {'data': blob}

    # Perform inference
    res = exec_net.infer(inputs=input_data)

    # Process detection results
    for detection in res['detection_out'][0][0]:
        conf = detection[2]
        if conf > 0.5:  # Adjust confidence threshold as needed
            num_people_total += 1

    # Display current frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Total number of people detected: {num_people_total}")
