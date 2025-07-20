# Import the Libraries
import cv2
import pandas as pd
from ultralytics import YOLO
import streamlit as st
from io import BytesIO
import numpy as np

# Load your custom YOLOv9 model
model = None
model_load_error = None
try:
    model = YOLO('Model/best.pt')
except Exception as e:
    model_load_error = str(e)

# Function for prediction using YOLOv9 model
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

# Function for drawing detections and bounding boxes on the image
def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    img_copy = img.copy()  # Create a copy of the original image
    results = predict(chosen_model, img_copy, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img_copy, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img_copy, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    return img_copy, results

# Streamlit app title
st.title("Upload the Image for Detections!!")

# Show warning if model is not loaded
if model_load_error or model is None:
    st.warning("Model file not found or failed to load. Please add a YOLO model (Model/best.pt) that detects explicit images.")
    st.stop()

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the image file
    image_bytes = uploaded_file.getvalue()
    orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    result_img, results = predict_and_detect(model, orig_image, classes=[], conf=0.5)

    # Check if any objects are detected
    detected_classes = []
    for result in results:
        for box in result.boxes:
            detected_classes.append(result.names[int(box.cls[0])])
    
    # Display the original image
    st.subheader("Original Image")
    st.image(orig_image, caption='Original Image', use_container_width=True)

    # Display the detected image
    st.subheader("Detected Objects")
    st.image(result_img, caption='Detected Objects', use_container_width=True)

    # Generate an alert message based on detection
    if detected_classes:
        st.error(f"Alert! Detected: {', '.join(detected_classes)}")
    else:
        st.success("Safe! No custom objects detected.")
