# Explicit Content Detection

This project is an image-based explicit content detection system using a custom-trained YOLOv9 model. The application provides a user-friendly web interface built with Streamlit, allowing users to upload images and receive instant feedback on whether explicit content is detected.

Streamlit App: https://explicit-content-detection.streamlit.app/

## Features
- **Image Upload:** Users can upload JPG or PNG images for analysis.
- **Object Detection:** Utilizes a YOLOv9 model to detect explicit content or custom objects in images.
- **Visual Feedback:** Detected objects are highlighted with bounding boxes and labels.
- **Alert System:** The app displays a warning if explicit content is detected, or a success message if the image is safe.
- **Model File Check:** If the model file is missing, the app warns the user and prevents further processing.

## Project Structure
```
Explicit_Content_Detection/
├── Model/
│   └── best.pt           # Custom YOLOv9 model weights (not included)
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd Explicit_Content_Detection
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.8+ installed. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the Model File:**
   - Place your custom-trained YOLOv9 model file (`best.pt`) in the `Model/` directory.
   - If you do not have a model, the app will display a warning and will not process images.

4. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Usage:**
   - Open the provided local URL in your browser.
   - Upload an image (JPG or PNG).
   - The app will display the original and detected images, and show an alert if explicit content is found.

## Dependencies
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/) (optional, for future extensions)

Install all dependencies using the provided `requirements.txt` file.

## Notes
- The model file (`Model/best.pt`) is not included. You must provide your own trained model.
- The detection classes and performance depend on how the YOLOv9 model was trained.

## License
This project is for educational and research purposes. Please check the licenses of the dependencies and your model before deploying in production. 