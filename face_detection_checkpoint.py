# Import required libraries
import cv2  # OpenCV for real-time computer vision tasks
import streamlit as st  # Streamlit for web app interface
from PIL import Image  # For handling image operations
import numpy as np  # For numerical computations
import os  # For interacting with the operating system
import time

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(r'C:/Users/HARDEY/Documents/GOMYCODE/Deep Learning/haarcascade_frontalface_default.xml')

# Step 1: Function to detect faces from webcam feed
def detect_faces(scaleFactor, minNeighbors, color_choice):
    cap = cv2.VideoCapture(0)  # Opens the default webcam (0 refers to the default device)
    stframe = st.empty()  # Placeholder for displaying the webcam video in Streamlit
    saved = False  # Initialize a saved flag for the save button functionality

    # Check if stop detection flag exists in session state, initialize if not
    if "stop_detection" not in st.session_state:
        st.session_state.stop_detection = False

    # Checkbox to stop face detection
    stop_detection = st.checkbox("Stop Face Detection", key="stop_detection_checkbox")

    while not st.session_state.stop_detection:  # Loop until stop detection is triggered
        ret, frame = cap.read()  # Capture each frame
        if not ret:  # If the frame is not captured successfully
            st.error("Failed to capture image from webcam.")  # Display an error message
            break
        
        # Convert the frame to grayscale (face detection works better on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))
        
        # Convert the selected color from hex format (RGB) to BGR for OpenCV
        color_rgb = tuple(int(color_choice.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Reverse the RGB to BGR

        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)  # Draw rectangle on each detected face
        
        # Display the current frame with detected faces in the Streamlit app
        stframe.image(frame, channels="BGR", use_column_width=True)
        
        # Create a unique key for each save button using the current timestamp
        button_key = f"save_image_{str(time.time())}"
        
        # Button to save the image with detected faces
        if st.button("Save Image with Detected Faces", key=button_key) and not saved:
            cv2.imwrite("detected_faces.png", frame)  # Save the current frame as an image
            st.success("Image saved as detected_faces.png")  # Display success message
            saved = True  # Set the flag to prevent multiple saves
        
        # Update the stop detection flag if the checkbox is checked
        if stop_detection:
            st.session_state.stop_detection = True

    cap.release()  # Release the webcam resource
    cv2.destroyAllWindows()  # Close any OpenCV windows

# Step 2: Define the Streamlit app interface
def app():
    st.title("Face Detection App using Viola-Jones Algorithm")  # Title of the app
    
    # Instructions for the user
    st.markdown("""
    ### Instructions:
    1. Use the **slider** below to adjust the `scaleFactor` and `minNeighbors` parameters for face detection.
    2. Choose the **color** of the rectangle that will be drawn around the detected faces.
    3. Press the **'Detect Faces'** button to start detecting faces using your webcam.
    4. Optionally, save the detected face image by pressing the **'Save Image with Detected Faces'** button.
    5. Use the checkbox to **stop detection**.
    """)
    
    # Slider for adjusting the `scaleFactor` (resizing of the image for detection)
    scaleFactor = st.slider("Adjust scaleFactor (Image resizing)", min_value=1.01, max_value=2.0, value=1.1, step=0.01)
    
    # Slider for adjusting `minNeighbors` (controls detection sensitivity)
    minNeighbors = st.slider("Adjust minNeighbors (Detection sensitivity)", min_value=3, max_value=10, value=5, step=1)
    
    # Color picker for the rectangle that will highlight detected faces
    color_choice = st.color_picker("Pick a color for the detection rectangle", "#FF0000")
    
    # Button to start face detection
    if st.button("Detect Faces", key="detect_faces"):
        detect_faces(scaleFactor, minNeighbors, color_choice)  # Call the face detection function with parameters

# Step 3: Run the Streamlit app
if __name__ == "__main__":  # If the script is run directly
    app()  # Run the app
