import streamlit as st
from PIL import Image
import numpy as np
from fer import FER

# Function to detect emotions
def detect_emotion(image):
    detector = FER(mtcnn=True)
    results = detector.detect_emotions(image)
    if results:
        return results[0]['emotions']
    else:
        return None

# Streamlit app
def main():
    st.title("Human Emotion Detection App")
    st.write("Upload an image and I'll detect the emotions!")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Detect emotions
        emotions = detect_emotion(np.array(image))
        if emotions:
            st.write("Emotions detected:")
            for emotion, score in emotions.items():
                st.write(f"- {emotion}: {score}")
        else:
            st.write("No faces found in the image. Please upload another image.")

if __name__ == "__main__":
    main()
