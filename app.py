import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time

# @st.cache_data(allow_output_mutation=True)
@st.cache_resource()
def get_predictor_model():
    from model import Model
    model = Model()
    return model


header = st.container()
model = get_predictor_model()


with header:
    st.title('Hello!')
    st.text(
        'Using this app you can classify whether there is fight on a street? or fire? or car crash? or everything is okay?')

mode = st.radio("Select mode:", ["Analyze Image", "Live Webcam"])

if mode == "Analyze Image":

    uploaded_file = st.file_uploader("Or choose an image...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = np.array(image)
        label_text = model.predict(image=image)['label'].title()
        st.write(f'Predicted label is: **{label_text}**')
        st.write('Original Image')
        if len(image.shape) == 3:
            cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image)



elif mode == "Live Webcam":
    st.text("Webcam mode: real-time predictions")
    frame_placeholder = st.empty()
    label_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    ANALYZE_INTERVAL = 0.5  
    last_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(frame_rgb, (320, 240))

            now = time.time()
            if now - last_time >= ANALYZE_INTERVAL:
                last_time = now
                result = model.predict(image=small_frame)
                label_text = result['label'].title()
                label_placeholder.markdown(f"**Predicted label:** {label_text}")

            frame_placeholder.image(frame_rgb, caption="Live Webcam", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cap.release()
