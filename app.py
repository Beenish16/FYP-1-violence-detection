import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Cache model loading
@st.cache_resource()
def get_predictor_model():
    from model import Model
    model = Model()
    return model


# Streamlit Header
header = st.container()
model = get_predictor_model()

with header:
    st.title("Violence Detection System")
    st.text(
        "Using this app you can classify whether there is a fight on a street, a fire, a car crash, or everything is okay!"
    )

# Mode Selection
mode = st.radio("Select mode:", ["Analyze Image", "Live Webcam"])

# -----------------------------
# 📸 Image Upload Mode
# -----------------------------
if mode == "Analyze Image":

    uploaded_file = st.file_uploader("Upload an image to analyze...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)

        # Prediction
        label_text = model.predict(image=image)["label"].title()
        st.write(f"**Predicted label:** {label_text}")

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

# -----------------------------
# 🎥 Live Webcam Mode
# -----------------------------
elif mode == "Live Webcam":
    st.text("Webcam mode: real-time predictions")
    frame_placeholder = st.empty()
    label_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Try changing camera index (0, 1, or 2).")

    ANALYZE_INTERVAL = 0.5  
    last_time = 0.0

    import os
    from datetime import datetime

    # Directory to save detected event frames
    save_dir = "detected_events"
    os.makedirs(save_dir, exist_ok=True)

    # All classes you want to save
    danger_labels = [
        "violence",
        "street violence",
        "fighting violence",
        "fire",
        "car crash",
        "robbery",
        "earthquake"
    ]

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("No frame captured. Please check your webcam connection.")
                continue

            current_time = time.time()
            if current_time - last_time >= ANALYZE_INTERVAL:
                last_time = current_time

                # Prediction
                result = model.predict(image=frame)
                label_text = result['label'].title()

                # Timestamp overlay
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                overlay_text = f"{label_text} - {timestamp}" 

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                color = (255, 255, 255)

                text_size = cv2.getTextSize(overlay_text, font, font_scale, font_thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 50

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], text_y + 15), (0, 0, 0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, overlay_text, (text_x, text_y), font, font_scale, color, font_thickness)

                # Show frame and prediction
                label_placeholder.text(f"Prediction: {label_text}")
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                # Save frame if danger detected
                if label_text.lower() in danger_labels:
                    sub_dir = os.path.join(save_dir, label_text.lower().replace(" ", "_"))
                    os.makedirs(sub_dir, exist_ok=True)

                    filename = f"{label_text.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    filepath = os.path.join(sub_dir, filename)
                    cv2.imwrite(filepath, frame)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
