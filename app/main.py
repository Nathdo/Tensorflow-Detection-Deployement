import streamlit as st 
from model.model import model, encode
import cv2
import numpy as np
from io import BytesIO

# --- Page setup ---
st.set_page_config(page_title="Apple Disease Detector", page_icon="🍎 ", layout="centered")
st.markdown("<h1 style='text-align: center;'>Apple Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("Upload an image of an apple leaf to detect possible diseases.", unsafe_allow_html=True)

# --- File uploader section ---
st.markdown("### Drag and drop or select an image file")
selected_file = st.file_uploader("", type=['JPG', 'PNG', 'JPEG'])
st.divider()

# --- Sample images section ---
st.markdown("### Or try with a sample image:")

col1, col2, col3 = st.columns(3)
sample_paths = {
    "Healthy Leaf": "app/Images/healthy.png",
    "Black Rot": "app/Images/test1.jpg",
    "Apple Scab": "app/Images/test2.jpg"
}

with col1:
    if st.button("🟢 Healthy"):
        selected_file = BytesIO(open(sample_paths["Healthy Leaf"], 'rb').read())
        selected_file.name = "healthy.png"
with col2:
    if st.button("⚠️ Black Rot"):
        selected_file = BytesIO(open(sample_paths["Black Rot"], 'rb').read())
        selected_file.name = "black_rot.jpg"
with col3:
    if st.button("🔴 Apple Scab"):
        selected_file = BytesIO(open(sample_paths["Apple Scab"], 'rb').read())
        selected_file.name = "apple_scab.jpg"


# --- Process and predict if file is uploaded or selected ---
if selected_file:
    st.divider()
    st.success("✅ File loaded successfully!")
    st.divider()

    def preprocessing(uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  
        return img

    def preprocess_predict(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255
        image = np.expand_dims(image, axis=0)
        return image

    def predict(image):
        probabilities = model.predict(image)[0]
        index_pred = np.argmax(probabilities)
        confidence = round(probabilities[index_pred] * 100, 2)
        label = encode.inverse_transform([index_pred])[0]

        if label == 'healthy':
            text = f"🟢 The apple leaf is **Healthy** ({confidence}%)."
        elif label == 'black_rot':
            text = f"⚠️ The predicted apple disease is **Black Rot** ({confidence}%)."
        elif label == 'apple_scab':
            text = f"🔴 The predicted apple disease is **Apple Scab** ({confidence}%)."
        else:
            text = f"🟠 The predicted apple disease is **Apple Rust** ({confidence}%)."

        return text, confidence

    # --- Prediction ---
    image_pixels = preprocessing(selected_file)
    image_pixels = preprocess_predict(image_pixels)
    prediction, confidence = predict(image_pixels)

    # --- Display results ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(selected_file, caption="📷 Uploaded Leaf", use_container_width=True)
    with col2:
        st.markdown(f"<h3 style='color:#4CAF50'>{prediction}</h3>", unsafe_allow_html=True)
        st.progress(confidence / 100)
