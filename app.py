# app.py
import streamlit as st
import tempfile
from predict import predict
from PIL import Image

st.set_page_config(page_title="Mango Leaf Disease Detector")

st.title("🍃 Mango Leaf Disease Classifier")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Submit"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        with st.spinner("Predicting..."):
            try:
                result = predict(tmp_path)
                st.success(f"🌿 Disease Predicted: **{result}**")
            except Exception as e:
                st.error(f"❌ Error during prediction:\n{str(e)}")