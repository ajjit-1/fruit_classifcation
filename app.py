import streamlit as st
from utils.predict import load_model, predict_image

# ----- App Title -----
st.title("🍎 Fruit Classification App")
st.write("Upload an image and the model will classify the fruit.")

# ----- Set Your Class Names -----
class_names = ["Apple", "Banana", "cherry", "strawberry"]  # change according to your dataset

# ----- Model Path -----
MODEL_PATH = "model/fruits_efficientnet_b0.pth"

# Load model once
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# ----- File Upload -----
uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        pred = predict_image(model, uploaded_file, class_names)
        st.success(f"Predicted Fruit: **{pred}** 🎉")
