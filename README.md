## Fruit Classification using EfficientNet-B0

A complete end-to-end fruit classification project using PyTorch, EfficientNet-B0, and a Streamlit-based inference app.

## This project allows you to:

Train a fruit classifier

Evaluate model performance

Run a Streamlit app for real-time fruit prediction

Upload images and get instant model output

## Project Structure
fruits-classification/
│
├── model/
│   └── fruits_efficientnet_b0.pth
│
├── app.py
│
├── utils/
│   └── predict.py
│
├── requirements.txt
└── README.md

## 1. Create Virtual Environment (venv)
✅ Step 1 — Create venv
python -m venv venv

✅ Step 2 — Activate venv
Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

## 2. Install Dependencies
pip install -r requirements.txt


If you don't have a file yet:

torch
torchvision
streamlit
pillow

## 3. Download Dataset (Roboflow)

Inside your training notebook:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("mypj").project("fruits-kdfh9")
version = project.version(1)
dataset = version.download("folder")   # recommended for classification


This downloads folders:

train/
valid/
test/

## 4. Train the Model (Jupyter Notebook)

You trained your model with EfficientNet-B0:

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# after training
torch.save(model.state_dict(), "model/fruits_efficientnet_b0.pth")

## 5. Run Streamlit Inference App
streamlit run app.py


App will open at:

http://localhost:8501


Features:

Upload any fruit image

Model returns predicted class

Runs entirely on CPU

## 6. How Prediction Works

predict.py loads the model:

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("model/fruits_efficientnet_b0.pth"))
model.eval()

## 7. Supported Fruit Classes

Update in your app.py:

class_names = ["Apple", "Banana", "Grapes", "Orange"]

🚀 Future Improvements

Add confidence score

Add webcam support in Streamlit

Deploy on Streamlit Cloud

Add Grad-CAM visualization for model explainability

### Author

Ajit Kumar
Fruit Classification Project
Machine Learning | Deep Learning | Computer Vision