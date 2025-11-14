import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_model(model_path):
    """
    Loads EfficientNet-B0 and then loads state_dict (.pth)
    """

    num_classes = 4   # <-- change based on your dataset

    # Load architecture
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Load state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.eval()
    return model


def predict_image(model, image_file, class_names):
    """
    Predict fruit class.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]
