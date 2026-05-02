import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Load pretrained model
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model_full = resnet18(weights=weights)
model_full.eval()

# Feature extractor (remove last layer)
feature_extractor = torch.nn.Sequential(*list(model_full.children())[:-1])
feature_extractor.eval()

# Get ImageNet categories
categories = weights.meta["categories"]

# Use the model's default transform
transform = weights.transforms()


def extract_features(images):
    features = []
    predictions = []

    for img in images:
        # Convert OpenCV (BGR) to RGB
        img_rgb = img[:, :, ::-1]
        
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img_rgb)
        
        # Apply transforms (handles normalization, etc.)
        input_tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            # Extract features
            feat = feature_extractor(input_tensor)
            # Get predictions
            output = model_full(input_tensor)

        features.append(feat.flatten().numpy())
        
        # Get the predicted class index
        pred_class = output.argmax(dim=1).item()
        predictions.append(pred_class)

    return np.array(features), predictions