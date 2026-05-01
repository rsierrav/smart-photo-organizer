import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

# Load pretrained model
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(images):
    features = []

    for img in images:
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            feature = model(img_t)

        features.append(feature.flatten().numpy())

    return np.array(features)