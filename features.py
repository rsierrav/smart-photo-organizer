import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

# CLIP zero-shot labels for common photo categories
categories = [
    "a person",
    "a selfie",
    "people at a dinner",
    "a group of people",
    "a wedding",
    "an event",
    "a dog",
    "a cat",
    "food",
    "a restaurant",
    "a landscape",
    "a beach",
    "a sunset",
    "a building",
    "a car",
    "a blurry photo",
]

prompt_template = "a photo of {}"


def extract_features(images):
    features = []
    predictions = []

    for img in images:
        # Convert OpenCV (BGR) to RGB
        img_rgb = img[:, :, ::-1]
        
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img_rgb)

        prompts = [prompt_template.format(label) for label in categories]
        inputs = processor(text=prompts, images=img_pil, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLIP image embeddings are good feature vectors for clustering.
        features.append(outputs.image_embeds.squeeze(0).cpu().numpy())

        # Use the highest-scoring text prompt as the predicted label.
        pred_class = outputs.logits_per_image.softmax(dim=1).argmax(dim=1).item()
        predictions.append(pred_class)

    return np.array(features), predictions


# BLIP captioning helpers
blip_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
blip_model.eval()


def generate_caption(img):
    img_rgb = img[:, :, ::-1]
    img_pil = Image.fromarray(img_rgb)

    inputs = blip_processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        out_ids = blip_model.generate(**inputs)

    # decode to string
    caption = blip_processor.decode(out_ids[0], skip_special_tokens=True)
    return caption


def get_cluster_representatives(features_array, labels, top_k=3):
    reps = {}
    features_array = np.asarray(features_array)
    labels = np.asarray(labels)

    for cluster_id in np.unique(labels):
        idxs = np.where(labels == cluster_id)[0]
        cluster_feats = features_array[idxs]
        centroid = cluster_feats.mean(axis=0)

        # distances to centroid
        dists = np.linalg.norm(cluster_feats - centroid, axis=1)
        order = np.argsort(dists)
        selected = idxs[order[:min(top_k, len(order))]].tolist()
        reps[cluster_id] = selected

    return reps


def summarize_captions(captions, top_n=3):
    if not captions:
        return "Miscellaneous Photos"

    # If all captions identical, return that cleaned caption
    if len(set(captions)) == 1:
        best = captions[0]
    else:
        # Prefer captions that appear more than once; if all unique, pick longest (most descriptive)
        counts = Counter(captions)
        most_common, freq = counts.most_common(1)[0]
        if freq > 1:
            best = most_common
        else:
            best = max(captions, key=len)

    def clean_caption(text):
        text = (text or "").lower()
        for phrase in ["a photo of", "an image of", "a picture of", "photo of", "image of", "picture of"]:
            text = text.replace(phrase, "")
        text = text.strip()
        if not text:
            return ""
        return text[0].upper() + text[1:]

    label = clean_caption(best)
    if not label or len(label) < 5:
        return "Miscellaneous Photos"
    return label