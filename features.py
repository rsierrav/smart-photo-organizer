import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter
import re

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


def _simplify_repetition(text: str) -> str:
    # simplify repetition
    parts = re.split(r"\s*(?:,| and |; )\s*", text.lower())
    seen = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p not in seen:
            seen.append(p)
    if len(seen) == 1:
        return seen[0]
    return " and ".join(seen)


def _shorten_phrase(text: str) -> str:
    # shorten wordy phrases
    replacements = [
        (r"on the shore of a body of water", "by the water"),
        (r"on the shore of the body of water", "by the water"),
        (r"on the shore of the", "by the water"),
        (r"on the shore", "by the water"),
        (r"a small town", "town"),
        (r"a small village", "village"),
        (r"in the background", ""),
        (r"in the distance", ""),
        (r"close up of", "close-up"),
    ]
    s = text.lower()
    for pat, repl in replacements:
        s = re.sub(pat, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _collapse_duplicate_words(text: str) -> str:
    # replace repeated adjacent words
    return re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)


def _remove_repeated_phrases(text: str) -> str:
    # remove repeated phrases
    text = text.lower()
    # pattern to match "X and a X" or "X and X"
    pattern = r'(\w+(?:\s+\w+)*?)\s+and\s+(?:a|an)?\s*\1'
    iteration = 0
    while re.search(pattern, text) and iteration < 5:
        text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
        iteration += 1
    return text


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

    # Clean and simplify the chosen caption
    text = (best or "").strip()
    # remove common prefixes
    for phrase in ["a photo of", "an image of", "a picture of", "photo of", "image of", "picture of"]:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    text = text.strip()

    # remove repetition
    text = _simplify_repetition(text)

    # collapse duplicate adjacent words
    text = _collapse_duplicate_words(text)

    # remove repeated phrases
    text = _remove_repeated_phrases(text)

    # shorten wordy phrases
    text = _shorten_phrase(text)

    text = text.strip()
    if not text:
        return "Miscellaneous Photos"

    # Capitalize first letter and limit length
    label = text[0].upper() + text[1:]
    if len(label) > 60:
        # heuristic shortening
        parts = re.split(r",| by | near | on ", label)
        label = parts[0].strip()

    # Force 'Blurry Images' if label itself contains 'blurry'
    if 'blurry' in label.lower():
        return "Blurry Images"

    if len(label) < 3:
        return "Miscellaneous Photos"
    return label