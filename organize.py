import os
import shutil

def create_output_dirs():
    folders = [
        "output/duplicates",
        "output/blurry",
        "output/clusters"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def save_duplicates(duplicates, names):
    for i, j, _ in duplicates:
        shutil.copy(f"images/{names[i]}", f"output/duplicates/{names[i]}")
        shutil.copy(f"images/{names[j]}", f"output/duplicates/{names[j]}")


def save_blurry(images, names, is_blurry_fn):
    for i, img in enumerate(images):
        blurry, _ = is_blurry_fn(img)
        if blurry:
            shutil.copy(f"images/{names[i]}", f"output/blurry/{names[i]}")


def save_clusters(labels, names):
    for i, label in enumerate(labels):
        folder = f"output/clusters/group_{label}"
        os.makedirs(folder, exist_ok=True)

        shutil.copy(f"images/{names[i]}", f"{folder}/{names[i]}")