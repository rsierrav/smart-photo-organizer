import os
import shutil
import zipfile

def create_output_dirs():
    folders = [
        "output/duplicates",
        "output/blurry",
        "output/clusters"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def save_duplicates(duplicates, names, images, is_blurry_fn):
    for i, j, _ in duplicates:
        _, score_i = is_blurry_fn(images[i])
        _, score_j = is_blurry_fn(images[j])

        # Higher score = sharper image
        best_idx = i if score_i > score_j else j

        src = f"images/{names[best_idx]}"
        dst = f"output/duplicates/{names[best_idx]}"

        shutil.copy(src, dst)


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


# Optional: create a zip of the output folder for easy download
def zip_output():
    zip_path = "output.zip"

    with zipfile.ZipFile(zip_path, 'w') as z:
        for root, dirs, files in os.walk("output"):
            for file in files:
                full_path = os.path.join(root, file)
                z.write(full_path)

    return zip_path