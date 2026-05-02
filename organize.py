import os
import shutil
import zipfile
import re
from collections import defaultdict, deque

def create_output_dirs():
    folders = [
        "output/keep",
        "output/trash",
        "output/organized",
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
        folder = f"output/organized/group_{label}"
        os.makedirs(folder, exist_ok=True)

        shutil.copy(f"images/{names[i]}", f"{folder}/{names[i]}")


# Zip the output folder for easy download
def zip_output():
    zip_path = "output.zip"

    with zipfile.ZipFile(zip_path, 'w') as z:
        for root, dirs, files in os.walk("output"):
            for file in files:
                full_path = os.path.join(root, file)
                z.write(full_path)

    return zip_path


def _safe_folder_name(name, max_len=40):
    # Remove unsafe characters, trim length, and replace spaces with underscores
    name = name.strip()
    name = name.replace("/", " ")
    name = re.sub(r"[^0-9A-Za-z _-]", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    name = name.replace(" ", "_")
    if not name:
        return "misc"
    return name


def _build_duplicate_groups(duplicates, n):
    # Given a list of duplicate pairs (i, j), build groups of connected images.
    # Build graph and find connected components
    adj = [[] for _ in range(n)]
    for i, j, _ in duplicates:
        adj[i].append(j)
        adj[j].append(i)

    seen = [False] * n
    groups = []
    for i in range(n):
        if seen[i]:
            continue
        if not adj[i]:
            continue
        # BFS component
        q = deque([i])
        comp = []
        seen[i] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        groups.append(comp)
    return groups


def save_organized(duplicates, names, images, is_blurry_fn, labels, cluster_names):
    create_output_dirs()

    n = len(names)

    # mark blurry images
    blurry_flags = [False] * n
    blur_scores = [0.0] * n
    for i, img in enumerate(images):
        blurry, score = is_blurry_fn(img)
        blurry_flags[i] = blurry
        blur_scores[i] = score

    # handle duplicates: group connected components
    dup_groups = _build_duplicate_groups(duplicates, n)

    kept = set()
    trashed = set()

    # process duplicate groups
    for group in dup_groups:
        # pick best image by highest blur score (least blurry) to keep; trash the rest
        best = max(group, key=lambda idx: blur_scores[idx])
        kept.add(best)
        for idx in group:
            if idx != best:
                trashed.add(idx)

    # process remaining images
    for i in range(n):
        if i in kept or i in trashed:
            continue
        if blurry_flags[i]:
            trashed.add(i)
        else:
            kept.add(i)

    # copy files
    for idx in kept:
        src = os.path.join("images", names[idx])
        dst = os.path.join("output", "keep", names[idx])
        shutil.copy(src, dst)

    for idx in trashed:
        src = os.path.join("images", names[idx])
        dst = os.path.join("output", "trash", names[idx])
        shutil.copy(src, dst)

    # organized clusters
    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(i)

    for label, indices in cluster_map.items():
        raw_name = cluster_names.get(label, f"group_{label}")
        # if most are blurry, rename as 'Blurry Images'
        blurry_count = sum(1 for i in indices if blurry_flags[i])
        if blurry_count > len(indices) / 2:
            folder_name = _safe_folder_name("Blurry Images")
        else:
            folder_name = _safe_folder_name(raw_name)

        folder_path = os.path.join("output", "organized", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for i in indices:
            # copy kept images into organized folder; also copy trashed images if desired
            src = os.path.join("images", names[i])
            dst = os.path.join(folder_path, names[i])
            shutil.copy(src, dst)