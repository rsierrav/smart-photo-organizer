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


def _is_blurry_cluster_name(name):
    return "blurry" in str(name).lower()


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
    # Clear output folder before saving
    if os.path.exists("output"):
        shutil.rmtree("output")
    
    create_output_dirs()

    n = len(names)

    # mark blurry images and compute blur scores
    blurry_flags = [False] * n
    blur_scores = [0.0] * n
    for i, img in enumerate(images):
        blurry, score = is_blurry_fn(img)
        blurry_flags[i] = blurry
        blur_scores[i] = score

    # Treat the cluster prediction as another blur signal. This catches images
    # that CLIP/BLIP grouped as blurry even when the Laplacian threshold misses.
    cluster_blurry_flags = [False] * n
    for i, label in enumerate(labels):
        cluster_blurry_flags[i] = _is_blurry_cluster_name(cluster_names.get(label, ""))

    trash_reason_flags = [
        blurry_flags[i] or cluster_blurry_flags[i]
        for i in range(n)
    ]

    # handle duplicates: group connected components
    dup_groups = _build_duplicate_groups(duplicates, n)

    kept = set()
    trashed = set()

    # process duplicate groups and keep sharpest image, trash the rest
    for group in dup_groups:
        keepable = [idx for idx in group if not trash_reason_flags[idx]]

        # If every duplicate is blurry or in a blurry cluster, trash the group.
        if not keepable:
            for idx in group:
                trashed.add(idx)
        else:
            # pick best keepable image by highest blur score (least blurry)
            best = max(keepable, key=lambda idx: blur_scores[idx])
            kept.add(best)
            for idx in group:
                if idx != best:
                    trashed.add(idx)

    # process remaining non-duplicate images
    for i in range(n):
        if i in kept or i in trashed:
            continue
        # If image is blurry or belongs to a blurry cluster, it goes to trash
        if trash_reason_flags[i]:
            trashed.add(i)
        else:
            kept.add(i)

    # Ensure no blurry or blurry-cluster images remain in kept
    for idx in list(kept):
        if trash_reason_flags[idx]:
            kept.remove(idx)
            trashed.add(idx)

    # copy files to "keep" folder (only non-trash images)
    for idx in kept:
        src = os.path.join("images", names[idx])
        dst = os.path.join("output", "keep", names[idx])
        shutil.copy(src, dst)

    # copy files to "trash" folder (duplicates, blurry images, and blurry clusters)
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

        # Only organize images that were kept. Fully blurry clusters live in trash.
        kept_indices = [i for i in indices if i in kept]
        if not kept_indices:
            continue

        folder_name = _safe_folder_name(raw_name)

        folder_path = os.path.join("output", "organized", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Only copy non-trash images into organized folders
        for i in kept_indices:
            src = os.path.join("images", names[i])
            dst = os.path.join(folder_path, names[i])
            shutil.copy(src, dst)

    # Return count of images moved to trash because of blur detection or blur labels.
    blurry_count = sum(1 for i in range(n) if trash_reason_flags[i])
    return blurry_count
