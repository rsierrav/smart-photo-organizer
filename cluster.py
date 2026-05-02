from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

def cluster_images(features, k=5):
    features = normalize(features)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def find_best_k(features, k_range=(3, 10)):
    best_k = None
    best_score = -1
    
    # Normalize features for clustering
    features_norm = normalize(features)

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_norm)

        # silhouette score measures cluster quality (higher is better)
        score = silhouette_score(features_norm, labels)

        print(f"k={k}, silhouette_score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest k: {best_k} (score={best_score:.4f})")

    return best_k