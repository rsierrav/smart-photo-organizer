from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def cluster_images(features, k=5):
    features = normalize(features)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels