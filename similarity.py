from sklearn.metrics.pairwise import cosine_similarity

def find_duplicates(features, threshold=0.9):
    sim_matrix = cosine_similarity(features)

    duplicates = []

    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i][j] > threshold:
                duplicates.append((i, j, sim_matrix[i][j]))

    return duplicates