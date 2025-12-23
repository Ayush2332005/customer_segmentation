from sklearn.cluster import KMeans

def train_kmeans(X_scaled, k=4):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels
