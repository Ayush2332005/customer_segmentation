import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.clustering import train_kmeans
from src.evaluation import evaluate_model
from src.visualization import plot_elbow, plot_clusters


if __name__ == "__main__":

    # ===============================
    # 1. Load data
    # ===============================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Mall_Customers.csv")

    df = load_data(DATA_PATH)

    # ===============================
    # 2. Preprocess data
    # ===============================
    df = preprocess_data(df)

    # ===============================
    # 3. Feature selection & scaling
    # ===============================
    FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    # ===============================
    # 4. Elbow Method (SAVE PNG)
    # ===============================
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plot_elbow(wcss)

    # ===============================
    # 5. Train final KMeans model
    # ===============================
    model, labels = train_kmeans(X_scaled, k=4)
    df['Cluster'] = labels

    # ===============================
    # 6. Evaluate model
    # ===============================
    score = evaluate_model(X_scaled, labels)
    print("Silhouette Score:", score)

    # ===============================
    # 7. Save model & scaler
    # ===============================
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # ===============================
    # 8. Save cluster visualization
    # ===============================
    plot_clusters(df)

    # ===============================
    # 9. Save cluster summary table
    # ===============================
    os.makedirs("outputs/tables", exist_ok=True)
    df.groupby('Cluster').mean(numeric_only=True)\
      .to_csv("outputs/tables/cluster_summary.csv")

    print("\nCluster Summary Saved Successfully.")
