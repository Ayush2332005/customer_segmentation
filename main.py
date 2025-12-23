from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.clustering import train_kmeans
from src.evaluation import evaluate_model
from src.visualization import plot_clusters
from sklearn.preprocessing import StandardScaler
import joblib
import os

if __name__ == "__main__":

    # 1. Load data
    df = load_data("data/raw/Mall_Customers.csv")

    # 2. Preprocess
    df = preprocess_data(df)

    # 3. Feature selection & scaling
    FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    # 4. Train KMeans (THIS MUST COME FIRST)
    model, labels = train_kmeans(X_scaled, k=4)

    # 5. Add Cluster column (CRITICAL STEP)
    df['Cluster'] = labels

    # 6. Evaluate
    score = evaluate_model(X_scaled, labels)
    print("Silhouette Score:", score)

    # 7. Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # 8. NOW plot clusters (AFTER Cluster column exists)
    plot_clusters(df)
