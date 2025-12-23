# Customer Segmentation using K-Means Clustering

## 📌 Project Overview
Customer segmentation is the process of dividing customers into distinct groups based on shared characteristics.  
This project applies **K-Means clustering** to segment customers based on demographic and spending behavior, helping businesses design **targeted marketing strategies**.

The project follows an **industry-standard machine learning pipeline**, from data preprocessing to model evaluation and visualization.

---

## 🎯 Objective
- Segment customers into meaningful groups
- Understand customer purchasing behavior
- Support data-driven marketing and business decisions

---

## 📊 Dataset
**Mall Customers Dataset**

**Features used:**
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

The dataset contains demographic and behavioral information of mall customers.

---

## 🧠 Techniques & Concepts Used
- Data Cleaning & Preprocessing
- Feature Scaling using `StandardScaler`
- K-Means Clustering
- Elbow Method (WCSS / Inertia)
- Silhouette Analysis
- Cluster Profiling & Interpretation
- Data Visualization

---

## 🏗️ Project Structure

customer_segmentation/
│
├── data/
│ └── raw/
│ └── Mall_Customers.csv
│
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── clustering.py
│ ├── evaluation.py
│ └── visualization.py
│
├── notebooks/
│ └── rk.ipynb
│
├── outputs/
│ ├── figures/
│ │ ├── elbow_plot.png
│ │ └── customer_clusters.png
│ └── tables/
│ └── cluster_summary.csv
│
├── models/
│ ├── kmeans_model.pkl
│ └── scaler.pkl
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/customer-segmentation.git
cd customer-segmentation


2️⃣ Create Virtual Environment (Optional but Recommended)
    python -m venv venv


Activate virtual environment

    Windows:

        venv\Scripts\activate


    Linux / macOS:

        source venv/bin/activate

3️⃣ Install Dependencies
    pip install -r requirements.txt


▶️ How to Run the Project

    From the project root directory, run:

    python main.py


📈 Outputs Generated

    After execution, the following outputs are automatically created:


📊 Visualizations

    Elbow Plot – Optimal number of clusters

    Customer Cluster Scatter Plot – Visual customer segments

    Saved in:

    outputs/figures/


📋 Tables

    Cluster Summary Table (mean values per cluster)

    Saved in:

    outputs/tables/cluster_summary.csv


💾 Models

    Trained K-Means model

    Fitted scaler

    Saved in:

    models/


📌 Results & Insights

    Optimal number of clusters selected using Elbow Method and Silhouette Analysis

    Final silhouette score: ~0.40

    Customers segmented into 4 meaningful groups, such as:

    High income – high spending (Premium customers)

    High income – low spending (Potential customers)

    Low income – high spending (Impulsive buyers)

    Low income – low spending (Budget customers)


🧪 Evaluation Metric

    Silhouette Score

    Range: -1 to +1

    Higher score indicates better-defined clusters

    A score of ~0.40 indicates reasonable cluster separation, which is common in real-world customer data

🛠️ Technologies Used

    Python

    Pandas

    NumPy

    Matplotlib

    Seaborn

    Scikit-learn

    Joblib


🚀 Future Improvements

    Try other clustering algorithms (DBSCAN, Hierarchical)

    Include additional features (Gender, Purchase History)

    Deploy as a Streamlit web application

    Automate cluster interpretation