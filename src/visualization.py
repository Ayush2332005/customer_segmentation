import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df):
    """
    Saves and displays customer cluster scatter plot
    """
    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set1',
        data=df
    )
    plt.title("Customer Segments")

    plt.savefig(
        "outputs/figures/customer_clusters.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
