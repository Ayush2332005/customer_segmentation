import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_elbow(wcss):
    """
    Saves and displays Elbow Method plot
    """
    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")

    plt.savefig(
        "outputs/figures/elbow_plot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def plot_clusters(df):
    """
    Saves and displays customer segmentation scatter plot
    """
    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure(figsize=(8, 6))
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
