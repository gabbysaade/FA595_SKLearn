# SKLearn Basics HW: Script 2
# Using the Iris data set, create a script that trains multiple KMeans clustering models. Each KMeans model
# should have a different number of clusters, starting at 1 ending at 7. Graph the number of clusters vs
# the "inertia" and confirm, how many clusters should be used, using the elbow heuristic.

# Import packages
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Initialize dataset and view head
iris = datasets.load_iris(as_frame=True)
iris_data = iris.data
iris_data.head()


# Function to determine best number of clusters for the dataset, graph, and confirm
def cluster_analysis(data):
    # Train multiple KMeans clustering models, get inertia (average distance to centroids)
    inertia = []
    num_clust = [1, 2, 3, 4, 5, 6, 7]

    for i in range(1, 8):
        kmeans = KMeans(n_clusters=i)
        train = kmeans.fit(data)
        inertia.append(train.inertia_)

    # Plot number of clusters vs. inertia
    plt.plot(num_clust, inertia, color='green', marker='o')  # Elbow Heuristic shows 3 clusters is ideal
    plt.title('Number of Clusters vs. Inertia for Iris Dataset')
    plt.ylabel('Number of Clusters')
    plt.xlabel('Inertia')
    plt.plot(3, inertia[2], marker='*', markersize=12, markerfacecolor='red', markeredgecolor='red')  # Elbow Heuristic
    plt.show()

    print('The best number of clusters to describe this dataset is 3.')


if __name__ == '__main__':
    cluster_analysis(iris_data)
