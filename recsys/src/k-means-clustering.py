# k-Means Clustering Algorithm for Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The k-means clustering algorithm categorizes users into distinct clusters
# based on their preferences or behaviors. Similarly, for item-based clustering,
# it groups items that exhibit similar characteristics or are frequently
# interacted with together. The 'k' in k-means represents the number of clusters,
# and the algorithm aims to partition the users or items into 'k' groups, each
# characterized by a centroid that minimizes the sum of squared distances between
# members of the cluster and the centroid.
# 	The process begins by randomly selecting 'k' initial centroids,
# representing the centers of the clusters. Subsequently, each user or item is
# assigned to the cluster whose centroid is closest to it, usually based on a
# specified distance metric such as Euclidean distance. After the initial
# assignment, the centroids are recalculated as the mean of all members within
# each cluster. This assignment and centroid update process iterates until
# convergence, with users or items potentially switching clusters to minimize
# the within-cluster variance. The resulting clusters capture inherent patterns
# and similarities in user preferences or item characteristics


import pandas as pd
import numpy as np

# Introduction:
# The class 'KMeansFromScratch' implements the k-means clustering algorithm
# from scratch. It partitions a given dataset into 'n_clusters' clusters based
# on similarity, using centroids to represent each cluster. Users can specify
# the number of clusters, maximum iterations for convergence, and a random seed
# for reproducibility.

# Parameters:
#   - n_clusters: Integer
#     The number of clusters to form as well as the number of centroids to
#     generate.
#   - max_iters: Integer (default: 100)
#     The maximum number of iterations for the k-means algorithm to converge.
#   - random_state: Integer or None (default: None)
#     Seed for random initialization. If 'None', randomness is based on system
#     time.

# Returns:
#   An instance of the 'KMeansFromScratch' class with the following attributes:
#   - centroids: Numpy array
#     Centroids representing the center of each cluster after fitting the model.
#   - labels: Numpy array
#     Labels indicating the cluster assignment for each data point after
#     fitting.

class KMeansFromScratch:
    def __init__(self, n_clusters, max_iters=100, random_state=None):
        # Initialize the KMeansFromScratch object with user-defined parameters
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None  # To store cluster centroids
        self.labels = None     # To store cluster labels for each data point

    def fit(self, X):
        np.random.seed(self.random_state)

        # Initialize centroids randomly by choosing data points
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]

        # Iteratively update centroids and labels until convergence or max_iters
        for _ in range(self.max_iters):
            # Assign each point to the nearest centroid
            self.labels = self._assign_labels(X)

            # Update centroids based on assigned labels
            new_centroids = self._update_centroids(X)

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def _assign_labels(self, X):
        # Assign labels to each data point based on the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        # Update centroids based on the mean of data points in each cluster
        new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
        return new_centroids
        


# cosine_similarity_matrix(X, Y):
# This function computes the cosine similarity matrix between two sets of vec-
# tors, represented by matrices X and Y.
# 	Firstly, the code calculates the dot product between the vectors in ma-
# trix X and the transpose of matrix Y using the np.dot function. The dot pro-
# duct operation essentially captures the similarity between corresponding 
# vectors. Next, the code computes the L2 norm (Euclidean norm) along the rows 
# of matrices X and Y using np.linalg.norm. These norms represent the lengths of
# the vectors and are essential for normalizing the dot product to obtain cosine
# similarity. The actual cosine similarity is then calculated by dividing the
# dot product by the outer product of the vector norms, ensuring that the simi-
# larity values are normalized. To prevent division by zero, a small constant 
# e^(-8) is added to the denominator, where e := Euler constant. The resulting
# matrix contains cosine similarity scores between vectors in X and Y, where 
# higher values indicate greater similarity.

def cosine_similarity_matrix(X, Y):
    dot_product = np.dot(X, Y.T)
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)
    cosine_similarity = dot_product / (np.outer(norm_X, norm_Y) + 1e-8)  # Avoid division by zero
    return cosine_similarity

# load_data():
# This function, named 'load_data', serves the purpose of loading data from a
# CSV file into a Pandas DataFrame. It is designed to facilitate the initial
# step in data preprocessing for further analysis.
# Parameters:
#   - file_path: String
#     The 'file_path' parameter specifies the path to the CSV file that contains
#     the data to be loaded.
# Objective:
#   The primary goal of this function is to read the data from the specified CSV
#   file and transform it into a structured format. It leverages the 'pd.read_csv'
#   method from the Pandas library to create a DataFrame.
# Returns:
#   The function returns a Pandas DataFrame containing the loaded data. This
#   DataFrame is then available for use in subsequent data analysis or processing.

def load_data(file_path):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Return the loaded DataFrame
    return df

# The function 'create_user_item_matrix' is designed to generate a user-item
# interaction matrix from a given Pandas DataFrame containing user-item interaction data.
# This matrix provides a structured representation of user interactions with items and
# is a crucial step in collaborative filtering-based recommendation systems.
# Parameters:
#   - data: Pandas DataFrame
#     The 'data' parameter is expected to be a DataFrame containing user-item
#     interaction data. It should include columns such as 'user_id', 'click_article_id',
#     and 'session_size', where 'session_size' indicates the strength of the interaction.
# Objective:
#   The primary objective of this function is to transform the input user-item interaction
#   data into a matrix format. It utilizes the 'pd.pivot_table' method to pivot the DataFrame,
#   setting 'session_size' as values, 'user_id' as the index, and 'click_article_id' as columns.
#   The 'fill_value' parameter is employed to handle cases where no explicit interaction is present,
#   with a default value of 0 indicating no interaction.
# Returns:
#   The function returns a Pandas DataFrame representing the user-item interaction matrix.
#   This matrix serves as a fundamental data structure for collaborative filtering algorithms
#   in recommendation systems.
def create_user_item_matrix(data):
    # Create a user-item interaction matrix
    user_item_matrix = pd.pivot_table(data, values='session_size', index='user_id', columns='click_article_id', fill_value=0)

    # Return the generated user-item interaction matrix
    return user_item_matrix

def main():
    # Load data
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
    data = load_data(file_path)

    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(data)

    # Apply KMeans clustering from scratch
    num_clusters = 5
    kmeans = KMeansFromScratch(n_clusters=num_clusters, random_state=42)
    kmeans.fit(user_item_matrix.values)
    user_clusters = kmeans.labels

    # Compute cosine similarity between users and centroids from scratch
    user_similarity = cosine_similarity_matrix(user_item_matrix.values, kmeans.centroids)

    # Get top 5 recommendations for all users
    top_n_recommendations = []
    for user_id in user_item_matrix.index:
        user_cluster = user_clusters[user_id]
        similarities = user_similarity[user_id, user_cluster]
        similar_users = np.argsort(similarities)[::-1][:5]

        recommendations = []
        for cluster_user_id in similar_users:
            recommendations.extend(user_item_matrix.loc[cluster_user_id, user_item_matrix.columns[:-1]][user_item_matrix.loc[cluster_user_id, user_item_matrix.columns[:-1]] == 0].index)

        top_n_recommendations.append({'user_id': user_id, 'recommendations': recommendations[:5]})

    # Print the top 5 recommendations for all users
    for user_rec in top_n_recommendations:
        print(f"Top 5 recommendations for user {user_rec['user_id']}: {user_rec['recommendations']}")

if __name__ == "__main__":
    main()

