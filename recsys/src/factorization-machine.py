# factorization-machine.py
#
# Igor Mol <igor.mol@makes.ai>
#
# Factorization Machines (FM) for Recommender Systems:
#   - FM is a machine learning model designed for recommender systems,
#     specifically addressing sparse user-item interaction data.
#   - It represents users and items as combinations of latent factors,
#     capturing hidden patterns in the data.
#   - The model excels at generalizing from limited user-item interactions,
#     providing personalized recommendations.
#   - FM considers interactions between users and items, as well as additional
#     features like demographics or item characteristics.
#   - It learns the strength of these interactions and features, resulting in a
#     predictive model for generating personalized recommendations.
#   - FM is efficient and scalable, making it suitable for large-scale
#     recommendation tasks, enhancing recommendation quality.

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define a function to load data from a CSV file into a Pandas DataFrame.
# Parameters:
#   - file_path: The file path of the CSV file containing the data.

def load_data(file_path):
    # Use the Pandas read_csv function to read data from the specified file path
    # and load it into a DataFrame.
    return pd.read_csv(file_path)

# Define a function to create a user-item matrix from a DataFrame.
# Parameters:
#   - df: DataFrame containing user-item interactions.

def create_user_item_matrix(df):
    # Create a new DataFrame 'user_item_df' by selecting only the 'user_id' and 'click_article_id' columns from the input DataFrame 'df'.
    user_item_df = df[['user_id', 'click_article_id']].copy()
    
    # Create a pivot table from 'user_item_df'. The index is set as 'user_id', columns as 'click_article_id', and values are the count of occurrences (size).
    # This means the resulting table will represent how many times each user clicked on each article.
    user_item_matrix = user_item_df.pivot_table(index='user_id', columns='click_article_id', aggfunc='size', fill_value=0)
    
    # Return the user-item matrix.
    return user_item_matrix

# Define a function to standardize a user-item matrix.
# Parameters:
#   - matrix: The original user-item matrix.

def standardize_matrix(matrix):
    # Create a StandardScaler object.
    scaler = StandardScaler()
    
    # Fit the StandardScaler to the input matrix and transform the matrix to obtain standardized values.
    standardized_matrix = scaler.fit_transform(matrix)
    
    # Set any values in the standardized matrix that are less than 0 to 0.
    # This is done to ensure that all values in the resulting matrix are non-negative.
    standardized_matrix[standardized_matrix < 0] = 0
    
    # Return the standardized matrix.
    return standardized_matrix

# Define a function to cluster users based on a standardized user-item matrix.
# Parameters:
#   - user_item_matrix_standardized: Standardized matrix of user-item interactions.
#   - n_clusters: Number of clusters for KMeans algorithm (default is 5).

def cluster_users(matrix, n_clusters=5):
    # Create a KMeans clustering model with the specified number of clusters (default is 5).
    # n_init is the number of times the KMeans algorithm will be run with different centroid seeds.
    # random_state is set for reproducibility.
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    
    # Fit the KMeans model to the input matrix and predict the cluster labels for each user.
    user_clusters = kmeans.fit_predict(matrix)
    
    # Return the cluster labels for each user.
    return user_clusters

# Define a function to create a mapping between user IDs and their respective clusters.
# Parameters:
#   - user_item_matrix: Matrix representing user-item interactions.
#   - user_clusters: Array identifying the cluster membership of each user.

def create_user_cluster_mapping(user_item_matrix, user_clusters):
    # Create a dictionary by zipping user IDs (from the index of user_item_matrix) with their corresponding cluster labels.
    user_cluster_mapping = dict(zip(user_item_matrix.index, user_clusters))
    
    # Return the user-cluster mapping.
    return user_cluster_mapping

# Define a function to train recommendation models for user-item
# interactions within user clusters.
# Parameters:
#   - user_item_matrix: Matrix representing user-item interactions.
#   - user_clusters: Array identifying the cluster membership of each user.
#   - n_components: Number of components for NMF algorithm (default is 5).
#   - max_iter: Maximum number of iterations for NMF algorithm (default is 500).

def train_recommendation_models(user_item_matrix, user_clusters, n_components=5, max_iter=500):
    # Initialize an empty dictionary to store recommendation models for each cluster
    recommendation_models = {}
    
    # Iterate through each cluster (identified by cluster_id)
    for cluster_id in range(max(user_clusters) + 1):
        # Create a boolean mask to select rows corresponding to the current cluster
        cluster_indices = user_clusters == cluster_id
        
        # Extract the user-item matrix for the current cluster using the boolean mask
        cluster_data = user_item_matrix[cluster_indices, :]
        
        # Initialize a Non-Negative Matrix Factorization (NMF) model for the current cluster
        # NMF is used for matrix factorization to discover latent patterns in the data
        model = NMF(n_components=n_components, init='random', max_iter=max_iter, random_state=42)
        
        # Fit the NMF model to the user-item interactions in the current cluster
        model.fit(cluster_data)
        
        # Store the trained model in the dictionary, using the cluster_id as the key
        recommendation_models[cluster_id] = model
    
    # Return the dictionary containing recommendation models for each cluster
    return recommendation_models

# Define a function to generate top recommendations for users using recommenda-
# tion models.
# Parameters:
#   - user_item_df: DataFrame containing user-item interactions and user clusters.
#   - recommendation_models: Dictionary of recommendation models for user clusters.
#   - user_item_matrix_standardized: Standardized user-item matrix.

def generate_top_recommendations(user_item_df, user_clusters, recommendation_models):
    # Initialize an empty dictionary to store top recommendations for each user
    top_recommendations = {}
    
    # Extract user indices from the DataFrame's index
    user_indices = user_item_df.index.values
    
    # Iterate through each user identified by user_id
    for user_id in user_indices:
        # Identify the user's cluster using the precomputed 'user_clusters' array
        user_cluster = user_clusters[user_id]
        
        # Retrieve the recommendation model associated with the user's cluster
        model = recommendation_models[user_cluster]
        
        # Extract the items the user has already interacted with
        user_items = user_item_df.loc[user_id, :].values
        
        # Get all unique items in the dataset and exclude those the user has already interacted with
        all_items = user_item_df.columns.values
        candidate_items = list(set(all_items) - set(user_items))
        
        # Transform the user's standardized interaction vector using the recommendation model
        scores = model.transform(user_item_matrix_standardized[user_id, :].reshape(1, -1))
        
        # Identify the indices of the top 5 recommendations based on the transformed scores
        top_indices = sorted(range(len(scores[0])), key=lambda i: scores[0][i], reverse=True)[:5]
        
        # Map the indices to actual item names to form the top recommended items for the user
        top_items = [candidate_items[i] for i in top_indices]
        
        # Store the top recommendations in the dictionary using the user_id as the key
        top_recommendations[user_id] = top_items
    
    # Return the dictionary containing top recommendations for each user
    return top_recommendations


# Define a function to print user recommendations.
# Parameters:
#   - top_recommendations: Dictionary containing top recommendations for each user.

def print_top_recommendations(top_recommendations):
    for user_id, recommendations in top_recommendations.items():
        print(f"User {user_id}: {recommendations}")

def main():
    # Load the original CSV file
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
    df = load_data(file_path)

    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(df)

    # Standardize the user-item matrix
    user_item_matrix_standardized = standardize_matrix(user_item_matrix)

    # Cluster users
    user_clusters = cluster_users(user_item_matrix_standardized)

    # Map user clusters to user IDs
    user_cluster_mapping = create_user_cluster_mapping(user_item_matrix, user_clusters)

    # Train recommendation models
    recommendation_models = train_recommendation_models(user_item_matrix_standardized, user_clusters)

    # Generate top recommendations
    top_recommendations = generate_top_recommendations(user_item_df, user_clusters, recommendation_models)

    # Print top recommendations
    print_top_recommendations(top_recommendations)

if __name__ == "__main__":
    main()


