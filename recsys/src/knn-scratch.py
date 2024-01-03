# k-Nearest Neighbors Approach to Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The following program aims to generate personalized article
# recommendations for users using a k-Nearest Neighbors (k-NN)
# approach with adjusted cosine similarity and the L1 norm. The
# program begins by defining a function called
# adjusted_cosine_similarity, which calculates the similarity
# between two vectors representing the click timestamps of
# articles. The adjustment involves considering user-specific
# biases by subtracting the average rating (click timestamp) for
# each article. The L1 norm is employed for normalization,
# measuring the magnitude of differences without squaring. This
# similarity function is then utilized in another function called
# k_nearest_neighbors, which computes the top k similar users
# for a specified user based on their article-clicking behavior.
# The main function loads a dataset of user clicks from a CSV file
# into a Pandas DataFrame and prints the top 5 personalized
# recommendations for each user.
# 	To execute the program, the main function reads a CSV file
# containing user click data, processes it using the k-NN
# algorithm with adjusted cosine similarity, and prints the top 5
# recommendations for each unique user. The k-NN approach
# leverages the similarity between users to suggest articles that
# similar users have clicked. The use of adjusted cosine similarity
# and the L1 norm ensures a personalized recommendation system
# that takes into account individual user biases and preferences.
# The program is designed to be versatile, allowing users to
# replace the file path with their own dataset for generating
# recommendations tailored to their specific user base.


import numpy as np
import pandas as pd

# Define a function to perform ALS Matrix Factorization for Collaborative Filtering
# Parameters:
#   - user_item_matrix: Matrix representing user-item interactions
#   - num_factors: Number of latent factors (default is 10)
#   - lambda_reg: Regularization parameter (default is 0.01)
#   - num_iterations: Number of iterations (default is 10)

def als_matrix_factorization(user_item_matrix, num_factors=10, lambda_reg=0.01, num_iterations=10):
    """
    ALS Matrix Factorization for Collaborative Filtering
    
    This function performs Alternating Least Squares (ALS) Matrix Factorization, a collaborative filtering
    technique commonly used in recommender systems. The goal is to decompose a user-item interaction matrix
    into two lower-dimensional matrices, one representing users and the other representing items. The
    decomposition captures latent factors that reflect user preferences and item characteristics.
    
    Parameters:
    - user_item_matrix: Matrix representing user-item interactions
    - num_factors: Number of latent factors (default is 10)
    - lambda_reg: Regularization parameter (default is 0.01)
    - num_iterations: Number of iterations (default is 10)
    
    Returns:
    - user_matrix: Matrix representing users and their latent factors
    - item_matrix: Matrix representing items and their latent factors
    """
    # Initialize user and item matrices randomly
    num_users, num_items = user_item_matrix.shape
    user_matrix = np.random.rand(num_users, num_factors)
    item_matrix = np.random.rand(num_items, num_factors)

    # ALS Algorithm
    for iteration in range(num_iterations):
        # Update user matrix
        for i in range(num_users):
            user_matrix[i, :] = np.linalg.solve(
                np.dot(item_matrix.T, item_matrix) + lambda_reg * np.eye(num_factors),
                np.dot(item_matrix.T, user_item_matrix.iloc[i, :].values)
            )

        # Update item matrix
        for j in range(num_items):
            item_matrix[j, :] = np.linalg.solve(
                np.dot(user_matrix.T, user_matrix) + lambda_reg * np.eye(num_factors),
                np.dot(user_matrix.T, user_item_matrix.iloc[:, j].values)
            )

    return user_matrix, item_matrix

# Define a function to calculate k-NN recommendations for a user using adjusted cosine similarity and L1 norm
# Parameters:
#   - user_id: Target user for recommendations
#   - df: DataFrame containing user-item interactions
#   - k: Number of nearest neighbors to consider (default is 5)

def k_nearest_neighbors(user_id, df, k=5):
    """
    Calculate k-NN recommendations for a user using adjusted cosine similarity and L1 norm.
    
    This function calculates k-NN recommendations for a target user based on adjusted cosine similarity and L1
    norm. It considers the user's click timestamps on articles to find similar users. The recommendations are
    determined by selecting the top-k users with the highest similarity scores and aggregating their clicked
    articles. The function excludes articles that the target user has already clicked to provide personalized
    suggestions.
    
    Parameters:
    - user_id: Target user for recommendations
    - df: DataFrame containing user-item interactions
    - k: Number of nearest neighbors to consider (default is 5)
    
    Returns:
    - recommendations: List of top-k recommended articles for the target user
    """
    # Extract the click timestamps of the target user and calculate their average rating
    user_clicks = df[df['user_id'] == user_id].set_index('click_article_id')['click_timestamp'].to_dict()
    user_ratings = {item: np.mean(df[df['user_id'] == user_id]['click_timestamp']) for item in user_clicks}

    # Get unique users in the dataset
    unique_users = df['user_id'].unique()
    similarity_scores = []

    # Calculate adjusted cosine similarity for each user
    for other_user_id in unique_users:
        if other_user_id != user_id:
            other_user_clicks = df[df['user_id'] == other_user_id].set_index('click_article_id')['click_timestamp'].to_dict()
            similarity = adjusted_cosine_similarity(user_clicks, other_user_clicks, user_ratings)
            similarity_scores.append((other_user_id, similarity))

    # Sort users by similarity score and select top-k
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_users = similarity_scores[:k]

    # Collect clicked articles from top-k users and generate recommendations
    recommendations = set()
    for user, _ in top_k_users:
        user_clicks = df[df['user_id'] == user].set_index('click_article_id')['click_timestamp'].to_dict()
        recommendations.update(user_clicks)

    # Filter out articles the target user has already clicked
    recommendations = [item for item in recommendations if item not in user_clicks]
    # Sort recommendations by click timestamp and select the top 5
    recommendations = sorted(recommendations, key=lambda x: user_clicks.get(x, 0), reverse=True)[:5]

    return recommendations

def main():
    """
    Main function to load data and print top 5 recommendations for all users.
    """
    # Replace this line with the actual path to your CSV file
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Print top 5 recommendations for all users
    unique_users = df['user_id'].unique()

    for user_id in unique_users:
        recommendations = k_nearest_neighbors(user_id, df)
        print(f"User {user_id}: {recommendations}")

if __name__ == "__main__":
    main()
