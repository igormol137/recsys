# Recommender System: A Collaborative Filtering Approach Using
# Alternating Least Squares
#
# Igor Mol <igor.mol@makes.ai>
#
# ALS (Alternating Least Squares) Matrix Factorization is a
# collaborative filtering technique widely used in recommender
# systems. The goal of ALS is to decompose a user-item interaction
# matrix into two lower-dimensional matrices, one representing users
# and the other representing items. This decomposition captures
# latent factors that reflect user preferences and item
# characteristics. The name "Alternating Least Squares" stems from
# the iterative nature of the method, where it alternates between
# fixing one matrix and optimizing the other through the least
# squares minimization.
# 	The ALS method iteratively refines the user and item matrices to
# minimize the difference between the predicted and observed
# user-item interactions. During each iteration, it optimizes one
# matrix while holding the other constant. This process continues
# until convergence, resulting in user and item matrices that
# effectively capture the underlying patterns in the interaction
# data.

import numpy as np
import pandas as pd

# Load the CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Define a function to create a user-item matrix from a DataFrame containing 
# user-item interactions,considering the session size as the interaction 
# strength.
# It receive the following parameters:
#
#   - df: DataFrame with columns 'user_id', 'click_article_id', and 'session_
# size' representing interactions.
#
# And returns:
#
#   - user_item_matrix: DataFrame where rows represent users, columns represent 
# items, and values represent the session size (interaction strength) between 
# users and items.

def create_user_item_matrix(df):
    user_item_matrix = {}

    for _, row in df.iterrows():
        user_id = row['user_id']
        article_id = row['click_article_id']
        session_size = row['session_size']

        if user_id not in user_item_matrix:
            user_item_matrix[user_id] = {}

        user_item_matrix[user_id][article_id] = session_size

    return pd.DataFrame(user_item_matrix).fillna(0)

# Define a function for Alternating Least Squares (ALS) Matrix Factorization to 
# decompose a user-item matrix.
# Parameters:
#   - user_item_matrix: DataFrame representing user-item interactions.
#   - num_factors: Number of latent factors for the decomposition (default is 10).
#   - lambda_reg: Regularization parameter to prevent overfitting (default is 0.01).
#   - num_iterations: Number of iterations for the ALS algorithm (default is 10).
# Returns:
#   - user_matrix: Matrix representing users and their latent factors.
#   - item_matrix: Matrix representing items and their latent factors.

def als_matrix_factorization(user_item_matrix, num_factors=10, lambda_reg=0.01, num_iterations=10):
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

# Predict ratings for all users and items
def predict_ratings(user_matrix, item_matrix):
    return np.dot(user_matrix, item_matrix.T)

# Get top recommendations for a user
def get_top_recommendations(predicted_ratings, user):
    return np.argsort(predicted_ratings[user, :])[::-1][:5]

# Main function to run the recommendation system
def main():
    # Load data
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
    df = load_data(file_path)

    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(df)

    # ALS Matrix Factorization
    user_matrix, item_matrix = als_matrix_factorization(user_item_matrix)

    # Predict ratings
    predicted_ratings = predict_ratings(user_matrix, item_matrix)

    # Print top 5 recommendations for all users
    for user in user_item_matrix.columns:
        top_recommendations = get_top_recommendations(predicted_ratings, user)
        print(f"Top 5 recommendations for User {user}: {top_recommendations}")

# Run the recommendation system
if __name__ == "__main__":
    main()
