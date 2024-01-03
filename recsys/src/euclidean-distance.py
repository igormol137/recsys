# Recommender System Using an Euclidean Norm for k-NN Algorithm
#
# Igor Mol <igor.mol@makes.ai>
#
# k-Nearest Neighbors (k-NN) algorithm is a collaborative filtering technique
# used in recommender systems to provide personalized suggestions based on user-
# item interactions. In this variant, the algorithm measures user similarities 
# using the Euclidean norm, also known as Euclidean distance.
#
# Process:
#   1. For a target user, compute the Euclidean distance with all other users based
#      on their interactions with items.
#   2. Identify the k-nearest neighbors with the smallest Euclidean distances.
#   3. Aggregate the item preferences of these neighbors to make recommendations.
#      This is often achieved through weighted averaging or other interpolation
#      techniques.
#   4. Offer the top recommendations to the target user based on the aggregated
#      preferences of similar users.

import numpy as np
import pandas as pd

# Load the CSV file into a DataFrame
file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
df = pd.read_csv(file_path)

# Create a user-item matrix
user_item_matrix = pd.pivot_table(df, values='session_size', index='user_id', columns='click_article_id', fill_value=0)

# The function 'euclidean_distance' calculates the pairwise Euclidean distances
# between users based on their interactions with items in a user-item matrix. It
# then converts these distances into similarity scores, making them suitable for
# collaborative filtering applications.
#
# Parameters:
#   - user_item_matrix: Pandas DataFrame
#     A matrix representing user-item interactions, where rows correspond to users,
#     columns to items, and values to interaction strengths or ratings.
#
# Returns:
#   A matrix of user similarities derived from Euclidean distances. The values are
#   transformed to the range [0, 1] using the formula 1 / (1 + distance), where
#   smaller distances result in higher similarity scores.

def euclidean_distance(user_item_matrix):
    # Calculate pairwise Euclidean distances between users
    distance_matrix = np.sqrt(np.sum((user_item_matrix.values[:, None, :] - user_item_matrix.values) ** 2, axis=2))

    # Convert distances to similarities and rescale to the range [0, 1]
    return 1 / (1 + distance_matrix)


# The function 'get_top_similar_users' is designed to identify the top k similar
# users for a specified target user in a collaborative filtering recommender
# system. It utilizes a similarity matrix, which quantifies the similarity
# between users based on their interactions with items.
#
# Parameters:
#   - similarity_matrix: NumPy array
#     A matrix representing user similarities, commonly computed using a
#     similarity measure like cosine similarity.
#   - target_user: int
#     The ID of the target user for whom similar users are to be determined.
#   - k: int (default: 5)
#     The number of top similar users to retrieve.
#
# Returns:
#   An array containing the IDs of the top k similar users for the target user.

def get_top_similar_users(similarity_matrix, target_user, k=5):
    # Extract similarity values for the target user
    user_similarity = similarity_matrix[target_user]

    # Find the indices of top k similar users in descending order
    top_similar_users = np.argsort(user_similarity)[::-1][1:k+1]

    return top_similar_users


# The function 'predict_ratings' aims to predict item ratings for a target user
# in our collaborative filtering recommender system. It employs a weighted
# interpolation approach, considering the ratings of top similar users and their
# similarities to the target user.
#
# Parameters:
#   - user_item_matrix: Pandas DataFrame
#     A matrix representing user-item interactions, where rows correspond to users,
#     columns to items, and values to interaction strengths or ratings.
#   - similarity_matrix: NumPy array
#     A matrix representing user similarities, typically computed using a similarity
#     measure such as cosine similarity.
#   - target_user: int
#     The ID of the target user for whom item ratings are to be predicted.
#
# Returns:
#   An array of predicted ratings for each item in the user-item matrix, based on
#   the weighted interpolation of ratings from top similar users.

def predict_ratings(user_item_matrix, similarity_matrix, target_user):
    # Get the IDs of top similar users
    top_similar_users = get_top_similar_users(similarity_matrix, target_user)

    # Extract ratings and weights
    ratings = user_item_matrix.values
    weights = similarity_matrix[target_user, top_similar_users]

    # Perform weighted interpolation to predict ratings
    predictions = np.sum(weights.reshape(-1, 1) * ratings[top_similar_users], axis=0) / np.sum(weights)

    return predictions


# The function 'get_top_5_recommendations' is designed to generate personalized
# recommendations for a specific user in our collaborative filtering recommender
# system. It leverages predicted ratings based on user similarities and item
# interactions.
#
# Parameters:
#   - user_item_matrix: Pandas DataFrame
#     A matrix representing user-item interactions, where rows correspond to users,
#     columns to items, and values to interaction strengths or ratings.
#   - similarity_matrix: NumPy array
#     A matrix representing user similarities, typically computed using a
#     similarity measure such as cosine similarity.
#   - target_user: int
#     The ID of the user for whom recommendations are to be generated.

# Returns:
#   An array containing the IDs of the top 5 recommended items for the target user,
#   based on predicted ratings and collaborative filtering.

def get_top_5_recommendations(user_item_matrix, similarity_matrix, target_user):
    # Obtain predicted ratings for items
    predictions = predict_ratings(user_item_matrix, similarity_matrix, target_user)

    # Retrieve the indices of the top 5 recommended items in descending order
    top_recommendations = np.argsort(predictions)[::-1][:5]

    return top_recommendations


# Compute Euclidean distance similarity matrix
similarity_matrix = euclidean_distance(user_item_matrix)

# Print top 5 recommendations for all users
for user in user_item_matrix.index:
    recommendations = get_top_5_recommendations(user_item_matrix, similarity_matrix, user)
    print(f"Top 5 recommendations for User {user}: {recommendations}")

