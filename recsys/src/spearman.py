# A Recommender System Based Upon Spearman Similarity Algorithm
#
# Igor Mol <igor.mol@makes.ai>
#
# The Spearman similarity algorithm in collaborative filtering is a method
# for measuring the similarity between items based on the ranking of user
# interactions. It assesses the monotonic relationship between the ranked
# values of two items across users. The algorithm computes the Spearman rank
# correlation matrix, which quantifies the degree of similarity between all
# pairs of items. The process involves ranking the interactions for each item
# across users, calculating differences in ranks, and using these differences
# to determine the correlation. A higher Spearman rank correlation indicates
# greater similarity between items. This algorithm is valuable for systems
# where the ordinal relationships between user preferences are crucial.


import numpy as np
import pandas as pd

# Load the CSV file into a DataFrame
file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
df = pd.read_csv(file_path)

# Create a user-item matrix
user_item_matrix = pd.pivot_table(df, values='session_size', index='user_id', columns='click_article_id', fill_value=0)

# spearman_rank_correlation():
# This function calculates the Spearman rank correlation matrix for a user-item
# interaction matrix. This correlation measures the monotonic relationship bet-
# ween the ranked values of pairs of items across users.
#
# Parameters:
#   - user_item_matrix: Pandas DataFrame
#     The user-item interaction matrix where rows represent users, columns represent
#     items, and the values represent the strength of the interaction (e.g., ratings).
#
# Returns:
#   A 2D NumPy array representing the Spearman rank correlation matrix for the items
#   in the user-item matrix.

def spearman_rank_correlation_from_scratch(user_item_matrix):
    # Rank values for each item across users
    ranks = user_item_matrix.apply(lambda x: x.rank(), axis=1)

    # Get the dimensions of the user-item matrix
    n, m = user_item_matrix.shape

    # Initialize a matrix for the Spearman rank correlation values
    correlation_matrix = np.zeros((m, m))

    # Calculate Spearman rank correlation for each pair of items
    for i in range(m):
        for j in range(i, m):
            x = ranks.iloc[:, i]  # Rank values for item i
            y = ranks.iloc[:, j]  # Rank values for item j
            diff = x - y
            squared_diff = diff ** 2
            sum_squared_diff = np.sum(squared_diff)
            correlation = 1 - (6 * sum_squared_diff) / (n * (n**2 - 1))
            
            # Assign correlation values symmetrically in the matrix
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation

    # Return the Spearman rank correlation matrix
    return correlation_matrix


# get_top_similar_users():
# This function is designed to identify the top 'k' similar
# users for a specified target user based on a given similarity matrix. It is
# commonly used in collaborative filtering recommendation systems to find users
# with similar preferences or behaviors.
#
# Parameters:
#   - similarity_matrix: Numpy array or Pandas DataFrame
#     The similarity matrix containing pairwise similarities between users. The
#     similarity values indicate how closely the users' preferences align.
#   - target_user: Integer
#     The identifier of the target user for whom similar users are to be retrieved.
#   - k: Integer (default: 5)
#     The number of top similar users to retrieve.
#
# Returns:
#   An array containing the identifiers of the top 'k' similar users to the
#   specified target user based on the provided similarity matrix.

def get_top_similar_users(similarity_matrix, target_user, k=5):
    # Extract similarity values for the target user
    user_similarity = similarity_matrix[target_user]

    # Identify the indices of the top 'k' similar users (excluding the target user)
    top_similar_users = np.argsort(user_similarity)[::-1][1:k+1]

    # Return the array of top similar users
    return top_similar_users


# predict_ratings():
# The function 'predict_ratings' aims to predict ratings for a target user using
# weighted interpolation based on the similarity between the target user and other
# users. This is commonly employed in collaborative filtering recommendation
# systems to estimate preferences of a user for items they have not interacted 
# with.
#
# Parameters:
#   - user_item_matrix: Pandas DataFrame
#     The user-item interaction matrix where rows represent users, columns represent
#     items, and the values represent the strength of the interaction (e.g., ratings).
#   - similarity_matrix: Numpy array or Pandas DataFrame
#     The similarity matrix containing pairwise similarities between users. It
#     measures how closely the preferences of users align.
#   - target_user: Integer
#     The identifier of the target user for whom ratings are to be predicted.
#
# Returns:
#   An array of predicted ratings for the items in the user-item matrix for the
#   specified target user.

# Function to predict ratings using weighted interpolation
def predict_ratings(user_item_matrix, similarity_matrix, target_user):
    # Get the identifiers of the top similar users to the target user
    top_similar_users = get_top_similar_users(similarity_matrix, target_user)

    # Extract ratings and similarity weights for calculations
    ratings = user_item_matrix.values
    weights = similarity_matrix[target_user, top_similar_users]

    # Perform weighted interpolation to predict ratings
    predictions = np.sum(weights.reshape(-1, 1) * ratings[top_similar_users], axis=0) / np.sum(weights)

    # Return the array of predicted ratings
    return predictions


# get_top_5_recommendations():
# This function is designed to provide the top 5
# recommendations for a specified target user based on collaborative filtering.
# It utilizes predicted ratings obtained through the 'predict_ratings' function
# to suggest items the user might be interested in.
#
# Parameters:
#   - user_item_matrix: Pandas DataFrame
#     The user-item interaction matrix where rows represent users, columns represent
#     items, and the values represent the strength of the interaction (e.g., ratings).
#   - similarity_matrix: Numpy array or Pandas DataFrame
#     The similarity matrix containing pairwise similarities between users. It
#     measures how closely the preferences of users align.
#   - target_user: Integer
#     The identifier of the target user for whom recommendations are to be provided.
#
# Returns:
#   An array containing the identifiers of the top 5 recommended items for the
#   specified target user.

def get_top_5_recommendations(user_item_matrix, similarity_matrix, target_user):
    # Predict ratings for the target user
    predictions = predict_ratings(user_item_matrix, similarity_matrix, target_user)

    # Identify the indices of the top 5 recommended items
    top_recommendations = np.argsort(predictions)[::-1][:5]

    # Return the array of top 5 recommended items
    return top_recommendations


# Compute Spearman Rank Correlation similarity matrix from scratch
similarity_matrix = spearman_rank_correlation_from_scratch(user_item_matrix)

# Print top 5 recommendations for all users
for user in user_item_matrix.index:
    recommendations = get_top_5_recommendations(user_item_matrix, similarity_matrix, user)
    print(f"Top 5 recommendations for User {user}: {recommendations}")
