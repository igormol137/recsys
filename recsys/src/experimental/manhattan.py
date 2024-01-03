# k-NN Algorithm with L1 Norm for Collaborative Filtering:
# The k-Nearest Neighbors (k-NN) algorithm is utilized in collaborative filtering
# recommender systems to provide personalized recommendations. The algorithm
# assesses user similarities based on their interactions with items and employs
# the L1 norm (Manhattan distance) to measure dissimilarity between users.
# 
# Process:
#   1. For a target user, calculate the L1 norm distance with all other users
#      in the dataset.
#   2. Identify the k-nearest neighbors with the smallest L1 norm distances.
#   3. Retrieve the items clicked by these neighbors but not by the target user.
#   4. Offer these items as recommendations, aligning with the preferences of
#      users sharing similar interaction patterns.
#
# The k-NN algorithm with L1 norm is effective in capturing user preferences and
# revealing patterns in item interactions, enabling accurate and personalized
# recommendations.

import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
df = pd.read_csv(file_path)

# The function 'calculate_l1_norm' computes the L1 norm (Manhattan distance)
# between two user interaction vectors, representing the clicks on items by
# each user. The L1 norm measures the absolute differences between corresponding
# elements in the vectors, providing a quantitative measure of dissimilarity.
#
# Parameters:
#   - user1_clicks: List or NumPy array
#     The vector representing the clicks on items by the first user.
#   - user2_clicks: List or NumPy array
#     The vector representing the clicks on items by the second user.
#
# Returns:
#   An integer representing the L1 norm (Manhattan distance) between the two
#   user interaction vectors.

def calculate_l1_norm(user1_clicks, user2_clicks):
    # Determine the maximum length of the vectors
    max_len = max(len(user1_clicks), len(user2_clicks))

    # Pad the vectors with zeros to make them of equal length
    user1_clicks_padded = np.pad(user1_clicks, (0, max_len - len(user1_clicks)))
    user2_clicks_padded = np.pad(user2_clicks, (0, max_len - len(user2_clicks)))

    # Calculate the L1 norm (Manhattan distance) between the padded vectors
    return np.sum(np.abs(np.array(user1_clicks_padded) - np.array(user2_clicks_padded)))

# The function 'calculate_l1_norm' computes the L1 norm (Manhattan distance)
# between two user interaction vectors, representing the clicks on items by
# each user. The L1 norm measures the absolute differences between corresponding
# elements in the vectors, providing a quantitative measure of dissimilarity.
#
# Parameters:
#   - user1_clicks: List or NumPy array
#     The vector representing the clicks on items by the first user.
#   - user2_clicks: List or NumPy array
#     The vector representing the clicks on items by the second user.
#
# Returns:
#   An integer representing the L1 norm (Manhattan distance) between the two
#   user interaction vectors.

def calculate_l1_norm(user1_clicks, user2_clicks):
    # Determine the maximum length of the vectors
    max_len = max(len(user1_clicks), len(user2_clicks))

    # Pad the vectors with zeros to make them of equal length
    user1_clicks_padded = np.pad(user1_clicks, (0, max_len - len(user1_clicks)))
    user2_clicks_padded = np.pad(user2_clicks, (0, max_len - len(user2_clicks)))

    # Calculate the L1 norm (Manhattan distance) between the padded vectors
    return np.sum(np.abs(np.array(user1_clicks_padded) - np.array(user2_clicks_padded)))

# The function 'get_top_recommendations' provides the personalized
# recommendations for a specified user based on the items clicked by their
# k-nearest neighbors. It leverages the collaborative filtering approach,
# where the preferences of similar users influence the recommendations for the
# target user. The function identifies articles clicked by neighbors but not
# by the target user, offering a selection of items that align with the tastes
# of users sharing similar preferences.
#
# Parameters:
#   - user_id: int
#     The ID of the target user for whom recommendations are sought.
#   - df: Pandas DataFrame
#     The DataFrame containing user-item interaction data, with columns
#     'user_id' and 'click_article_id'.
#   - k_nearest_neighbors: list of int
#     The list of user IDs representing the k-nearest neighbors of the target user.
#   - num_recommendations: int (default: 5)
#     The number of top recommendations to retrieve for the target user.
#
# Returns:
#   A list of article IDs representing the top recommended items for the target user.

def get_top_recommendations(user_id, df, k_nearest_neighbors, num_recommendations=5):
    # Extract the articles clicked by the target user
    user_clicks = df[df['user_id'] == user_id]['click_article_id'].tolist()

    # Find articles clicked by neighbors but not by the target user
    neighbor_clicks = df[df['user_id'].isin(k_nearest_neighbors)]['click_article_id'].tolist()
    recommendations = list(set(neighbor_clicks) - set(user_clicks))

    # Return the top recommendations for the target user
    return recommendations[:num_recommendations]

# Main routine:
#   - Iterate through each user in the dataset.
#   - For each user, calculate their k-nearest neighbors using the function
#     'get_k_nearest_neighbors'.
#   - Retrieve the top recommendations for the user based on the k-nearest neighbors
#     using the function 'get_top_recommendations'.
#   - Store the recommendations in the 'all_user_recommendations' dictionary.
#   - Print the top 5 recommendations for each user.

# Get top recommendations for all users
all_users = df['user_id'].unique()
all_user_recommendations = {}

for user_id in all_users:
    # Get k-nearest neighbors
    k_nearest_neighbors = get_k_nearest_neighbors(user_id, df, k=5)

    # Get top recommendations for the user
    recommendations = get_top_recommendations(user_id, df, k_nearest_neighbors, num_recommendations=5)
    all_user_recommendations[user_id] = recommendations

# Print recommendations for all users
for user_id, recommendations in all_user_recommendations.items():
    print(f"Top 5 Recommendations for User {user_id}: {recommendations}")


