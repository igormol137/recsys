# jaccard.py
#
# Igor Mol <igor.mol@makes.ai>
#
# In this recommender system, the Jaccard similarity is employed to assess the 
# similarity between sets of items, specifically in the context of user-item 
# interactions. This similarity metric calculates the ratio of the number of 
# common items between two sets to the total number of distinct items across 
# both sets. It quantifies the overlap or intersection of items between two 
# users' preferences, providing a way to gauge how similar their tastes are.
# By calculating the Jaccard similarity between users, the system can identify
# those who share a significant number of common preferences, indicating a 
# higher likelihood that they have similar tastes in items. This similarity mea-
# sure is particularly useful when dealing with sparse datasets, where users in-
# teract with only a small subset of the available items.
# 

import numpy as np
import pandas as pd

# Replace this line with the actual path to your CSV file
file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Extract user_id, click_article_id, and create a user-item matrix
user_item_matrix = df.pivot(index='user_id', columns='click_article_id', values='click_timestamp').fillna(0)

# l1_norm(user1, user2):
# In functional analysis, the L1 norm, also known as the Manhattan norm or taxi-
# cab norm, is a way to measure the "size" or magnitude of a vector. Specifical-
# ly, for a vector in a vector space, the L1 norm is calculated by summing the 
# absolute values of its individual components. In other words, it represents 
# the total "distance" one would need to travel along the coordinate axes to 
# reach the vector from the origin. 
# 	Mathematically, let x := (x_1, ..., x_n) be a vector. Thus, the L1 norm 
# is given by the sum of the absolute values: ||x|| := |x_1| + ... + |x_n|. The 
# L1 norm is commonly used in various mathematical and computational applications
# due to its simplicity and computational efficiency.

def l1_norm(user1, user2):
    # Calculate the L1 norm (Manhattan distance) between two users
    return np.sum(np.abs(user1 - user2))

# k_nearest_neighbors():
# This function identifies the k-nearest neighbors of a given user based on the
# L1 norm (Manhattan distance) between their preferences or interactions. The 
# function takes three parameters: the target user's ID (user_id), a user-item 
# matrix representing interactions (user_item_matrix), and an optional parameter
# specifying the number of neighbors to find (k, defaulting to 5).
# 	The function starts by initializing an empty list called distances to
# store tuples containing the index of each user and their corresponding L1 norm
# distance from the target user. It iterates through each user in the user-item
# matrix, excluding the target user, and calculates the L1 norm distance using a
# helper function named l1_norm. The calculated distances are then appended to 
# the distances list as tuples.
# 	After computing distances for all users, the code proceeds to sort the
# distances list based on the calculated distances. It selects the top k neigh-
# bors with the smallest distances, and their indices are stored in the neigh-
# bors list. Finally, the function returns the indices of the k-nearest neigh-
# bors for the specified user. 

def k_nearest_neighbors(user_id, user_item_matrix, k=5):
    # Find k-nearest neighbors for the given user using L1 norm
    distances = []
    for index, row in user_item_matrix.iterrows():
        if index != user_id:
            distance = l1_norm(user_item_matrix.loc[user_id], row)
            distances.append((index, distance))
    
    # Sort by distance and select the top k neighbors
    distances.sort(key=lambda x: x[1])
    neighbors = [neighbor[0] for neighbor in distances[:k]]
    return neighbors

# recommend_articles():
# This function generates article recommendations for a given user based on col-
# laborative filtering. The function takes three parameters: the target user's 
# ID (user_id), a user-item matrix representing interactions (user_item_matrix),
# and an optional parameter specifying the number of nearest neighbors to consi-
# der (k, defaulting to 5).
# 	First, the function calls another function, k_nearest_neighbors, to find
# the k-nearest neighbors of the target user using a collaborative filtering ap-
# proach. These neighbors are identified based on their similarity in article 
# interactions. Next, the code combines the articles clicked by the identified 
# neighbors. It counts the occurrences of each article being clicked by any of 
# the neighbors and stores the result in the combined_articles array.
# 	Subsequently, the function identifies articles that have not been cli-
# cked by the target user but have been clicked by the neighbors. It compares 
# the target user's article interactions with the combined interactions of the 
# neighbors, identifying articles that the user has not interacted with but have
# been popular among the neighbors.
# 	Finally, the function returns a list of recommended articles for the gi-
# ven user. These recommendations are determined by finding articles that are 
# popular among the user's nearest neighbors but have not been interacted with 
# by the user.

def recommend_articles(user_id, user_item_matrix, k=5):
    # Get k-nearest neighbors for the user
    neighbors = k_nearest_neighbors(user_id, user_item_matrix, k)

    # Combine articles clicked by the neighbors
    neighbor_articles = user_item_matrix.loc[neighbors].values
    combined_articles = np.sum((neighbor_articles > 0).astype(int), axis=0)

    # Find articles not clicked by the user
    user_articles = user_item_matrix.loc[user_id].values
    recommendations = np.where((user_articles == 0) & (combined_articles > 0))[0]

    return recommendations

# Print top 5 recommendations for all users
for user_id in user_item_matrix.index:
    recommendations = recommend_articles(user_id, user_item_matrix, k=5)
    print(f"User {user_id}: Top 5 Recommendations - {recommendations[:5]}")

