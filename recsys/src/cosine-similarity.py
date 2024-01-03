# cosine-similarity.py
#
# Igor Mol <igor.mol@makes.ai>
#
# The following program implements a collaborative filtering recommender system, 
# a type of system that predicts a user's preferences based on the preferences of 
# similar users. It primarily utilizes a user-item matrix, where rows correspond to 
# users, columns correspond to articles, and the matrix values represent user interactions
# or session sizes with specific articles.
# 	The core functionality lies in the computation of cosine similarity from scratch, 
# without relying on external libraries. Cosine similarity is a measure of the cosine of 
# the angle between two vectors, and in this context, it quantifies the similarity 
# between users' interaction patterns. The code then uses this similarity information
# to identify the top k similar users for each user and generate article recommendations
# based on the interactions of those similar users. The resulting recommendations are printed 
# for all users, providing a straightforward collaborative filtering approach to suggest
# articles to users based on their historical interactions and similarities with other users.

import numpy as np
import pandas as pd

# Function to create the user-item matrix from the interactions DataFrame
def create_user_item_matrix(interactions_df):
    return pd.pivot_table(interactions_df, values='session_size', index='user_id', columns='click_article_id', fill_value=0)

# cosine_similarity():
# This function calculates the cosine similarity between users based on a user-item
# matrix. The function takes a user-item matrix as input, where rows represent users,
# columns represent items, and matrix values indicate user-item interactions. It
# initializes an empty similarity matrix with zeros, with dimensions equal to the
# number of users. It then iterates through each pair of users, computing the cosine
# similarity between their vectors using dot products and vector norms. The resulting
# similarity values are stored in the similarity matrix. Self-similarity values are
# set to zero, and the final similarity matrix is converted to a Pandas-like format 
# for convenient indexing, with rows and columns labeled by user identifiers. The 
# function returns this DataFrame representing the cosine similarity between users.
#
def cosine_similarity_from_scratch(user_item_matrix):
    user_item_array = user_item_matrix.values
    similarity_matrix = np.zeros((user_item_array.shape[0], user_item_array.shape[0]))
    
    for i in range(user_item_array.shape[0]):
        for j in range(user_item_array.shape[0]):
            # Compute the dot product of the two user vectors
            dot_product = np.dot(user_item_array[i, :], user_item_array[j, :])
            
            # Compute the Euclidean norm (magnitude) of each user vector
            norm_i = np.linalg.norm(user_item_array[i, :])
            norm_j = np.linalg.norm(user_item_array[j, :])
            
            # Compute the cosine similarity
            similarity_matrix[i, j] = dot_product / (norm_i * norm_j) if norm_i != 0 and norm_j != 0 else 0

    # Set self-similarity to zero
    np.fill_diagonal(similarity_matrix, 0)
    
    # Convert the similarity matrix to a DataFrame for easy indexing
    return pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# get_top_k_similar_users():
# This function is designed to retrieve the top-k most similar users to a given
# user in a recommender system. The function takes three parameters: the target 
# user's ID (user_id), a matrix or dataframe representing the similarity scores 
# between users (similarity_matrix), and an optional parameter specifying the 
# number of similar users to retrieve (k, defaulting to 5).
# 	In the code, the similarity matrix is assumed to be organized such that 
# rows and columns correspond to user IDs, and the values in the matrix repre-
# sent the similarity scores between users. The function starts by extracting 
# the row corresponding to the target user from the similarity matrix. Using the
# nlargest method, it then identifies the top-k + 1 users with the highest simi-
# larity scores, excluding the target user itself from the results. The function
# returns a list of user IDs representing the top-k most similar users to the 
# given user.

def get_top_k_similar_users(user_id, similarity_matrix, k=5):
    similar_users = similarity_matrix.loc[user_id].nlargest(k + 1).index[1:]  # Exclude the user itself
    return similar_users

# recommend_articles()
# The function generates personalized article recommendations for a user in a 
# recommender system. The function takes four parameters: the user's ID 
# (user_id), a matrix representing user-item interactions (user_item_matrix), a 
# similarity matrix indicating how similar users are (similarity_matrix), and an
# optional parameter specifying the number of similar users to consider (k, 
# defaulting to 5).
# 	The first step in the code is to obtain the top-k most similar users to 
# the target user using the get_top_k_similar_users function. Next, it identifies
# the articles that the target user has not interacted with (not_interacted_articles) 
# by examining the user-item interaction matrix. The core of the recommendation 
# process involves calculating a weighted average of the interactions of similar 
# users for the articles that the target user has not interacted with. The weights
# are determined by the similarity scores between the target user and the similar
# users. This step effectively leverages the preferences of users with similar 
# tastes to enhance the accuracy of recommendations.
# 	Finally, the recommendations are sorted based on their weighted interac-
# tion scores in descending order, and the function returns the indices of the 
# recommended articles. 

def recommend_articles(user_id, user_item_matrix, similarity_matrix, k=5):
    similar_users = get_top_k_similar_users(user_id, similarity_matrix, k)
    
    user_articles = user_item_matrix.loc[user_id]
    not_interacted_articles = user_articles[user_articles == 0].index
    
    # Calculate the weighted average of similar users' interactions for not interacted articles
    recommendations = user_item_matrix.loc[similar_users, not_interacted_articles].multiply(similarity_matrix.loc[user_id, similar_users], axis=0).sum()
    
    # Sort recommendations by interaction score in descending order
    recommendations = recommendations.sort_values(ascending=False)
    
    return recommendations.index

# Function to print top recommendations for all users
def print_top_recommendations_for_all_users(user_item_matrix, similarity_matrix, k=5):
    for user_id in user_item_matrix.index:
        recommendations = recommend_articles(user_id, user_item_matrix, similarity_matrix, k)
        print(f"Top recommendations for User {user_id}: {recommendations.values[:5]}")
        
# Load the original CSV file
file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
df = pd.read_csv(file_path)

# Assuming df is your DataFrame
user_item_matrix = create_user_item_matrix(df[['user_id', 'click_article_id', 'session_size']])

# Using cosine similarity from scratch
user_similarity_matrix_scratch = cosine_similarity_from_scratch(user_item_matrix)

# Print top recommendations for all users using the manually computed similarity matrix
print_top_recommendations_for_all_users(user_item_matrix, user_similarity_matrix_scratch, k=5)
