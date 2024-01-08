# k-NN Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The RFM (Recency, Frequency, Monetary) model in recommender systems analyzes 
# customer transaction data to calculate recency, frequency, and monetary 
# scores. These scores contribute to a unified RFM score, providing a general 
# measure of a customer's engagement and spending behavior.
#
# In collaborative filtering, the k-NN (k-Nearest Neighbors) algorithm is used
# to produce recommendations. It computes the similarity between users or items 
# based on their RFM scores. By identifying the k-nearest neighbors, the 
# algorithm suggests items by considering the preferences of these neighbors. 
# This approach enhances personalized recommendations by utilizing the RFM 
# model to capture user behavior patterns.

import pandas as pd
import numpy as np

# Define cosine similarity function
def cosine_measure(vector1, vector2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = sum(v ** 2 for v in vector1) ** 0.5
    magnitude2 = sum(v ** 2 for v in vector2) ** 0.5

    similarity = dot_product / (magnitude1 * magnitude2 + 1e-10)  # To avoid division by zero
    return similarity

# The calculate_unified_rfm_scores function computes unified RFM (Recency, 
# Frequency, Monetary) scores for each customer in the provided DataFrame.
# The primary objectives include calculating recency, frequency, monetary values, 
# normalizing these values, and finally, combining them into a single unified RFM 
# score.
#
# Parameters:
#   - df: The DataFrame containing customer transactions with columns like 
#         'customer_id', 'purchase_date', 'item_id', and 'amount'.
#
# Returns:
#   - rfm_data: A DataFrame with customer-wise unified RFM scores, including 
#               columns 'customer_id', 'recency', 'frequency', 'monetary', and 
#               'unified_score'.
#
# Processing Steps:
#   - Determine the current date from the maximum purchase date in the DataFrame.
#   - Group the data by customer_id and calculate recency, frequency, and monetary 
#     values.
#   - Normalize recency, frequency, and monetary values individually.
#   - Compute the unified RFM score as the average of the normalized recency, 
#     frequency, and monetary values.
#   - Return the resulting DataFrame with customer-wise unified RFM scores.

def calculate_unified_rfm_scores(df):
    current_date = df['purchase_date'].max()
    rfm_data = df.groupby('customer_id').agg({
        'purchase_date': lambda x: (current_date - x.max()).days,  # Recency
        'item_id': 'count',  # Frequency
        'amount': 'sum'  # Monetary
    }).reset_index()

    # Normalize and create a unified RFM score
    rfm_data['recency'] = rfm_data['purchase_date'] / rfm_data['purchase_date'].max()
    rfm_data['frequency'] = rfm_data['item_id'] / rfm_data['item_id'].max()
    rfm_data['monetary'] = rfm_data['amount'] / rfm_data['amount'].max()

    # Calculate unified RFM score
    rfm_data['unified_score'] = (rfm_data['recency'] + rfm_data['frequency'] + rfm_data['monetary']) / 3

    return rfm_data

# The calculate_similarity function computes the similarity matrix between 
# users based on their unified RFM scores. The main objective is to measure the 
# cosine similarity between users, providing a pairwise similarity matrix.
#
# Parameters:
#   - rfm_data: A DataFrame containing customer-wise unified RFM scores with 
#               columns like 'customer_id' and 'unified_score'.
#
# Returns:
#   - similarity_matrix: A 2D list representing the pairwise cosine similarity 
#                        between users. Each element (i, j) corresponds to the 
#                        cosine similarity between users i and j.
#
# Computation Steps:
#   - Obtain the number of users in the DataFrame.
#   - Create a unified RFM matrix by reshaping the 'unified_score' column.
#   - Compute the pairwise cosine similarity between users using the cosine_measure 
#     function, resulting in a similarity matrix.
#   - Return the computed similarity matrix.

def calculate_similarity(rfm_data):
    num_users = rfm_data.shape[0]
    unified_rfm_matrix = np.array(rfm_data['unified_score']).reshape(-1, 1)
    similarity_matrix = [[cosine_measure(unified_rfm_matrix[i], unified_rfm_matrix[j]) for j in range(num_users)] for i in range(num_users)]
    return similarity_matrix

# The get_top_recommendations function generates top recommendations for each 
# user based on collaborative filtering using unified RFM scores. The goal is 
# to predict user preferences for items the user has not interacted with.
#
# Parameters:
#   - user_item_matrix: A DataFrame representing user-item interactions, where 
#                       rows are users, columns are items, and values indicate 
#                       interactions (e.g., purchase history).
#   - rfm_similarity: A 2D list representing cosine similarity between users 
#                     based on unified RFM scores.
#   - num_recommendations: The number of top recommendations to generate for each 
#                          user (default is 5).
#
# Returns:
#   - top_recommendations: A dictionary where keys are user IDs, and values are 
#                          lists of tuples containing item IDs and corresponding 
#                          predicted ratings. Each user's recommendations are 
#                          sorted in descending order of predicted ratings.
#
# Recommendation Generation Steps:
#   - Obtain the number of users and a list of all user IDs.
#   - Iterate over each user to predict ratings for items not yet interacted with.
#   - Calculate predicted ratings using collaborative filtering based on 
#     unified RFM scores.
#   - Sort the predictions in descending order of estimated ratings.
#   - Store the top recommendations for each user in a dictionary and return it.

def get_top_recommendations(user_item_matrix, rfm_similarity, num_recommendations=5):
    num_users = len(rfm_similarity)
    all_user_ids = user_item_matrix.index

    top_recommendations = {}
    for user_id in all_user_ids:
        # Get items the user has not bought yet
        items_to_predict = user_item_matrix.columns[~user_item_matrix.loc[user_id].astype(bool)]

        # Calculate predicted ratings for the items using unified RFM scores
        predicted_ratings = []
        for item_id in items_to_predict:
            numerator = sum(rfm_similarity[user_id][j] * user_item_matrix.iloc[j][item_id] for j in range(num_users))
            denominator = sum(rfm_similarity[user_id][j] for j in range(num_users))
            predicted_rating = numerator / (denominator + 1e-10)  # To avoid division by zero
            predicted_ratings.append((item_id, predicted_rating))

        # Sort predictions by estimated rating in descending order
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)

        # Get top recommendations
        top_recommendations[user_id] = predicted_ratings[:num_recommendations]

    return top_recommendations

# Main routine
def main():
    # Load the CSV file into a DataFrame
    csv_file_path = "./generated_data.csv"
    df = pd.read_csv(csv_file_path)

    # Convert 'purchase_date' to datetime format
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    # Calculate unified RFM scores
    rfm_data = calculate_unified_rfm_scores(df)

    # Create a user-item matrix
    user_item_matrix = pd.pivot_table(df, values='amount', index='customer_id', columns='item_id', fill_value=0)

    # Calculate cosine similarity between users based on unified RFM score
    rfm_similarity = calculate_similarity(rfm_data)

    # Get top recommendations for all users
    top_recommendations = get_top_recommendations(user_item_matrix, rfm_similarity)

    # Print top recommendations for all users
    for user_id, recommendations in top_recommendations.items():
        print(f"User {user_id}:")
        for item_id, rating in recommendations:
            print(f"  - Item {item_id}: Estimated Rating = {rating:.2f}")
        print()

if __name__ == "__main__":
    main()
