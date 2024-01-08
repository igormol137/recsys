# RFM/Matrix Recommender Engine
#
# Igor Mol <igor.mol@makes.ai>
#
# A recommender system within the RFM (Recency, Frequency, Monetary) model is
# implemented using the cosine similarity metric to measure the similarity
# between users based on their purchase behavior. The RFM model focuses on
# assessing customers' recency of purchases, the frequency of transactions, and
# the monetary value of their spending. By representing users as vectors in a
# multidimensional space, where each dimension corresponds to one of these RFM
# metrics, cosine similarity calculates the cosine of the angle between these
# vectors. The resulting similarity scores indicate how closely the purchasing
# patterns of different users align. In the recommender system, higher cosine
# similarity values imply greater similarity between users, facilitating the
# identification of customers with comparable preferences. This information is
# then leveraged to generate personalized recommendations for users, enhancing
# the system's ability to suggest items that align with their historical
# purchasing behavior.

import pandas as pd
import numpy as np

# load_data():
# This function loads data from a CSV file, processes the 'purchase_date' column
# as a datetime object, and returns the resulting DataFrame.
#
# Parameters:
# - file_path: The file path to the CSV file containing the data.
#
# Returns:
# A pandas DataFrame containing the loaded data, with the 'purchase_date'
# column converted to datetime format.

def load_data(file_path):
    # Read data from the specified CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Convert the 'purchase_date' column to datetime format
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    
    # Return the processed DataFrame
    return data

# calculate_rfm_scores():
# This function calculates RFM (Recency, Frequency, Monetary) scores for each
# customer based on their purchase history. It assigns scores to customers based
# on how recent their last purchase was (Recency), how frequently they make
# purchases (Frequency), and how much money they spend (Monetary).
#
# Parameters:
# - data: DataFrame containing purchase data with columns 'customer_id',
#   'purchase_date', 'item_id', and 'amount'.
#
# Returns:
# A DataFrame with columns 'customer_id' and 'rfm_score', representing the
# calculated RFM scores for each customer.

def calculate_rfm_scores(data):
    # Find the current date as the maximum purchase date in the dataset
    current_date = data['purchase_date'].max()
    
    # Group the data by customer_id and aggregate metrics for recency, frequency,
    # and monetary
    rfm_data = data.groupby('customer_id').agg({
        'purchase_date': lambda x: (current_date - x.max()).days,
        'item_id': 'count',
        'amount': 'sum'
    }).reset_index()

    # Calculate relative scores for recency, frequency, and monetary metrics
    rfm_data['recency_score'] = rfm_data['purchase_date'] / rfm_data['purchase_date'].max()
    rfm_data['frequency_score'] = rfm_data['item_id'] / rfm_data['item_id'].max()
    rfm_data['monetary_score'] = rfm_data['amount'] / rfm_data['amount'].max()

    # Combine the scores to calculate the overall RFM score for each customer
    rfm_data['rfm_score'] = rfm_data['recency_score'] + rfm_data['frequency_score'] + rfm_data['monetary_score']

    # Return a DataFrame with customer_id and rfm_score columns
    return rfm_data[['customer_id', 'rfm_score']]

def create_user_item_matrix(data):
    return pd.pivot_table(data, values='amount', index='customer_id', columns='item_id', aggfunc='count', fill_value=0)

# Cosine similarity:
# This function computes the cosine similarity matrix from scratch for a given input matrix.
# Cosine similarity measures the cosine of the angle between two non-zero vectors and
# is often used to assess the similarity between rows of a matrix.
#
# Parameters:
# - matrix: The input matrix for which cosine similarity is to be computed.
#
# Returns:
# A pandas DataFrame representing the cosine similarity matrix, where each entry (i, j)
# represents the cosine similarity between row i and row j of the input matrix.

def compute_cosine_similarity_from_scratch(matrix):
    # Normalize the input matrix by dividing each row by its Euclidean norm
    matrix_normalized = matrix.div(np.linalg.norm(matrix, axis=1), axis=0)
    
    # Calculate the cosine similarity matrix by taking the dot product of the normalized matrix with its transpose
    similarity_matrix = matrix_normalized @ matrix_normalized.T
    
    # Return the resulting similarity matrix as a DataFrame with row and column indices from the original matrix
    return pd.DataFrame(similarity_matrix, index=matrix.index, columns=matrix.index)


# get_recommendations():
# This function provides personalized recommendations for a given user based on
# collaborative filtering. It calculates recommendations by considering the user's
# past purchases, the similarity between the user and other users, and additional
# information from RFM (Recency, Frequency, Monetary) data.
#
# Parameters:
# - user_id: The identifier of the target user for whom recommendations are generated.
# - similarity_matrix: A matrix representing the similarity between users.
# - user_item_matrix: A matrix representing the user-item interactions (purchases).
# - rfm_data: Additional data containing RFM scores for customers.
# - n: The number of recommendations to return (default is 5).
#
# Returns:
# A list of tuples containing recommended items for the user, where each tuple
# consists of (item_id, recommendation_score).

def get_recommendations(user_id, similarity_matrix, user_item_matrix, rfm_data, n=5):
    # Extract the user's past purchases from the user-item matrix
    user_purchases = user_item_matrix.loc[user_id]

    # Get a list of similar users, sorted by similarity in descending order
    similar_users = similarity_matrix[user_id].sort_values(ascending=False).index[1:]

    # Initialize an empty list to store recommendations
    recommendations = []

    # Iterate over all items in the user-item matrix
    for item_id in user_item_matrix.columns:
        # Check if the user has not purchased the item
        if user_purchases[item_id] == 0:
            # Calculate the weighted sum of similar users' purchases for the item
            weighted_sum = sum(
                user_item_matrix.loc[similar_user, item_id] *
                similarity_matrix.loc[user_id, similar_user] *
                rfm_data[rfm_data['customer_id'] == similar_user]['rfm_score'].values[0]
                for similar_user in similar_users
            )
            # Append the item and its weighted sum to the recommendations list
            recommendations.append((item_id, weighted_sum))

    # Sort recommendations by the calculated scores in descending order and select
    # the top 'n'
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]

    # Return the final list of recommendations
    return recommendations


# print_recommendations():
# This function generates and prints personalized recommendations for each user
# in the user-item matrix based on collaborative filtering. It utilizes the
# get_recommendations function to calculate recommendations for each user.
#
# Parameters:
# - user_item_matrix: A matrix representing the user-item interactions (purchases).
# - similarity_matrix: A matrix representing the similarity between users.
# - rfm_data: Additional data containing RFM scores for customers.

def print_recommendations(user_item_matrix, similarity_matrix, rfm_data):
    # Iterate over each user in the user-item matrix
    for user_id in user_item_matrix.index:
        # Get personalized recommendations for the current user using collaborative filtering
        recommendations = get_recommendations(user_id, similarity_matrix, user_item_matrix, rfm_data)
        
        # Print header for user recommendations
        print(f"\nTop recommendations for user {user_id}:")
        
        # Iterate over each recommended item and its score, and print the details
        for item_id, score in recommendations:
            print(f"Item {item_id} - Score: {score}")

# Main routine:
# The next function is the entry point for a recommendation system pipeline.
# It loads data from a CSV file, calculates Recency-Frequency-Monetary scores
# for each customer, creates a user-item matrix representing interactions,
# computes cosine similarity between users, and finally prints personalized
# recommendations for each user.
#
# Parameters:
# - file_path: The file path to the CSV file containing the purchase data.

def main(file_path):
    # Load purchase data from the specified CSV file
    data = load_data(file_path)
    
    # Calculate RFM scores for each customer based on their purchase history
    rfm_data = calculate_rfm_scores(data)
    
    # Create a user-item matrix representing user-item interactions (purchases)
    user_item_matrix = create_user_item_matrix(data)
    
    # Compute cosine similarity between users using the user-item matrix
    similarity_matrix = compute_cosine_similarity_from_scratch(user_item_matrix)
    
    # Print personalized recommendations for each user based on collaborative filtering
    print_recommendations(user_item_matrix, similarity_matrix, rfm_data)

if __name__ == "__main__":
    file_path = "./generated_data.csv"
    main(file_path)
