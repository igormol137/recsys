# Bayesian Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The Bayesian Personalized Ranking (BPR) recommendation system, integrated
# with Recency, Frequency, Monetary (RFM) analysis, employs a collaborative
# filtering approach to provide personalized item recommendations. In this
# system, each user and item are represented by latent factors, which are
# learned during the training process.
#
# BPR focuses on optimizing the ranking of positive interactions (purchases)
# over negative interactions (non-purchases) for each user. The RFM analysis
# enhances the recommendation model by incorporating user-specific metrics
# related to the recency, frequency, and monetary value of their interactions.
#
# By considering these RFM scores during training, the system adapts its
# recommendations to user preferences based not only on historical
# interactions but also on the user's specific purchase behavior patterns.
#
# We hope this hybrid approach may improve the accuracy and relevance of item
# recommendations for individual users in the recommendation system.


import pandas as pd
import numpy as np

# The next function calculates RFM (Recency, Frequency, Monetary) scores for
# each customer based on their purchase history.
#
# Parameters:
# - data: DataFrame containing purchase history with columns 'customer_id',
#   'purchase_date', 'item_id', and 'amount'.
#
# Returns:
# - DataFrame with customer ID and unified RFM scores.

def calculate_rfm_scores(data):
    # Get the current date from the data
    current_date = data['purchase_date'].max()

    # Group data by customer and calculate recency, frequency, and monetary metrics
    rfm_data = data.groupby('customer_id').agg({
        'purchase_date': lambda x: (current_date - x.max()).days,
        'item_id': 'count',
        'amount': 'sum'
    }).reset_index()

    # Normalize and score each metric
    rfm_data['recency_score'] = rfm_data['purchase_date'] / rfm_data['purchase_date'].max()
    rfm_data['frequency_score'] = rfm_data['item_id'] / rfm_data['item_id'].max()
    rfm_data['monetary_score'] = rfm_data['amount'] / rfm_data['amount'].max()

    # Calculate unified RFM score by summing individual scores
    rfm_data['unified_rfm_score'] = rfm_data['recency_score'] + rfm_data['frequency_score'] + rfm_data['monetary_score']

    # Return DataFrame with customer ID and unified RFM scores
    return rfm_data[['customer_id', 'unified_rfm_score']]


# The function create_user_item_matrix() creates a user-item matrix from purcha-
# se history data, where rows represent users, columns represent items, and the 
# values indicate the number of times each user has interacted with each item.
#
# Parameters:
# - data: DataFrame containing purchase history with columns 'customer_id',
#   'item_id', and 'amount'.
#
# Returns:
# - User-item matrix with rows representing users, columns representing items,
#   and values representing interaction counts.

def create_user_item_matrix(data):
    return pd.pivot_table(data, values='amount', index='customer_id', columns='item_id', aggfunc='count', fill_value=0)

# The sigmoid function returns σ(x) := 1/(1+e^(-x)) for a given input 'x'.

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The function train_bpr trains a Bayesian Personalized Ranking (BPR) model with
# stochastic gradient descent. It takes as input a user-item matrix, RFM data,
# and several optional hyperparameters. The goal is to learn latent factors for
# users and items, incorporating RFM scores in the training process. The model is
# trained for a specified number of epochs, adjusting the latent factors to
# improve ranking performance.
#
# Parameters:
# - user_item_matrix: Matrix representing user-item interactions, where rows
#   are users and columns are items.
# - rfm_data: DataFrame containing RFM (Recency, Frequency, Monetary) data for
#   users.
# - learning_rate: Step size for gradient descent (default is 0.01).
# - regularization: Regularization term to prevent overfitting (default is 0.01).
# - latent_factors: Number of latent factors for users and items (default is 5).
# - epochs: Number of training epochs (default is 10).
# - rfm_weight: Weight of RFM score in the gradient calculation (default is 0.1).

def train_bpr(user_item_matrix, rfm_data, learning_rate=0.01, regularization=0.01, latent_factors=5, epochs=10, rfm_weight=0.1):
    # Get the number of users and items in the user-item matrix
    num_users, num_items = user_item_matrix.shape

    # Initialize random latent factors for users and items
    user_latent_factors = np.random.rand(num_users, latent_factors)
    item_latent_factors = np.random.rand(num_items, latent_factors)

    # Iterate over training epochs
    for epoch in range(epochs):
        # Iterate over users in the user-item matrix
        for user_idx, row in user_item_matrix.iterrows():
            # Find items with positive interactions (purchased items)
            positive_items = np.where(row > 0)[0]
            if len(positive_items) == 0:
                continue

            # Randomly sample a positive item
            sampled_item = np.random.choice(positive_items)

            # Find items with no interactions (negative items)
            negative_items = np.where(row == 0)[0]
            if len(negative_items) == 0:
                continue

            # Randomly sample a negative item
            sampled_negative_item = np.random.choice(negative_items)

            # Calculate the difference in scores for the positive and negative items
            x_uij = np.dot(user_latent_factors[user_idx], item_latent_factors[sampled_item]) - \
		np.dot(user_latent_factors[user_idx], item_latent_factors[sampled_negative_item])

            sigmoid_x_uij = sigmoid(x_uij)

            # Include RFM score in the gradient calculation
            rfm_score = rfm_data[rfm_data['customer_id'] == user_idx]['unified_rfm_score'].values[0]

            # Compute gradients for user and items
            user_gradient = (sigmoid_x_uij * (item_latent_factors[sampled_negative_item] -
                             item_latent_factors[sampled_item]) + regularization * user_latent_factors[user_idx]) * rfm_score
            item_gradient_positive = (sigmoid_x_uij * user_latent_factors[user_idx] +
                                     regularization * item_latent_factors[sampled_item]) * rfm_score
            item_gradient_negative = (-sigmoid_x_uij * user_latent_factors[user_idx] +
                                     regularization * item_latent_factors[sampled_negative_item]) * rfm_score

            # Update latent factors using gradient descent
            user_latent_factors[user_idx] += learning_rate * user_gradient
            item_latent_factors[sampled_item] += learning_rate * item_gradient_positive
            item_latent_factors[sampled_negative_item] += learning_rate * item_gradient_negative

    # Return the learned user and item latent factors
    return user_latent_factors, item_latent_factors


# The next function predicts interaction scores for all users and items based on
# the learned user/items relations.
#
# Parameters:
# - user_latent_factors: Latent factors for users learned during training.
# - item_latent_factors: Latent factors for items learned during training.
#
# Returns:
# - A matrix of predicted interaction scores for all users and items.

def predict_all(user_latent_factors, item_latent_factors):
    # Calculate the dot product of user and item latent factors
    return np.dot(user_latent_factors, item_latent_factors.T)


# The function recommend_top_k() recommends top k items for a given user based 
# on learned users/items relations. It considers the user's past interactions,
# excludes items already purchased, and incorporates the user's RFM score to
# prioritize recommendations.
#
# Parameters:
# - user_id: The ID of the target user.
# - user_latent_factors: Latent factors for users learned during training.
# - item_latent_factors: Latent factors for items learned during training.
# - user_item_matrix: Matrix representing user-item interactions.
# - rfm_data: DataFrame containing RFM (Recency, Frequency, Monetary) data.
# - k: Number of items to recommend (default is 5).
#
# Returns:
# - A list of top k recommended item indices.

def recommend_top_k(user_id, user_latent_factors, item_latent_factors, user_item_matrix, rfm_data, k=5):
    # Check if user_id exists in the user_item_matrix
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found in the user_item_matrix.")
        return []

    # Get the positional index of the user in the user_item_matrix
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    # Predict scores for all items for the target user
    user_scores = predict_all(user_latent_factors, item_latent_factors)[user_idx]

    # Identify items already purchased by the user
    already_purchased = np.where(user_item_matrix.loc[user_id].values > 0)[0]
    user_scores[already_purchased] = -np.inf  # Set already purchased items to -inf

    # Retrieve unified RFM score for the user and multiply by the scores
    rfm_score_multiplier = rfm_data[rfm_data['customer_id'] == user_id]['unified_rfm_score'].values[0]
    user_scores *= rfm_score_multiplier  # Multiply by unified RFM score

    # Get the indices of the top k items based on scores
    top_k_items = np.argpartition(user_scores, -k)[-k:]

    # Return the top k items in descending order of scores
    return top_k_items[np.argsort(user_scores[top_k_items])][::-1]
    
# Loads data from a CSV file into a pandas DataFrame.

def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocesses data by converting 'purchase_date' to datetime format and 
# sorts the DataFrame by 'purchase_date' in descending order.

def preprocess_data(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    return data.sort_values(by='purchase_date', ascending=False)

# The main function calls the following routines:
#
# 1. Loads data from a CSV file located at "./generated_data.csv".
# 2. Preprocesses the loaded data.
# 3. Calculates RFM (Recency, Frequency, Monetary) scores for the preprocessed data.
# 4. Creates a user-item matrix based on the preprocessed data.
# 5. Trains Bayesian Personalized Ranking (BPR) to obtain latent factors for users and items.
# 6. For each user in the user-item matrix, recommends the top items using the trained BPR model and prints the results.

def main():
    # Set the path to the CSV file containing the data
    file_path = "./generated_data.csv"
    
    # Step 1: Load data from the specified CSV file
    data = load_data(file_path)
    
    # Step 2: Preprocess the loaded data
    preprocessed_data = preprocess_data(data)
    
    # Step 3: Calculate RFM scores for the preprocessed data
    rfm_data = calculate_rfm_scores(preprocessed_data)
    
    # Step 4: Create a user-item matrix based on the preprocessed data
    user_item_matrix = create_user_item_matrix(preprocessed_data)
    
    # Step 5: Train BPR to obtain latent factors for users and items
    user_latent_factors, item_latent_factors = train_bpr(user_item_matrix, rfm_data)
    
    # Step 6: For each user in the user-item matrix, recommend top items using the trained BPR model and print the results
    for user_id, _ in user_item_matrix.iterrows():
        top_recommendations = recommend_top_k(user_id, user_latent_factors, item_latent_factors, user_item_matrix, rfm_data)
        print(f"\nTop recommendations for user {user_id}: {top_recommendations}")


if __name__ == "__main__":
    main()
