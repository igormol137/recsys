# RFM-SVD Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# The Singular Value Decomposition (SVD) algorithm is used as a recommendation
# engine, incorporating the principles of Recency, Frequency, and Monetary (RFM)
# analysis. In this context, SVD factors users and items into latent vectors in
# a multidimensional space, capturing the underlying patterns in user-item
# interactions.
#
# The RFM analysis provides additional context by assigning each user an RFM
# score based on their recentness of purchases, frequency of transactions, and
# monetary spending. These RFM scores are integrated into the SVD model, such as
# to capture personalized user preferences.
#
# The collaborative filtering aspect of SVD allows the model to recommend items
# by identifying patterns and similarities between users with comparable RFM
# profiles. This combined approach leverages both the inherent structure of SVD
# and the business-focused insights from RFM analysis to generate more accurate
# and personalized recommendations for users.

import pandas as pd
import numpy as np
import datetime

# create_custom_dataset():
# This function creates a custom dataset from the given DataFrame by extracting
# unique users and items. It also generates mapping dictionaries to convert
# user and item IDs to corresponding numerical indices.
#
# Parameters:
# - df: The input DataFrame containing columns 'customer_id' and 'item_id'.
#
# Returns:
# - unique_users: A list of unique user IDs.
# - unique_items: A list of unique item IDs.
# - user_to_index: A dictionary mapping user IDs to numerical indices.
# - item_to_index: A dictionary mapping item IDs to numerical indices.

def create_custom_dataset(df):
    # Extract unique user and item IDs
    unique_users = df['customer_id'].unique()
    unique_items = df['item_id'].unique()

    # Create dictionaries for mapping user and item IDs to numerical indices
    user_to_index = {user: index for index, user in enumerate(unique_users)}
    item_to_index = {item: index for index, item in enumerate(unique_items)}

    # Return the generated lists and dictionaries
    return unique_users, unique_items, user_to_index, item_to_index

# get_ratings():
# This function generates a user-item ratings matrix based on the input dataset.
# It initializes a matrix of zeros and populates it with the purchase amounts
# from the dataset, mapping user and item IDs to corresponding numerical indices.
#
# Parameters:
# - dataset: A tuple containing information from the custom dataset generated
#   using the create_custom_dataset function. It includes unique_users, unique_items,
#   user_to_index, and item_to_index.
#
# Returns:
# - ratings: A 2D NumPy array representing the user-item ratings matrix,
#   where each row corresponds to a user, each column corresponds to an item,
#   and the matrix values are the purchase amounts.

def get_ratings(dataset):
    # Initialize a matrix of zeros with dimensions based on the number of unique users and items
    ratings = np.zeros((len(dataset[0]), len(dataset[1])))
    
    # Iterate over each row in the original dataset
    for _, row in df.iterrows():
        # Map user and item IDs to their corresponding numerical indices
        user_index = dataset[2][row['customer_id']]
        item_index = dataset[3][row['item_id']]
        
        # Set the value in the ratings matrix to the purchase amount
        ratings[user_index, item_index] = row['amount']
    
    # Return the generated ratings matrix
    return ratings


# calculate_rfm_scores():
# This function calculates the RFM (Recency, Frequency, Monetary) score for each
# customer based on their purchase history. It uses the maximum purchase date
# to compute recency, the number of purchases for frequency, and the sum of
# purchase amounts for monetary. The RFM scores are then combined into a single
# score using specified weightings.

# Parameters:
# - df: The input DataFrame containing purchase data with columns 'customer_id',
#   'purchase_date', and 'amount'.

# Returns:
# - rfm_score: A Series representing the calculated RFM scores for each customer.

def calculate_rfm_scores(df):
    # Find the current date as the maximum purchase date in the dataset
    current_date = df['purchase_date'].max()
    
    # Calculate recency as the number of days since the last purchase for each customer
    recency = (current_date - df.groupby('customer_id')['purchase_date'].max()).dt.days
    
    # Calculate frequency as the number of purchases for each customer
    frequency = df.groupby('customer_id').size()
    
    # Calculate monetary as the sum of purchase amounts for each customer
    monetary = df.groupby('customer_id')['amount'].sum()

    # Combine RFM scores into a single RFM score with specified weightings
    rfm_score = recency * 0.4 + frequency * 0.3 + monetary * 0.3
    
    # Return the calculated RFM scores as a Series
    return rfm_score


def create_custom_reader(rating_scale):
    return {'rating_scale': rating_scale}

# The next function creates and trains a Singular Value Decomposition with
# Personalized Probabilities (SVD++)-based collaborative filtering model.
# It takes a dataset, RFM scores, and optional hyperparameters as input, and
# returns trained user and item factors, global bias, and biases for users and items.
#
# Parameters:
# - dataset: Tuple of (users, items, user_indices, item_indices), where users
#   and items are lists of unique user and item identifiers, and user_indices
#   and item_indices are dictionaries mapping user and item identifiers to
#   their corresponding indices.
# - rfm_scores: Dictionary mapping user indices to RFM (Recency, Frequency,
#   Monetary) scores.
# - n_factors: Number of latent factors for the user and item matrices (default
#   is 50).
# - n_epochs: Number of training epochs (default is 20).
# - learning_rate: Learning rate for updating biases and factors (default is
#   0.005).
# - decay_factor: Decay factor for updating biases and factors to prevent
#   overfitting (default is 0.95).
#
# Initialization of user and item factors, biases, and global bias using random
# values and mean rating. User factors and item factors are initialized with
# random values. Global bias is set to the mean rating of all entries in the
# dataset. User biases and item biases are initialized to zeros.
#
# Training loop over epochs and data points in the dataset:
# - Retrieve user and item indices for the current data point.
# - Use the RFM score in the prediction function to compute the predicted
#   rating.
# - Calculate the prediction error.
# - Update user and item biases using stochastic gradient descent.
# - Update user and item factors with consideration for the RFM score and decay
#   factor, preventing overfitting.
# - Clip values to prevent overflow.
#
# Return a dictionary containing the trained model parameters:
# - 'user_factors': Trained user factors matrix.
# - 'item_factors': Trained item factors matrix.
# - 'global_bias': Global bias for the entire dataset.
# - 'user_biases': Trained biases for each user.
# - 'item_biases': Trained biases for each item.

def create_and_train_svdpp(dataset, rfm_scores, n_factors=50, n_epochs=20, learning_rate=0.005, decay_factor=0.95):
    # Initialize user and item factors matrices with random values
    user_factors = np.random.rand(len(dataset[0]), n_factors)
    item_factors = np.random.rand(len(dataset[1]), n_factors)
    
    # Calculate global bias as the mean of all ratings in the dataset
    global_bias = np.mean(get_ratings(dataset))
    
    # Initialize user and item biases with zeros
    user_biases = np.zeros(len(dataset[0]))
    item_biases = np.zeros(len(dataset[1]))

    # Training loop over epochs
    for epoch in range(n_epochs):
        # Iterate over each data point in the dataset
        for _, row in df.iterrows():
            # Retrieve user and item indices for the current data point
            user_index = dataset[2][row['customer_id']]
            item_index = dataset[3][row['item_id']]
            
            # Use the RFM score in the prediction function
            rfm_score = rfm_scores[user_index]
            predicted_rating = predict_svdpp(user_index, item_index, user_biases, item_biases, user_factors, item_factors, global_bias, rfm_score)
            
            # Calculate the prediction error
            error = row['amount'] - predicted_rating

            # Update biases using stochastic gradient descent
            user_biases[user_index] += learning_rate * (error - decay_factor * user_biases[user_index])
            item_biases[item_index] += learning_rate * (error - decay_factor * item_biases[item_index])

            # Update user and item factors with consideration for RFM score and decay factor
            user_factor_update = learning_rate * (error * (item_factors[item_index, :] + rfm_score) - decay_factor * user_factors[user_index, :])
            item_factor_update = learning_rate * (error * (user_factors[user_index, :] + item_biases[item_index]) - decay_factor * item_factors[item_index, :])

            # Manually clip values to prevent overflow
            max_value = 1e10
            min_value = -1e10
            user_factors[user_index, :] = np.maximum(min_value, np.minimum(max_value, user_factors[user_index, :] + user_factor_update))
            item_factors[item_index, :] = np.maximum(min_value, np.minimum(max_value, item_factors[item_index, :] + item_factor_update))

    # Return a dictionary containing the trained model parameters
    return {
        'user_factors': user_factors,
        'item_factors': item_factors,
        'global_bias': global_bias,
        'user_biases': user_biases,
        'item_biases': item_biases
    }


def predict_svdpp(user_index, item_index, user_biases, item_biases, user_factors, item_factors, global_bias, rfm_score):
    return (
        global_bias +
        user_biases[user_index] +
        item_biases[item_index] +
        np.dot(user_factors[user_index, :] + rfm_score, item_factors[item_index, :])
    )





def time_decay_weight(date, current_date, decay_factor=0.95):
    days_diff = (current_date - date).days
    return decay_factor ** days_diff

def weighted_prediction(predictions, current_date):
    for prediction in predictions:
        user_index = prediction['uid']
        item_index = prediction['iid']
        date = datetime.datetime.fromtimestamp(prediction['r_ui'])
        weight = time_decay_weight(date, current_date)
        prediction['est'] = prediction['est'] * weight
    return predictions

# generate_recommendations():
# This function organizes a list of weighted predictions into a dictionary of
# recommendations for each user. It ensures that each recommended item is unique
# for a user and stores the items along with their estimated scores.
#
# Parameters:
# - weighted_predictions: A list of weighted predictions, each containing 'uid',
#   'iid', and 'est' values.
# - dataset: A tuple containing information from the custom dataset generated
#   using the create_custom_dataset function.
#
# Returns:
# A dictionary of recommendations for each user. It has user IDs as keys, and for
# each user, a dictionary with 'items' and 'scores' containing recommended items
# and their scores, respectively.
    
def generate_recommendations(weighted_predictions, dataset):
    # Initialize an empty dictionary to store recommendations for each user
    recommendations = {}
    
    # Iterate over each prediction in the list of weighted predictions
    for prediction in weighted_predictions:
        # Extract user and item IDs from the prediction
        user_id = prediction['uid']
        item_id = prediction['iid']
        
        # Check if the user is not already in the recommendations dictionary
        if user_id not in recommendations:
            # If not, add the user to the dictionary with empty 'items' and 'scores'
            recommendations[user_id] = {'items': set(), 'scores': []}
        
        # Check if the item is not already recommended to the user
        if item_id not in recommendations[user_id]['items']:
            # If not, add the item to the user's set of recommended items
            recommendations[user_id]['items'].add(item_id)
            
            # Append the item and its estimated score to the user's scores list
            recommendations[user_id]['scores'].append((item_id, prediction['est']))
    
    # Return the generated dictionary of recommendations for each user
    return recommendations


# sort_and_print_recommendations():
# This function sorts the recommendations for each user based on the associated
# scores in descending order and prints the top 5 recommended items for each user.
#
# Parameters:
# - recommendations: A dictionary of recommendations for each user. It has user IDs
#   as keys, and for each user, a dictionary with 'items' and 'scores' containing
#   recommended items and their scores, respectively.
# - dataset: A tuple containing information from the custom dataset generated
#   using the create_custom_dataset function.

def sort_and_print_recommendations(recommendations, dataset):
    # Iterate over each user in the recommendations dictionary
    for user_id, rec_data in recommendations.items():
        # Sort the recommended items based on their scores in descending order
        rec_data['scores'].sort(key=lambda x: x[1], reverse=True)
        
        # Extract the top 5 recommended items for each user
        recommended_items = [dataset[1][item_id] for item_id, _ in rec_data['scores'][:5]]
        
        # Print the recommendations for the current user
        print(f"Recommendations for Customer {user_id}: {recommended_items}")

# The main routine orchestrates the recommendation system pipeline, and performs
# the following activities:
#
#      - Loads purchase data from a CSV file.
#      - Processes 'purchase_date' as datetime.
#      - Creates a custom dataset.
#      - Calculates RFM scores and combines them into a single RFM score.
#      - Creates a custom reader.
#      - Trains an SVD++ model with RFM scores.
#      - Makes predictions for all customers with time decay.
#      - Weights predictions based on time decay.
#      - Generates recommendations for each user.
#      - Sorts and prints the top recommendations to the console.

def main():
    # Load the CSV file
    file_path = "./generated_data.csv"
    df = pd.read_csv(file_path)

    # Convert 'purchase_date' to datetime
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    # Create a custom dataset
    my_dataset = create_custom_dataset(df)

    # Calculate RFM scores and combine them into a single RFM score
    rfm_score = calculate_rfm_scores(df)
    df['rfm_score'] = rfm_score[df['customer_id'].values].values

    # Create a custom reader
    my_reader = create_custom_reader(rating_scale=(df['amount'].min(), df['amount'].max()))

    # Create and train a custom SVD++ model
    my_svdpp_model = create_and_train_svdpp(my_dataset, df['rfm_score'].values, n_factors=50, n_epochs=20)

    # Make predictions for all customers with time decay
    current_date = df['purchase_date'].max()
    predictions = []
    for _, row in df.iterrows():
        user_index = my_dataset[2][row['customer_id']]
        item_index = my_dataset[3][row['item_id']]
        predicted_amount = predict_svdpp(user_index, item_index, my_svdpp_model['user_biases'],
                                         my_svdpp_model['item_biases'], my_svdpp_model['user_factors'],
                                         my_svdpp_model['item_factors'], my_svdpp_model['global_bias'],
                                         df['rfm_score'][user_index])
        predictions.append({'uid': user_index, 'iid': item_index, 'r_ui': row['purchase_date'].timestamp(),
                            'est': predicted_amount})

    # Apply time decay to predictions
    weighted_predictions = weighted_prediction(predictions, current_date)

    # Display recommendations for all customers without repeats
    recommendations = generate_recommendations(weighted_predictions, my_dataset)

    # Sort and print recommendations
    sort_and_print_recommendations(recommendations, my_dataset)


if __name__ == "__main__":
    main()

