# knn.py
#
# Igor Mol <igor.mol@makes.ai>
#
# The k-nearest neighbors (KNN) algorithm is a technique used in
# recommender systems to suggest items to users based on their similarity to
# other users. In this algorithm, each user is represented in a
# multidimensional space, where the dimensions correspond to various features
# or characteristics of the items they have interacted with.
# The KNN algorithm works as follows:
#
# 1. User Representation:
#    Users are represented as points in a space, where each coordinate
#    corresponds to a different aspect of their interaction history, such as
#    the articles they clicked on or rated.
#
# 2. Item Similarity:
#    The algorithm identifies the similarity between items based on user
#    interactions. Items that are interacted with by similar users are
#    considered more similar.
#
# 3. Nearest Neighbors:
#    For a given user, the algorithm identifies the k-nearest neighbors,
#    which are other users with similar interaction patterns. The value of 'k'
#    is a parameter that determines how many neighbors to consider.
#
# 4. Recommendation:
#    The items liked or interacted with by the k-nearest neighbors are then
#    recommended to the user. The idea is that users who are similar in their
#    preferences are likely to appreciate similar items.
#
# 5. Customization:
#    The system allows customization by adjusting the value of 'k' or
#    considering different features in the user-item interaction matrix.

import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Function to load data from a CSV file into a Pandas DataFrame.
# Parameters:
#   - file_path: String, the file path of the CSV file.
# Objective:
#   This function reads the data from a CSV file specified by the 
#   file path and returns it as a Pandas DataFrame using the 
#   pd.read_csv method.
# Returns:
#   A Pandas DataFrame containing the loaded data.
def load_data(file_path):
    # Load data from CSV into a DataFrame
    return pd.read_csv(file_path)

# Function to prepare data for the Surprise library.
# Parameters:
#   - df: Pandas DataFrame containing user-item interactions.
# Objective:
#   This function takes a Pandas DataFrame representing user-item 
#   interactions, creates a Surprise Reader object with a specified 
#   rating scale, and loads the data into a Surprise Dataset using the 
#   load_from_df method.
# Returns:
#   A Surprise Dataset object prepared for collaborative filtering.
def prepare_data(df):
    # Prepare data for Surprise library
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'click_article_id', 'click_timestamp']], reader)
    return data

# Function to split data into training and testing sets.
# Parameters:
#   - data: Surprise Dataset object containing user-item interactions.
#   - test_size: Proportion of the data to include in the test split 
#     (default is 0.2).
#   - random_state: Seed used for randomization (default is 42).
# Objective:
#   This function takes a Surprise Dataset object representing user-item 
#   interactions, splits it into training and testing sets using the 
#   train_test_split function, and returns the resulting trainset and testset.
# Returns:
#   A tuple containing the trainset and testset.
def split_data(data, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
    return trainset, testset

# Function to build a collaborative filtering model.
# Parameters:
#   - sim_options: Dictionary specifying similarity options 
#     (default is {'name': 'cosine', 'user_based': True}).
# Objective:
#   This function creates a collaborative filtering model using 
#   the k-nearest neighbors (KNN) approach. The default similarity 
#   option is set to cosine similarity, and the collaborative 
#   filtering model is configured to be user-based.
# Returns:
#   A collaborative filtering model configured based on the 
#   provided similarity options.
def build_model(sim_options={'name': 'cosine', 'user_based': True}):
    # Use user-based collaborative filtering with KNN
    return KNNBasic(sim_options=sim_options)

# Function to train a collaborative filtering model.
# Parameters:
#   - model: The collaborative filtering model to be trained.
#   - trainset: Training set for the collaborative filtering model.
# Objective:
#   This function takes a collaborative filtering model and its training set,
#   then trains the model using the provided training set.
# Returns:
#   None. The model is trained in place, and no explicit output is returned.
def train_model(model, trainset):
    # Train the collaborative filtering model
    model.fit(trainset)

# Function to get top N recommendations for each user based on collaborative
# filtering predictions.
# Parameters:
#   - predictions: A list of tuples representing predictions (uid, iid, true_r,
#     est, _).
#   - n: Number of top recommendations to retrieve for each user (default is 5).
# Objective:
#   This function takes a list of predictions and organizes them by user, sorting
#   the items for each user based on estimated ratings in descending order. It
#   then returns a dictionary where keys are user IDs, and values are lists of
#   the top N recommended items along with their estimated ratings.
# Returns:
#   A dictionary where keys are user IDs, and values are lists of the top N
#   recommended items along with their estimated ratings.

def get_top_n_recommendations(predictions, n=5):
    # Function to get top N recommendations for each user
    top_n = {}

    # Iterate through each prediction in the list
    for uid, iid, true_r, est, _ in predictions:
        # If the user ID is not in the top_n dictionary, initialize it
        if uid not in top_n:
            top_n[uid] = []

        # Append the tuple (item ID, estimated rating) to the user's entry
        top_n[uid].append((iid, est))

    # Sort the predictions for each user based on estimated ratings in descending order
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

        # Keep only the top N recommendations for each user
        top_n[uid] = user_ratings[:n]

    return top_n

# Function to make predictions on a specified testset using a collaborative
# filtering model.
# Parameters:
#   - model: The collaborative filtering model used for predictions.
#   - testset: Surprise testset containing user-item interactions data.
# Objective:
#   This function takes a collaborative filtering model and a testset
#   containing user-item interactions. It generates predictions for all user-item
#   pairs in the testset using the provided model.
# Returns:
#   A list of predictions in the form (uid, iid, true_r, est, _), where:
#   - uid: User ID
#   - iid: Item ID
#   - true_r: True rating (not used)
#   - est: Estimated rating by the collaborative filtering model
#   - _: Additional information (not used)
def make_predictions(model, testset):
    # Make predictions on the testset
    return model.test(testset)

if __name__ == "__main__":
    # Main program
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"

    # Load data
    df = load_data(file_path)

    # Prepare data
    data = prepare_data(df)

    # Split data into training and testing sets
    trainset, _ = split_data(data)

    # Build and train the collaborative filtering model
    model = build_model()
    train_model(model, trainset)

    # Extract the testset from the DatasetAutoFolds object
    _, testset = split_data(data)

    # Make predictions on the testset
    test_predictions = make_predictions(model, testset)

    # Get and print top 5 recommendations for all users
    top_recommendations = get_top_n_recommendations(all_predictions, n=5)
    for user, recommendations in top_recommendations.items():
        print(f"\nTop 5 recommendations for User {user}: {[item[0] for item in recommendations]}")
