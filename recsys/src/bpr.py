import pandas as pd
import numpy as np

# initialize_bpr():
# This function is responsible for setting up the initial configuration of the 
# Bayesian Personalized Ranking (BPR) model.
# 	Inside the function, the user and item latent vectors are initialized 
# using NumPy's random.normal function, generating matrices with random values 
# from a normal distribution. The dimensions of these matrices are determined by
# the number of users, number of items, and the specified latent dimensionality.
# 	The function then returns a tuple containing the initialized user latent
# matrix, item latent matrix, latent dimension, learning rate, regularization 
# parameter, and the number of epochs.

def initialize_bpr(num_users, num_items, latent_dim=10, learning_rate=0.01, reg_param=0.01, num_epochs=10):
    """
    Initialize the BPR model with specified parameters.
    """
    user_latent = np.random.normal(size=(num_users, latent_dim))
    item_latent = np.random.normal(size=(num_items, latent_dim))
    return user_latent, item_latent, latent_dim, learning_rate, reg_param, num_epochs

# fit_bpr():
# This function implements the Bayesian Personalized Ranking (BPR) model for 
# collaborative filtering. The purpose of this function is to train the model u-
# sing stochastic gradient descent on user-item interactions.
# 	The training process involves looping through each epoch and, within ea-
# ch epoch, iterating over triplets of (user, positive item, negative item) ge-
# nerated by the generate_triplets function. For each triplet, the model compu-
# tes the scores for positive and negative items, computes the loss using a lo-
# gistic loss function, and updates the user and item latent vectors using sto-
# chastic gradient descent.
# 	The gradients are computed based on the difference between the scores of
# positive and negative items, and regularization terms are added to prevent 
# overfitting. The update rule for each parameter involves subtracting the pro-
# duct of the learning rate and the corresponding gradient.
# 	After completing each epoch, the function prints a progress message in-
# dicating the current epoch out of the total number of epochs. Finally, the 
# function returns the updated user and item latent vectors.

def fit_bpr(user_latent, item_latent, latent_dim, learning_rate, reg_param, num_epochs, interactions, num_users, num_items):
    """
    Fit the BPR model to the user-item interactions.
    """
    for epoch in range(num_epochs):
        for user, pos_item, neg_item in generate_triplets(interactions, num_users, num_items):
            pos_score = np.dot(user_latent[user], item_latent[pos_item])
            neg_score = np.dot(user_latent[user], item_latent[neg_item])
            loss = -np.log(sigmoid(pos_score - neg_score))

            # Update user and item latent vectors using stochastic gradient descent
            user_gradient = (1 / (1 + np.exp(pos_score - neg_score))) * (item_latent[pos_item] - item_latent[neg_item]) - reg_param * user_latent[user]
            pos_item_gradient = (1 / (1 + np.exp(pos_score - neg_score))) * user_latent[user] - reg_param * item_latent[pos_item]
            neg_item_gradient = -(1 / (1 + np.exp(pos_score - neg_score))) * user_latent[user] - reg_param * item_latent[neg_item]

            user_latent[user] -= learning_rate * user_gradient
            item_latent[pos_item] -= learning_rate * pos_item_gradient
            item_latent[neg_item] -= learning_rate * neg_item_gradient

        print(f"Epoch {epoch + 1}/{num_epochs} completed.")

    return user_latent, item_latent

# generate_triplets():
# The purpose of this function is to create triplets, denoted as (user, pos_item
# , neg_item), intended for training the Bayesian Personalized Ranking (BPR) mo-
# del in collaborative filtering. The function begins by initializing an empty 
# list named triplets to store the generated triplets. It then iterates through
# each user, extracting positive items by finding non-zero entries in the inte-
#ractions matrix. Additionally, it identifies negative items by subtracting the 
# set of positive items from the set of all items.
# 	For each positive item associated with a user, the function randomly se-
# lects a negative item from the set of available negative items. This random-
# ness is introduced using NumPy's random.choice function. The resulting triplet
# composed of the user, positive item, and negative item, is appended to the 
# triplets list. To avoid duplication, the selected negative item is removed 
# from the set of available negative items. Once this process is completed for 
# all users, the function returns the list of generated triplets.

def generate_triplets(interactions, num_users, num_items):
    """
    Generate triplets (user, pos_item, neg_item) for training the BPR model.
    """
    triplets = []
    for user in range(num_users):
        pos_items_for_user = set(interactions[user].nonzero()[0])
        neg_items_for_user = set(range(num_items)) - pos_items_for_user

        for pos_item in pos_items_for_user:
            if neg_items_for_user:
                neg_item = np.random.choice(list(neg_items_for_user))
                triplets.append((user, pos_item, neg_item))
                neg_items_for_user.remove(neg_item)

    return triplets

# sigmoid():
# The purpose of this function is to take a numerical input, denoted as 'x', and 
# compute the corresponding output using the sigmoid activation function.
# The sigmoid function is a mathematical expression represented as:
#	1 / (1 + e^(-x)),
# where 'e' is the mathematical constant approximately equal to 2.71828. So, the 
# sigmoid function maps any real-valued input to a range between 0 and 1. This 
# property makes it particularly useful in machine learning and neural networks,
# where it is often employed to introduce non-linearity and normalize values 
# between certain limits.

def sigmoid(x):
    """
    Sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

# recommend():
# This function aims to offer the top N recommendations for a specified user 
# within a collaborative filtering system. This function takes four parameters:
#
# - user_latent: A matrix representing the latent vectors associated with users.
# - item_latent: A matrix representing the latent vectors associated with items.
# - user: The particular user for whom recommendations are desired.
# - top_n: An optional parameter specifying the number of recommendations to 
# provide, with a default value of 5.
#
# 	The primary purpose of this function is to calculate recommendation sco-
# res for items based on the latent vectors of users and items. This is achieved
# by computing the dot product of the latent vector of the specified user and 
# the transpose of the matrix containing latent vectors for all items (item_
# latent.T). The resulting scores represent the model's estimate of the user's 
# preference for each item.
#	Subsequently, these scores are utilized to identify the top N recommen-
# ded items. The scores are sorted in descending order using the NumPy function
# argsort, and the first N items are selected using slicing ([::-1][:top_n]). 
# The function concludes by returning a list containing the indices of the reco-
# mmended items.

def recommend(user_latent, item_latent, user, top_n=5):
    """
    Get top N recommendations for a given user.
    """
    scores = np.dot(user_latent[user], item_latent.T)
    recommended_items = np.argsort(scores)[::-1][:top_n]
    return recommended_items
    
# load_data():
# The aim of this function is to load data from a CSV file into a DataFrame. 
# This function takes a single parameter:
#
# - file_path: A string specifying the path to the CSV file containing the data.
#
# 	The primary purpose of the function is to leverage the pandas library, 
# denoted by pd.read_csv, to read the contents of the specified CSV file and 
# convert it into a DataFrame. A DataFrame is a tabular data structure commonly
# used in data analysis and manipulation.
#	The load_data function employs the pandas method read_csv to read the 
# CSV file and create a DataFrame, encapsulating the data in a format that al-
# lows for easy analysis and manipulation. The function concludes by returning
# this DataFrame

def load_data(file_path):
    """
    Load data from a CSV file into a DataFrame.
    """
    return pd.read_csv(file_path)

# create_user_item_matrix(df):
# The purpose of this function is to generate a user-item interaction matrix 
# from a DataFrame `df'. This function takes a single parameter:
#
# - df: A DataFrame containing relevant data, modelling user-item interactions.
#
# 	The primary objective of this function is to create a matrix that repre-
# sents the interactions between users and items, used here in collaborative 
# filtering for our recommendation systems. The pivot_table function is employed
# with specific parameters:
#
# 1. values='session_size': Specifies that the values to be aggregated in the 
# matrix are taken from the 'session_size' column of the DataFrame.
# 2. index='user_id': The 'user_id' column is designated as the index of the
# resulting matrix, indicating user-related information.
# 2. columns='click_article_id': The 'click_article_id' column is selected as the
# columns of the matrix, representing items.
# 4. fill_value=0: Any missing values in the matrix are filled with zeros.
#
# 	The resulting matrix encapsulates user-item interactions, where each 
# cell denotes the interaction or session size between a specific user and item. 

def create_user_item_matrix(df):
    """
    Create a user-item interaction matrix from the DataFrame.
    """
    return pd.pivot_table(df, values='session_size', index='user_id', columns='click_article_id', fill_value=0)

def main():
    # Load data
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
    df = load_data(file_path)

    # Create user-item interaction matrix
    user_item_matrix = create_user_item_matrix(df)

    # Instantiate BPR model
    num_users, num_items = user_item_matrix.shape
    user_latent, item_latent, latent_dim, learning_rate, reg_param, num_epochs = initialize_bpr(num_users, num_items, latent_dim=10, learning_rate=0.01, reg_param=0.01, num_epochs=10)

    # Fit the BPR model
    user_latent, item_latent = fit_bpr(user_latent, item_latent, latent_dim, learning_rate, reg_param, num_epochs, user_item_matrix.values, num_users, num_items)

    # Get top 5 recommendations for all users
    top_n_recommendations = []
    for user_id in user_item_matrix.index:
        recommendations = recommend(user_latent, item_latent, user_id, top_n=5)
        top_n_recommendations.append({'user_id': user_id, 'recommendations': recommendations})

    # Print the top 5 recommendations for all users
    for user_rec in top_n_recommendations:
        print(f"Top 5 recommendations for user {user_rec['user_id']}: {user_rec['recommendations']}")

if __name__ == "__main__":
    main()
