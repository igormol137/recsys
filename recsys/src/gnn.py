# Recommender System: The Graph Neural Network Approach
#
# Igor Mol <igor.mol@makes.ai>
#
# The Graph Neural Network represents the interactions between users and items 
# as nodes in a graph. The edges in the graph holds information about the inte-
# ractions, e.g., the user-item engagement, such that the GNN algorithm learns 
# to capture the patterns within this interconnected structure.
# 	In the following GNN-based recommender system, each user and item is as-
# sociated with an initial embedding vector, and the GNN iteratively refines 
# these embeddings by considering the information from neighboring nodes in the
# graph. Through multiple layers of graph convolutional operations, the GNN 
# learns to aggregate and propagate information across the graph

import pandas as pd
import numpy as np

# preprocess_data(df):
# The primary goal of this function is to guarantee that the indices of the 
# 'user_id' and 'click_article_id' columns initiate from 1, instead of the de-
# fault 0. To achieve this, the function utilizes Pandas' categorical data type
# by applying the astype('category') method to the 'user_id' and 'click_article_
# id' columns. 
# 	The function employs the .cat.codes attribute on both columns, which as-
# signs unique numerical codes to each category. By adding 1 to these codes, the
# function effectively shifts the starting indices to 1. This adjustment aligns 
# with common indexing conventions, particularly in scenarios where indices con-
# ventionally begin at 1. The final modified DataFrame is then returned

def preprocess_data(df):
    # Ensure user and article indices start from 1
    df['user_id'] = df['user_id'].astype('category').cat.codes + 1
    df['click_article_id'] = df['click_article_id'].astype('category').cat.codes + 1
    return df

# create_user_item_matrix():
# This function was designed to generate a user-item interaction matrix from a 
# given DataFrame (df). The function begins by calculating the number of unique 
# users and items in the dataset using the nunique function from pandas. It then 
# initializes a matrix, named user_item_matrix, with dimensions corresponding to 
# the number of users and items identified.
# 	Subsequently, the function iterates through each row of the DataFrame u-
# sing a for loop. For each row, it extracts the user and item IDs and sets the
# corresponding entry in the user-item matrix to 1. The indices are adjusted by
# subtracting 1 to ensure compatibility with zero-based indexing.
# 	Once the matrix is filled based on user-item interactions, the function
# returns three components: the resulting user_item_matrix, the total number of
# unique users (num_users), and the total number of unique items (num_items). 
# This user-item matrix can be used for collaborative filtering or recommendation
# systems, where the presence of a '1' in the matrix indicates that a user has
# interacted with a specific item. 

def create_user_item_matrix(df):
    num_users = df['user_id'].nunique()
    num_items = df['click_article_id'].nunique()

    user_item_matrix = np.zeros((num_users, num_items))

    # Fill the matrix based on user-item interactions
    for _, row in df.iterrows():
        user_item_matrix[row['user_id'] - 1, row['click_article_id'] - 1] = 1

    return user_item_matrix, num_users, num_items

# train_test_split_custom():
# This function is reponsible for splitting a given dataset into training and 
# testing sets. The function takes three parameters: data representing the data-
# set to be split, test_size indicating the proportion of the data to allocate 
# for testing (default is 20%), and random_state to ensure reproducibility by 
# setting the random seed if a value is provided.
# 	The function first checks if a random seed (random_state) is provided, 
# and if so, it sets the seed using NumPy to ensure that the random shuffling of
# data remains consistent across different runs. It then generates an array of 
# indices corresponding to the length of the dataset. These indices are shuffled
# using NumPy's shuffle function to randomize the order of the data.
# 	Next, the function calculates the number of samples to be included in 
# the testing set based on the specified test_size proportion. It then selects 
# the corresponding indices for the testing and training sets using array sli-
# cing. Finally, it creates new dataframes (train_data and test_data) by inde-
# xing into the original dataset using the selected indices. The resulting trai-
# ning and testing sets are returned by the function

def train_test_split_custom(data, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    test_size = int(test_size * len(data))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    
    return train_data, test_data

# initialize_embeddings():
# This function takes three parameters: num_users (number of users), num_items 
# (number of items), and embedding_size (dimensionality of the embeddings). It 
# returns two matrices, user_embeddings and item_embeddings, where each row cor-
# responds to a user or item, and each column represents a feature in the embed-
# ding space.
#	The initialization is performed using np.random.rand, which generates 
# random values between 0 and 1. The size of the matrices is determined by the 
# number of users/items and the specified embedding size. These initialized em-
# beddings serve as the starting point for the recommendation model, capturing 
# latent features that contribute to predicting user-item interactions.


def initialize_embeddings(num_users, num_items, embedding_size):
    # Initialize user and item embeddings
    user_embeddings = np.random.rand(num_users, embedding_size)
    item_embeddings = np.random.rand(num_items, embedding_size)
    
    return user_embeddings, item_embeddings

def update_embeddings(user_embeddings, item_embeddings, user_gradient, item_gradient, learning_rate):
    # Update embeddings using gradient descent
    user_embeddings -= learning_rate * user_gradient
    item_embeddings -= learning_rate * item_gradient

# train_model():
# This function implements training process of a collaborative filtering recom-
# mendation model using matrix factorization. The goal of the model is to predi-
# ct user-item interaction scores based on user and item embeddings. The train-
# ing process involves iterating through epochs, performing a forward pass to 
# calculate predictions, computing the mean squared error loss, and updating the
# embeddings using gradient descent.
# 	In the forward pass, the model predicts user-item interaction scores by
# taking the dot product of user embeddings and item embeddings transposed. The
# loss is then calculated as the mean squared error between the predicted scores
# and the actual user-item interaction matrix for the training data.
# 	For the backward pass (gradient descent), gradients are computed with 
# respect to both user and item embeddings. These gradients are used to update 
# the embeddings, nudging them in the direction that minimizes the mean squared 
# error loss. The learning rate parameter determines the step size of these
# updates.

def train_model(user_item_matrix_train, user_embeddings, item_embeddings, num_users, num_items, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward pass
        user_item_scores = np.dot(user_embeddings, item_embeddings.T)
        
        # Calculate loss (mean squared error)
        loss = np.sum((user_item_scores - user_item_matrix_train) ** 2) / (num_users * num_items)
        
        # Backward pass (gradient descent)
        user_gradient = 2 * np.dot((user_item_scores - user_item_matrix_train), item_embeddings) / (num_users * num_items)
        item_gradient = 2 * np.dot((user_item_scores - user_item_matrix_train).T, user_embeddings) / (num_users * num_items)
        
        # Update embeddings
        update_embeddings(user_embeddings, item_embeddings, user_gradient, item_gradient, learning_rate)

        # Print loss for each epoch
        print(f"Epoch {epoch + 1}, Loss: {loss}")

# get_top_recommendations():
# This function takes as input the user's identifier (user_id), user embeddings,
# item embeddings, and training data. It calculates the interaction scores between
# the user and all items by performing a dot product between the user's embedding 
# and the transposed item embeddings. The resulting scores are then sorted, and 
# the indices of the top N items are obtained.
# 	To ensure that the recommendations are personalized and relevant, the 
# function excludes articles that the user has already interacted with. This is
# achieved by checking the user's past interactions in the training data and re-
# moving those articles from the top recommendations. The final list of top re-
# commendations is returned.

def get_top_recommendations(user_id, user_embeddings, item_embeddings, train_data, N=5):
    user_scores = np.dot(user_embeddings[user_id - 1], item_embeddings.T)
    top_indices = np.argsort(user_scores)[::-1][:N]
    
    # Exclude articles the user has already clicked
    clicked_articles = set(train_data[train_data['user_id'] == user_id]['click_article_id'])
    top_recommendations = [article + 1 for article in top_indices if article + 1 not in clicked_articles][:N]
    
    return top_recommendations

def main():
    # Example usage within the main function

    # Load your data (assuming df is your DataFrame)
    df = pd.read_csv("/Volumes/KINGSTON/archive2/clicks_sample.csv")

    # Preprocess data
    df = preprocess_data(df)

    # Create user-item interaction matrix
    user_item_matrix, num_users, num_items = create_user_item_matrix(df)

    # Custom train-test split
    train_data, test_data = train_test_split_custom(df, test_size=0.2, random_state=42)

    # Initialize embeddings
    embedding_size = 64
    user_embeddings, item_embeddings = initialize_embeddings(num_users, num_items, embedding_size)

    # Train the model
    learning_rate = 0.01
    epochs = 20
    train_model(user_item_matrix, user_embeddings, item_embeddings, num_users, num_items, learning_rate, epochs)

    # Get top recommendations for all users
    all_users = np.unique(train_data['user_id'])
    for user_id in all_users:
        recommendations = get_top_recommendations(user_id, user_embeddings, item_embeddings, train_data)
        print(f"User {user_id}: {recommendations}")

# Execute the main function
if __name__ == "__main__":
    main()
