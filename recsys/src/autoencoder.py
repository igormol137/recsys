"""
Autoencoder Recommender System

Igor Mol <igor.mol@makes.ai>

The use o Autoencoder in recommender system can be summarise as follows:

1. Collaborative Filtering:
   - Autoencoders can be employed in collaborative filtering tasks where the goal is
     to make recommendations based on user-item interactions.

2. User-Item Interaction Matrix:
   - The user-item interaction matrix is encoded into a lower-dimensional latent space
     by the autoencoder, capturing patterns and relationships in user preferences.

3. Learning User and Item Representations:
   - The encoder network learns embeddings for both users and items, allowing the
     model to capture nuanced features and interactions between them.

4. Implicit and Explicit Feedback:
   - Autoencoders can handle both implicit (e.g., clicks, views) and explicit
     feedback (e.g., ratings), making them versatile for various recommendation
     scenarios.

5. Generating Recommendations:
   - Recommendations are generated by decoding the learned latent representations.
     Similar users or items in the latent space are likely to receive similar
     recommendations.

For completeness, let us briefly recall the mathematical definition of a 
Variational Autoencoder (VAE).

Let X be the input data, and Z be the latent variable.

1. Encoder (Recognition Model):
   - Output: Parameters of the approximate posterior distribution q(z | x),
     typically mean (μ) and variance (σ^2).
   - Forward Pass:
        μ, σ^2 = encoder(X)

2. Reparameterization Trick:
   - Draw a sample from the standard normal distribution ε ~ N(0, 1).
   - Sample a latent variable z from the approximate posterior:
        z = μ + ε * σ

3. Decoder (Generative Model):
   - Output: Parameters of the conditional distribution p(x | z).
   - Forward Pass:
        x̂ = decoder(z)

4. Objective Function (Variational Lower Bound):
   - Minimize the negative Evidence Lower Bound (ELBO):
        ELBO = E[log p(x | z) + log p(z) - log q(z | x)]
        where the expectation is approximated using Monte Carlo sampling.

5. Loss Function:
   - Negative ELBO as the loss to minimize:
        Loss = -ELBO
"""


import pandas as pd
import numpy as np

# Load data from a CSV file and preprocess it.
def load_data(file_path):
    """
    Load data from a CSV file and preprocess it.

    Parameters:
    - file_path (str): The path to the CSV file containing user-item interaction data.

    Returns:
    - tuple: A tuple containing three elements:
        1. pd.DataFrame: Preprocessed DataFrame with additional columns for user and item indices.
        2. int: Number of unique users in the dataset.
        3. int: Number of unique items (articles) in the dataset.
    """
    df = pd.read_csv(file_path)

    # Create user and item matrices
    user_ids = df['user_id'].unique()
    item_ids = df['click_article_id'].unique()

    num_users = len(user_ids)
    num_items = len(item_ids)

    user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_mapping = {item_id: idx for idx, item_id in enumerate(item_ids)}

    # Map user and item IDs to indices
    df['user_index'] = df['user_id'].map(user_mapping)
    df['item_index'] = df['click_article_id'].map(item_mapping)

    return df, num_users, num_items

# Custom train-test split function.
def custom_train_test_split(data, test_size=0.2, random_state=None):
    """
    Custom train-test split function.

    Parameters:
    - data (pd.DataFrame): DataFrame containing preprocessed user-item interaction data.
    - test_size (float): Proportion of users to include in the test split.
    - random_state (int or None): Seed for random number generation.

    Returns:
    - tuple: A tuple containing two DataFrames:
        1. train_data (pd.DataFrame): Training data with a subset of users.
        2. test_data (pd.DataFrame): Test data with the remaining users.
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_users = data['user_index'].unique()
    test_users = np.random.choice(unique_users, size=int(test_size * len(unique_users)), replace=False)

    test_mask = data['user_index'].isin(test_users)
    train_data = data[~test_mask]
    test_data = data[test_mask]

    return train_data, test_data

# Initialize weights and biases.
def initialize_parameters(num_users, num_items, latent_dim):
    """
    Initialize weights and biases for matrix factorization.

    Parameters:
    - num_users (int): Number of users in the dataset.
    - num_items (int): Number of items (articles) in the dataset.
    - latent_dim (int): Dimension of the latent space for embeddings.

    Returns:
    - tuple: A tuple containing initial embeddings and biases:
        1. user_embeddings (np.ndarray): Initial user embeddings matrix.
        2. item_embeddings (np.ndarray): Initial item embeddings matrix.
        3. user_biases (np.ndarray): Initial user biases vector.
        4. item_biases (np.ndarray): Initial item biases vector.
    """
    user_embeddings = np.random.randn(num_users, latent_dim)
    item_embeddings = np.random.randn(num_items, latent_dim)

    user_biases = np.zeros(num_users)
    item_biases = np.zeros(num_items)

    return user_embeddings, item_embeddings, user_biases, item_biases

# Clip gradients to prevent overflow.
def clip_gradients(gradients, max_gradient):
    """
    Clip gradients to prevent overflow.

    Parameters:
    - gradients (np.ndarray): Gradients to be clipped.
    - max_gradient (float): Maximum allowed gradient value.

    Returns:
    - np.ndarray: Clipped gradients.
    """
    return np.clip(gradients, -max_gradient, max_gradient)

# Compute predictions for a batch of user-item pairs.
def compute_predictions(user_embeddings, item_embeddings, user_biases, item_biases, user_indices, item_indices):
    """
    Compute predictions for a batch of user-item pairs.

    Parameters:
    - user_embeddings (np.ndarray): User embeddings matrix.
    - item_embeddings (np.ndarray): Item embeddings matrix.
    - user_biases (np.ndarray): User biases array.
    - item_biases (np.ndarray): Item biases array.
    - user_indices (np.ndarray): Indices of users in the batch.
    - item_indices (np.ndarray): Indices of items in the batch.

    Returns:
    - np.ndarray: Predicted ratings for the user-item pairs.
    """
    user_latent = user_embeddings[user_indices, :]
    item_latent = item_embeddings[item_indices, :]
    predictions = np.sum(user_latent * item_latent, axis=1) + user_biases[user_indices] + item_biases[item_indices]

    return predictions

# Train the collaborative filtering model.
def train_model(train_data, user_embeddings, item_embeddings, user_biases, item_biases,
                learning_rate, epochs, batch_size, latent_dim, max_gradient):
    """
    Train the collaborative filtering model using stochastic gradient descent.

    Parameters:
    - train_data (pd.DataFrame): Training data containing user, item, and ratings information.
    - user_embeddings (np.ndarray): Initial user embeddings matrix.
    - item_embeddings (np.ndarray): Initial item embeddings matrix.
    - user_biases (np.ndarray): Initial user biases array.
    - item_biases (np.ndarray): Initial item biases array.
    - learning_rate (float): Learning rate for gradient descent.
    - epochs (int): Number of training epochs.
    - batch_size (int): Size of each training batch.
    - latent_dim (int): Dimensionality of the latent space.
    - max_gradient (float): Maximum allowed gradient value for clipping.

    Philosophy:
    The model is trained using stochastic gradient descent. For each epoch, the training data is
    shuffled and divided into batches. Predictions are computed, and mean squared error loss is
    calculated. Weights and biases are then updated using gradient descent with clipping to prevent
    overflow. The loss is printed for each epoch.

    Returns:
    None
    """
    for epoch in range(epochs):
        # Shuffle training data
        train_data = train_data.sample(frac=1).reset_index(drop=True)

        for start in range(0, len(train_data), batch_size):
            end = start + batch_size
            batch_data = train_data.iloc[start:end]

            user_indices = batch_data['user_index'].values
            item_indices = batch_data['item_index'].values
            ratings = batch_data['session_size'].values

            # Compute predictions
            predictions = compute_predictions(user_embeddings, item_embeddings, user_biases, item_biases, user_indices, item_indices)

            # Compute loss
            errors = predictions - ratings
            mse_loss = np.mean(errors**2)

            # Update weights and biases using gradient descent with clipping
            user_gradients = -2 * errors[:, np.newaxis] * item_embeddings[item_indices, :]
            item_gradients = -2 * errors[:, np.newaxis] * user_embeddings[user_indices, :]

            # Clip gradients to prevent overflow
            user_gradients = clip_gradients(user_gradients, max_gradient)
            item_gradients = clip_gradients(item_gradients, max_gradient)

            user_embeddings[user_indices, :] -= learning_rate * user_gradients
            item_embeddings[item_indices, :] -= learning_rate * item_gradients

            user_biases[user_indices] -= learning_rate * errors
            item_biases[item_indices] -= learning_rate * errors

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {mse_loss}")

# Print top recommendations for all users.
def print_top_recommendations(user_embeddings, item_embeddings, user_biases, item_biases,
                              user_ids, item_ids, top_n=5):
    """
    Print top recommendations for all users.

    Parameters:
    - user_embeddings (np.ndarray): User embeddings matrix.
    - item_embeddings (np.ndarray): Item embeddings matrix.
    - user_biases (np.ndarray): User biases array.
    - item_biases (np.ndarray): Item biases array.
    - user_ids (np.ndarray): Array of user IDs.
    - item_ids (np.ndarray): Array of item IDs.
    - top_n (int): Number of top recommendations to print.

    Strategy:
    For each user, compute predictions for all items, excluding those already interacted with.
    Select the top-N items with the highest predictions and print the recommendations.

    Returns:
    None
    """
    all_user_indices = np.arange(len(user_embeddings))
    all_item_indices = np.arange(len(item_embeddings))

    recommendations = []

    for user_index in all_user_indices:
        user_latent = user_embeddings[user_index, :]
        item_latent = item_embeddings[all_item_indices, :]
        predictions = np.sum(user_latent * item_latent, axis=1) + user_biases[user_index] + item_biases[all_item_indices]

        # Exclude items that the user has already interacted with
        user_interactions = df[df['user_index'] == user_index]['item_index'].values
        predictions[user_interactions] = -np.inf

        top_items = np.argsort(predictions)[::-1][:top_n]
        recommendations.append((user_index, top_items))

    # Print recommendations
    for user_index, top_items in recommendations:
        user_id = user_ids[user_index]
        recommended_item_ids = [item_ids[item_index] for item_index in top_items]
        print(f"User {user_id}: Top Recommendations - {recommended_item_ids}")

# The main program loads data from a CSV file, performs a train-test split,
# sets hyperparameters, initializes weights and biases, trains the collaborative
# filtering model using stochastic gradient descent, and prints the top 5
# recommendations for all users.

if __name__ == "__main__":
    # Load data
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
    df, num_users, num_items = load_data(file_path)

    # Train-test split
    train_data, test_data = custom_train_test_split(df, test_size=0.2, random_state=42)

    # Hyperparameters
    latent_dim = 10
    learning_rate = 0.001
    epochs = 20
    batch_size = 64
    max_gradient = 5.0

    # Initialize weights and biases
    user_embeddings, item_embeddings, user_biases, item_biases = initialize_parameters(num_users, num_items, latent_dim)

    # Train the collaborative filtering model
    train_model(train_data, user_embeddings, item_embeddings, user_biases, item_biases,
                learning_rate, epochs, batch_size, latent_dim, max_gradient)

    # Print top 5 recommendations for all users
    user_ids = df['user_id'].unique()
    item_ids = df['click_article_id'].unique()
    print_top_recommendations(user_embeddings, item_embeddings, user_biases, item_biases, user_ids, item_ids, top_n=5)
