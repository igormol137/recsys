# AutoREC: Personalized Recommender System with Autoencoder
#
# Igor Mol <igor.mol@makes.ai>
#
# In a recommender system, autoencoders are employed as a technique to model and
# capture latent patterns in user-item interactions. An autoencoder consists of
# an encoder and decoder that aim to compress and reconstruct input data,
# respectively. In the context of a recommender system, these autoencoders can
# learn meaningful representations of user preferences and item characteristics.
# RFM (Recency, Frequency, Monetary) analysis is a method commonly used in
# marketing to segment and understand customer behavior. When integrated with an
# autoencoder, RFM scores can serve as additional features during the training
# process, enriching the model with information about the recency, frequency, and
# monetary aspects of user transactions. By combining RFM analysis with an
# autoencoder, the recommender system becomes capable of capturing both the
# inherent patterns in user-item interactions and the nuanced characteristics of
# individual user behavior. This integration allows for more personalized and
# effective recommendations by considering not only historical interactions but
# also the specific context of each user's engagement.


import numpy as np
import pandas as pd

# Load and preprocess customer transaction data for RFM analysis

# Load the CSV file containing customer transaction data
file_path = "./generated_data.csv"
df = pd.read_csv(file_path)

# Convert 'purchase_date' column to datetime format for date manipulation
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Encode customer and item IDs to sequential integers for efficient computation
df['customer_id'] = df['customer_id'].astype("category").cat.codes
df['item_id'] = df['item_id'].astype("category").cat.codes

# Create a user-item matrix representing customer transactions
# Rows represent customers, columns represent items, and values represent transaction amounts
user_item_matrix = pd.pivot_table(df, values='amount', index='customer_id', columns='item_id', fill_value=0)

# The data is now prepared for further analysis, including the calculation of RFM scores for customer segmentation

def calculate_rfm_scores(data):
    """
    Calculates RFM (Recency, Frequency, Monetary) scores for customer segmentation.

    Parameters:
    - data: Pandas DataFrame containing customer transaction data with columns:
        - 'customer_id': Unique identifier for each customer.
        - 'purchase_date': Date of the purchase.
        - 'item_id': Identifier for each purchased item.
        - 'amount': Monetary value of each transaction.

    Returns:
    A Pandas DataFrame with columns 'customer_id' and 'unified_rfm_score':
    - 'customer_id': Unique identifier for each customer.
    - 'unified_rfm_score': Combined RFM score for each customer based on recency, frequency, and monetary values.
    """
    # Find the most recent purchase date in the dataset
    current_date = data['purchase_date'].max()

    # Group the data by customer and calculate RFM metrics
    rfm_data = data.groupby('customer_id').agg({
        'purchase_date': lambda x: (current_date - x.max()).days,  # Calculate recency
        'item_id': 'count',  # Calculate frequency
        'amount': 'sum'  # Calculate monetary value
    }).reset_index()

    # Normalize the individual RFM scores to values between 0 and 1
    rfm_data['recency_score'] = rfm_data['purchase_date'] / rfm_data['purchase_date'].max()
    rfm_data['frequency_score'] = rfm_data['item_id'] / rfm_data['item_id'].max()
    rfm_data['monetary_score'] = rfm_data['amount'] / rfm_data['amount'].max()

    # Combine the normalized RFM scores into a unified score for each customer
    rfm_data['unified_rfm_score'] = rfm_data['recency_score'] + rfm_data['frequency_score'] + rfm_data['monetary_score']

    # Return a DataFrame containing customer_id and the unified RFM score
    return rfm_data[['customer_id', 'unified_rfm_score']]


# Merge RFM scores with user-item matrix
rfm_data = calculate_rfm_scores(df)
user_item_matrix_with_rfm = pd.merge(user_item_matrix, rfm_data, on='customer_id')

# Custom train-test split function from scratch

def train_test_split_scratch(data, test_size=0.2, random_state=None):
    """
    Split the input data into training and testing sets.

    Parameters:
    - data: Pandas DataFrame, the input dataset to be split.
    - test_size: float, optional (default=0.2), the proportion of the dataset to include in the test split.
    - random_state: int, optional (default=None), seed for reproducibility of the random shuffle.

    Returns:
    Two Pandas DataFrames representing the training and testing sets:
    - The training set includes (1 - test_size) proportion of the input data.
    - The testing set includes test_size proportion of the input data.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Shuffle the indices of the input data
    shuffled_indices = np.random.permutation(len(data))

    # Calculate the number of samples for the test set
    test_size = int(len(data) * test_size)

    # Extract indices for the test and training sets
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Return the training and testing sets based on the calculated indices
    return data.iloc[train_indices], data.iloc[test_indices]


# Data preparation and utility functions

# Split the user-item matrix with RFM scores into training and testing sets
train_data, test_data = train_test_split_scratch(user_item_matrix_with_rfm, test_size=0.2, random_state=42)

# Convert training and testing data to NumPy arrays
train_data_np = train_data.drop(columns=['unified_rfm_score']).values
train_rfm_scores = train_data['unified_rfm_score'].values.reshape(-1, 1)
test_data_np = test_data.drop(columns=['unified_rfm_score']).values

# Define the sigmoid activation function
def sigmoid(x):
    """
    Sigmoid activation function.

    Parameters:
    - x: NumPy array, input values.

    Returns:
    NumPy array, values transformed by the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    """
    Derivative of the sigmoid activation function.

    Parameters:
    - x: NumPy array, input values.

    Returns:
    NumPy array, derivative values.
    """
    return x * (1 - x)

# Define the mean squared error loss function
def mean_squared_error(y_true, y_pred):
    """
    Mean squared error loss function.

    Parameters:
    - y_true: NumPy array, true values.
    - y_pred: NumPy array, predicted values.

    Returns:
    Float, mean squared error between true and predicted values.
    """
    return np.mean((y_true - y_pred)**2)


# Forward pass function for a simple autoencoder

def forward(x, weights_encoder, weights_decoder, bias_encoder, bias_decoder):
    """
    Perform the forward pass through a simple autoencoder.

    Parameters:
    - x: NumPy array, the input data for the autoencoder.
    - weights_encoder: NumPy array, weights for the encoder layer.
    - weights_decoder: NumPy array, weights for the decoder layer.
    - bias_encoder: NumPy array, bias for the encoder layer.
    - bias_decoder: NumPy array, bias for the decoder layer.

    Returns:
    Two NumPy arrays:
    - x_hat: Output of the decoder, representing the reconstructed input.
    - z: Output of the encoder, representing the compressed latent space representation.
    """
    # Apply the sigmoid activation function to the weighted sum of input and encoder weights, add bias
    z = sigmoid(np.dot(x, weights_encoder) + bias_encoder)

    # Apply the sigmoid activation function to the weighted sum of encoder output and decoder weights, add bias
    x_hat = sigmoid(np.dot(z, weights_decoder) + bias_decoder)

    # Return the reconstructed input (x_hat) and the compressed latent space representation (z)
    return x_hat, z

# Backward pass function for updating weights and biases in a simple autoencoder

def backward(x, x_hat, z, weights_encoder, weights_decoder, bias_encoder, bias_decoder, learning_rate, rfm_scores):
    """
    Perform the backward pass to update weights and biases in a simple autoencoder.

    Parameters:
    - x: NumPy array, the input data used during the forward pass.
    - x_hat: NumPy array, the reconstructed output from the forward pass.
    - z: NumPy array, the compressed latent space representation from the forward pass.
    - weights_encoder: NumPy array, weights for the encoder layer.
    - weights_decoder: NumPy array, weights for the decoder layer.
    - bias_encoder: NumPy array, bias for the encoder layer.
    - bias_decoder: NumPy array, bias for the decoder layer.
    - learning_rate: float, the rate at which the weights and biases are updated during training.
    - rfm_scores: NumPy array, additional RFM scores to be incorporated in the update of encoder weights.

    Returns:
    Updated weights and biases for both encoder and decoder layers:
    - weights_encoder: Updated weights for the encoder layer.
    - weights_decoder: Updated weights for the decoder layer.
    - bias_encoder: Updated bias for the encoder layer.
    - bias_decoder: Updated bias for the decoder layer.
    """
    # Calculate the error between the input and reconstructed output
    error = x - x_hat

    # Compute the delta for the decoder layer using the error and sigmoid derivative of the reconstructed output
    delta_decoder = error * sigmoid_derivative(x_hat)

    # Compute the delta for the encoder layer using the delta from the decoder and sigmoid derivative of the latent space
    delta_encoder = delta_decoder.dot(weights_decoder.T) * sigmoid_derivative(z)

    # Update weights and biases for both decoder and encoder layers
    weights_decoder += learning_rate * z.T.dot(delta_decoder)
    bias_decoder += learning_rate * np.sum(delta_decoder, axis=0, keepdims=True)
    weights_encoder += learning_rate * (x.T.dot(delta_encoder) + rfm_scores.T.dot(delta_encoder))  # Include RFM scores
    bias_encoder += learning_rate * (np.sum(delta_encoder, axis=0, keepdims=True) + np.sum(delta_encoder * rfm_scores, axis=0, keepdims=True))

    # Return the updated weights and biases
    return weights_encoder, weights_decoder, bias_encoder, bias_decoder


# Training loop for a simple autoencoder with RFM scores

def train_autoencoder(data, rfm_scores, input_dim, latent_dim, learning_rate=0.001, epochs=20, batch_size=64):
    """
    Train a simple autoencoder on the provided data with an optional inclusion of RFM scores.

    Parameters:
    - data: Pandas DataFrame, the input dataset for training.
    - rfm_scores: Pandas DataFrame, RFM scores for each customer.
    - input_dim: int, the dimensionality of the input data.
    - latent_dim: int, the dimensionality of the compressed latent space representation.
    - learning_rate: float, optional (default=0.001), the rate at which the weights and biases are updated during training.
    - epochs: int, optional (default=20), the number of training epochs.
    - batch_size: int, optional (default=64), the size of each training batch.

    Returns:
    Updated weights and biases for both encoder and decoder layers after training:
    - weights_encoder: Updated weights for the encoder layer.
    - weights_decoder: Updated weights for the decoder layer.
    - bias_encoder: Updated bias for the encoder layer.
    - bias_decoder: Updated bias for the decoder layer.
    """
    # Initialize weights and biases with random values
    weights_encoder = np.random.randn(input_dim, latent_dim)
    weights_decoder = np.random.randn(latent_dim, input_dim)
    bias_encoder = np.zeros((1, latent_dim))
    bias_decoder = np.zeros((1, input_dim))

    # Training loop
    for epoch in range(epochs):
        # Iterate through the data in batches
        for i in range(0, len(data), batch_size):
            # Extract batch data and corresponding RFM scores
            batch_data = data[i:i+batch_size].drop(columns=['unified_rfm_score']).values
            batch_rfm_scores = data[i:i+batch_size]['unified_rfm_score'].values.reshape(-1, 1)

            # Perform forward pass and backward pass for weight and bias updates
            x_hat, z = forward(batch_data, weights_encoder, weights_decoder, bias_encoder, bias_decoder)
            weights_encoder, weights_decoder, bias_encoder, bias_decoder = \
                backward(batch_data, x_hat, z, weights_encoder, weights_decoder, bias_encoder, bias_decoder, learning_rate, batch_rfm_scores)

        # Print loss at the end of each epoch
        loss = mean_squared_error(data.drop(columns=['unified_rfm_score']).values, x_hat)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    # Return the updated weights and biases after training
    return weights_encoder, weights_decoder, bias_encoder, bias_decoder


# Train the autoencoder with RFM scores
latent_dim = 64
learning_rate = 0.001
epochs = 20
batch_size = 64

weights_encoder, weights_decoder, bias_encoder, bias_decoder = \
    train_autoencoder(train_data, train_rfm_scores, input_dim=user_item_matrix_with_rfm.shape[1] - 1, latent_dim=latent_dim,
                      learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)

# Generate top recommendations for a specific user using an autoencoder model

def get_top_recommendations(user_id, user_item_matrix, weights_encoder, weights_decoder, bias_encoder, bias_decoder):
    """
    Generate top recommendations for a specific user using an autoencoder model.

    Parameters:
    - user_id: int, the identifier of the target user.
    - user_item_matrix: Pandas DataFrame, a user-item matrix representing historical interactions.
    - weights_encoder: NumPy array, weights for the encoder layer of the autoencoder.
    - weights_decoder: NumPy array, weights for the decoder layer of the autoencoder.
    - bias_encoder: NumPy array, bias for the encoder layer of the autoencoder.
    - bias_decoder: NumPy array, bias for the decoder layer of the autoencoder.

    Returns:
    A Pandas Series containing top recommendations for the specified user.
    Recommendations are sorted by predicted ratings in descending order.
    """
    # Extract the historical interaction data for the target user
    user_history = user_item_matrix.loc[user_id].drop('unified_rfm_score').values.reshape(1, -1)

    # Perform a forward pass to reconstruct the user's interaction data
    x_hat, _ = forward(user_history, weights_encoder, weights_decoder, bias_encoder, bias_decoder)

    # Create a DataFrame with predicted ratings for items
    predicted_ratings = pd.DataFrame(x_hat, columns=user_item_matrix.columns.drop('unified_rfm_score'))

    # Sort items by predicted ratings in descending order to get top recommendations
    recommendations = predicted_ratings.iloc[0].sort_values(ascending=False)

    # Return the top recommendations for the specified user
    return recommendations

# Print top recommendations for all users using the trained autoencoder model

# Iterate through all users in the user-item matrix
for user_id in range(user_item_matrix.shape[0]):
    # Get top recommendations for the current user using the trained autoencoder
    recommendations = get_top_recommendations(user_id, user_item_matrix_with_rfm,
                                              weights_encoder, weights_decoder, bias_encoder, bias_decoder)

    # Select a specified number of top recommendations (adjust as needed)
    top_recommendations = recommendations.head(5)

    # Print the top recommendations for the current user
    print(f"\nTop Recommendations for User {user_id}:\n{top_recommendations}")

