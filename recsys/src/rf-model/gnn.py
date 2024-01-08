# Recommender System Using Graph Neural Network
#
# Igor Mol <igor.mol@makes.ai>
#
# In the following recommender system, a Graph Neural Network (GNN) is employed
# together with the Recency, Frequency, and Monetary (RFM) analysis to model
# user-item interactions. The GNN architecture is designed to capture patterns
# between users and items by incorporating user and item embeddings.
# These embeddings are learned through the network's training process, enabling the
# model to understand and predict user behavior based on historical interactions.
# The RFM metrics, representing user engagement patterns, are integrated into the
# GNN framework to enhance the accuracy and personalization of recommendations. By
# combining the strengths of GNNs in capturing complex dependencies in
# graph-structured data and the insights provided by RFM metrics, this approach aims
# to provide more effective and tailored recommendations in diverse application
# domains such as e-commerce or content platforms.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# The function load_data loads data from a CSV file, processes it, and calculates
# Recency, Frequency, and Monetary (RFM) metrics for customer segmentation.
# Parameters:
#   - csv_file_path: Path to the CSV file containing the data
# Returns:
#   - data: Processed DataFrame with added RFM metrics
#   - user_mapping: Mapping of customer IDs to unique indices

def load_data(csv_file_path):
    # Read data from the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Convert 'purchase_date' column to datetime format
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    
    # Factorize 'customer_id' and 'item_id' columns to obtain indices and mappings
    user_indices, user_mapping = pd.factorize(data['customer_id'])
    item_indices, item_mapping = pd.factorize(data['item_id'])
    
    # Add user and item indices to the DataFrame
    data['user_index'] = user_indices
    data['item_index'] = item_indices
    
    # Calculate Recency, Frequency, and Monetary metrics
    recency = (data['purchase_date'].max() - data['purchase_date']).dt.days
    frequency = data.groupby('customer_id')['purchase_date'].count()
    monetary = data.groupby('customer_id')['amount'].sum()
    
    # Map RFM metrics to the DataFrame based on 'customer_id'
    data['recency'] = data['customer_id'].map(recency)
    data['frequency'] = data['customer_id'].map(frequency)
    data['monetary'] = data['customer_id'].map(monetary)
    
    # Normalize the RFM metrics
    data['recency'] = data['recency'] / data['recency'].max()
    data['frequency'] = data['frequency'] / data['frequency'].max()
    data['monetary'] = data['monetary'] / data['monetary'].max()

    # Normalize and add 'timestamps' column based on purchase dates
    timestamps = data['purchase_date'].astype(np.int64) // 10**9
    timestamps_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    data['timestamps'] = timestamps_normalized
    
    # Return processed data and user mapping
    return data, user_mapping

# The function create_gnn_model creates a Graph Neural Network (GNN) for predicting
# Recency, Frequency, and Monetary (RFM) metrics based on user and item embeddings.
# Parameters:
#   - num_users: Number of unique users in the dataset
#   - num_items: Number of unique items in the dataset
#   - embedding_size: Size of the user and item embeddings
#   - hidden_size: Size of the hidden layer in the model
# Returns:
#   - GNN model consisting of user embedding, item embedding, and two linear layers

def create_gnn_model(num_users, num_items, embedding_size, hidden_size):
    # Initialize user and item embeddings using nn.Embedding
    user_embedding = nn.Embedding(num_users, embedding_size)
    item_embedding = nn.Embedding(num_items, embedding_size)
    
    # Define the neural network layers: linear1, relu activation, linear2
    # Adjusted input size to include RFM metrics (2 * embedding_size + 4)
    linear1 = nn.Linear(2 * embedding_size + 4, hidden_size)
    relu = nn.ReLU()
    linear2 = nn.Linear(hidden_size, 3)  # Output 3 values for RFM metrics
    
    # Return the GNN model as a ModuleList
    return nn.ModuleList([user_embedding, item_embedding, linear1, relu, linear2])


# The function forward_pass performs a forward pass through the Graph Neural Network (GNN)
# model to predict Recency, Frequency, and Monetary (RFM) metrics based on input features.
# Parameters:
#   - model: GNN model containing user and item embeddings, linear layers, and activation functions
#   - user_indices: Tensor of user indices for input embeddings
#   - item_indices: Tensor of item indices for input embeddings
#   - timestamps: Tensor of normalized timestamps for input features
#   - recency: Tensor of normalized recency values for input features
#   - frequency: Tensor of normalized frequency values for input features
#   - monetary: Tensor of normalized monetary values for input features
# Returns:
#   - Output tensor containing the predicted RFM metrics for the given inputs

def forward_pass(model, user_indices, item_indices, timestamps, recency, frequency, monetary):
    # Unpack model components
    user_embedding, item_embedding, linear1, relu, linear2 = model
    
    # Obtain user and item embeddings
    user_emb = user_embedding(user_indices)
    item_emb = item_embedding(item_indices)
    
    # Concatenate input features
    input_features = torch.cat([user_emb, item_emb, timestamps.unsqueeze(1),
                                recency.unsqueeze(1), frequency.unsqueeze(1),
                                monetary.unsqueeze(1)], dim=1)
    
    # Forward pass through linear layers and activation function
    h = linear1(input_features)
    h = relu(h)
    output = linear2(h)
    
    # Return the predicted RFM metrics
    return output


# The function train_model trains a Graph Neural Network (GNN) model using the provided
# training data, optimizing for Recency, Frequency, and Monetary (RFM) metrics prediction.
# Parameters:
#   - model: GNN model to be trained
#   - data: DataFrame containing the input features and target RFM metrics
#   - criterion: Loss function for model optimization
#   - optimizer: Optimization algorithm for updating model parameters
#   - num_epochs: Number of training epochs (default is 20)
#   - test_size: Proportion of data to use for testing (default is 0.2)
#   - random_state: Seed for reproducibility (default is None)
# Returns:
#   - Trained GNN model
#   - Test data used for evaluation

def train_model(model, data, criterion, optimizer, num_epochs=20, test_size=0.2, random_state=None):
    # Split data into training and testing sets
    train_data, test_data = train_test_split_custom(data, test_size, random_state)
    
    # Training loop over epochs
    for epoch in range(num_epochs):
        # Perform forward pass to obtain predictions
        predictions = forward_pass(model, torch.LongTensor(train_data['user_index'].values),
                                    torch.LongTensor(train_data['item_index'].values),
                                    torch.FloatTensor(train_data['timestamps'].values),
                                    torch.FloatTensor(train_data['recency'].values),
                                    torch.FloatTensor(train_data['frequency'].values),
                                    torch.FloatTensor(train_data['monetary'].values))
        
        # Use RFM metrics as target variables
        targets = torch.FloatTensor(train_data[['recency', 'frequency', 'monetary']].values)
        
        # Calculate loss and perform backpropagation
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current epoch and loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Return the trained model and the test data for evaluation
    return model, test_data


# The following function customizes the train-test split of a given DataFrame,
# allowing for a specified test size and random seed for reproducibility.
# Parameters:
#   - data: DataFrame to be split into training and testing sets
#   - test_size: Proportion of data to use for testing (default is 0.2)
#   - random_state: Seed for reproducibility (default is None)
# Returns:
#   - train_data: Subset of the input data for training
#   - test_data: Subset of the input data for testing

def train_test_split_custom(data, test_size=0.2, random_state=None):
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Create a mask to randomly assign data points to training or testing sets
    mask = np.random.rand(len(data)) < 1 - test_size
    
    # Use the mask to create training and testing subsets
    train_data = data[mask]
    test_data = data[~mask]

    # Return the split datasets
    return train_data, test_data


# The following function generates predictions using the provided GNN model
# on the given test data, without performing any gradient calculations.
# Parameters:
#   - model: Trained GNN model for making predictions
#   - test_data: DataFrame containing the test data features
# Returns:
#   - test_predictions: Predictions for Recency, Frequency, and Monetary (RFM) metrics
#                       on the provided test data

def make_predictions(model, test_data):
    # Disable gradient calculations during prediction
    with torch.no_grad():
        # Perform forward pass to obtain test predictions
        test_predictions = forward_pass(model, torch.LongTensor(test_data['user_index'].values),
                                        torch.LongTensor(test_data['item_index'].values),
                                        torch.FloatTensor(test_data['timestamps'].values),
                                        torch.FloatTensor(test_data['recency'].values),
                                        torch.FloatTensor(test_data['frequency'].values),
                                        torch.FloatTensor(test_data['monetary'].values))
    
    # Return the test predictions
    return test_predictions


# The next function generates and displays top recommendations for users
# based on predicted Recency, Frequency, and Monetary (RFM) metrics.
# Parameters:
#   - test_predictions: Predicted RFM metrics for test data
#   - user_mapping: Mapping of user indices to user IDs
#   - user_indices: Tensor of user indices for predictions
#   - data: DataFrame containing the original data
#   - num_users: Number of users for which recommendations are displayed (default is 5)
#   - use_rfm: Flag to indicate whether to use RFM metrics for recommendations (default is False)
# Returns: None (Prints top recommendations for users)

def get_top_recommendations(test_predictions, user_mapping, user_indices, data, num_users=5, use_rfm=False):
    # Use 'monetary' for ranking if RFM metrics are considered
    if use_rfm:
        top_recommendations = torch.argsort(test_predictions[:, 2], descending=True)[:num_users]
        print("\nTop Recommendations for All Users using RFM:")
    else:
        top_recommendations = torch.argsort(test_predictions.squeeze(), descending=True)[:num_users]
        print("\nTop Recommendations for All Users without RFM:")
    
    # Display recommendations for each user
    for user_idx in top_recommendations:
        user_id = user_mapping[user_indices[user_idx.item()]]
        recommended_items = data.loc[data['user_index'] == user_indices[user_idx.item()]].sort_values('amount', ascending=False)['item_id'].head(5).tolist()
        print(f"User {user_id}: Recommended Items {recommended_items}")


def main():
    # Load data
    csv_file_path = "./generated_data.csv"
    data, user_mapping = load_data(csv_file_path)

    # Create and initialize the model
    num_users = len(user_mapping)
    num_items = len(data['item_id'].unique())
    embedding_size = 16
    hidden_size = 64
    model = create_gnn_model(num_users, num_items, embedding_size, hidden_size)

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model and make predictions
    model, test_data = train_model(model, data, criterion, optimizer)

    # Make predictions on the test set
    test_predictions = make_predictions(model, test_data)

    # Print top recommendations for all users
    get_top_recommendations(test_predictions, user_mapping, data['user_index'], data, use_rfm=True)

if __name__ == "__main__":
    main()
