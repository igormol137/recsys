# attention.py
#
# Igor Mol <igor.mol@makes.ai>
#
# The Attention mechanism in our recommendation system focuses on capturing 
# user-item interactions by assigning weights to different parts of the 
# input-sequence during the model's training, so that the model learns how to 
# rank items per user, providing personalized recommendations.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define a function to load data from a CSV file into a Pandas DataFrame.
# Parameters:
#   - file_path: The file path of the CSV file containing the data.

def load_data(file_path):
    # Use Pandas read_csv function to read data from the specified file path
    # and load it into a DataFrame.
    return pd.read_csv(file_path)

# Define a function to preprocess data for a recommendation system.
# Parameters:
#   - df: DataFrame representing user-item interactions.
#   - user_col: Column name for user IDs (default is 'user_id').
#   - article_col: Column name for article IDs (default is 'click_article_id').

def preprocess_data(df, user_col='user_id', article_col='click_article_id'):
    # Create LabelEncoder instances for users and articles
    user_encoder = LabelEncoder()
    article_encoder = LabelEncoder()

    # Encode user and article IDs in the DataFrame using LabelEncoders
    df[user_col] = user_encoder.fit_transform(df[user_col])
    df[article_col] = article_encoder.fit_transform(df[article_col])

    # Split the preprocessed DataFrame into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Return the training and testing DataFrames, along with the encoders
    return train_df, test_df, user_encoder, article_encoder


# Define a function to build a collaborative filtering neural network model.
# Parameters:
#   - num_users: Total number of users in the system.
#   - num_articles: Total number of articles/items.
#   - embedding_size: Dimensionality of embedding vectors (default is 50).

def build_model(num_users, num_articles, embedding_size=50):
    # Input layer for user indices
    input_user = Input(shape=(1,))
    # Embedding layer for user indices
    embedding_user = Embedding(input_dim=num_users, output_dim=embedding_size)(input_user)
    # Flatten the user embedding tensor
    flat_user = Flatten()(embedding_user)

    # Input layer for article indices
    input_article = Input(shape=(1,))
    # Embedding layer for article indices
    embedding_article = Embedding(input_dim=num_articles, output_dim=embedding_size)(input_article)
    # Flatten the article embedding tensor
    flat_article = Flatten()(embedding_article)

    # Attention mechanism to capture interaction between user and article embeddings
    attention = Attention()([flat_user, flat_article])

    # Concatenate flattened user, article, and attention vectors
    merged = Concatenate()([flat_user, flat_article, attention])

    # Dense layer with ReLU activation to capture complex patterns
    hidden = Dense(128, activation='relu')(merged)
    # Final output layer with Sigmoid activation for binary output
    output = Dense(1, activation='sigmoid')(hidden)

    # Create the model using Keras functional API
    model = Model(inputs=[input_user, input_article], outputs=output)
    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Return the compiled model
    return model


# Define a function to train a given neural network model using provided data.
# Parameters:
#   - model: The neural network model to be trained.
#   - train_df: The training data in DataFrame format.
#   - user_col: Column name representing user indices (default is 'user_id').
#   - article_col: Column name representing article indices (default is
#     'click_article_id').
#   - timestamp_col: Column name representing click timestamps (default is
#     'click_timestamp').
#   - epochs: Number of training epochs (default is 10).
#   - batch_size: Size of mini-batches used during training (default is 64).
#   - validation_split: Fraction of training data for validation (default is 0.2).

def train_model(model, train_df, user_col='user_id', article_col='click_article_id', 
                timestamp_col='click_timestamp', epochs=10, batch_size=64, 
                validation_split=0.2):
    # Train the model using fit method with user indices, article indices, and
    # timestamps from the training DataFrame.
    # The training process runs for a specified number of epochs and utilizes
    # mini-batches of a specified size.
    # Optionally, a portion of the training data can be used for validation.
    model.fit(
        [train_df[user_col], train_df[article_col]],
        train_df[timestamp_col],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )
    
def make_predictions(model, df, user_col='user_id', article_col='click_article_id'):
    return model.predict([df[user_col], df[article_col]])

# Define a function to generate top-N recommendations for users based on
# predictions and user-item interaction data.
# Parameters:
#   - predictions: Matrix of predicted values from a recommendation model.
#   - df: DataFrame containing user-item interaction data.
#   - user_col: Column name representing user IDs.
#   - item_col: Column name representing item (or article) IDs.
#   - n: Number of top recommendations to generate for each user (default is 5).

def get_top_n_recommendations(predictions, df, user_col, item_col, n=5):
    # Initialize an empty dictionary to store top-N recommendations for each user
    top_n = {}
    
    # Iterate over unique user IDs in the provided DataFrame
    for i, user_id in enumerate(df[user_col].unique()):
        # Filter predictions matrix for the current user
        user_mask = (df[user_col] == user_id)
        user_predictions = predictions[user_mask].flatten()
        
        # Identify indices of top-N predicted items in descending order
        top_items = np.argsort(user_predictions)[::-1][:n]
        
        # Extract unique item IDs corresponding to top indices
        top_item_ids = df[item_col].iloc[top_items].unique()
        
        # Store user ID and top-N item IDs in the dictionary
        top_n[user_id] = top_item_ids
    
    # Return the dictionary containing top-N recommendations for each user
    return top_n

# Define a main function to execute steps for building and evaluating a
# recommendation system.
# Parameters:
#   - file_path: File path for the data in CSV format (default is a specific path).
#   - embedding_size: Size of embedding vectors in the model (default is 50).
#   - epochs: Number of training epochs for the model (default is 10).
#   - batch_size: Size of mini-batches during training (default is 64).
#   - validation_split: Fraction of training data for validation (default is 0.2).

def main(file_path="/Volumes/KINGSTON/archive2/clicks_sample.csv",
         embedding_size=50, epochs=10, batch_size=64, validation_split=0.2):
    # Load data from the specified CSV file path
    df = load_data(file_path)
    
    # Preprocess data, including splitting into train and test sets, and encoding
    train_df, test_df, user_encoder, article_encoder = preprocess_data(df)

    # Determine the number of unique users and articles in the dataset
    num_users = df['user_id'].nunique()
    num_articles = df['click_article_id'].nunique()

    # Build a neural network model with specified embedding size
    model = build_model(num_users, num_articles, embedding_size)
    
    # Train the model on the training data
    train_model(model, train_df, epochs=epochs, batch_size=batch_size, 
                validation_split=validation_split)

    # Make predictions for all interactions in the original dataset
    all_predictions = make_predictions(model, df)

    # Get top 5 recommendations for each user based on predictions
    top_recommendations = get_top_n_recommendations(all_predictions, df,
                                                    'user_id', 'click_article_id', n=5)

    # Print top recommendations for each user
    for user, recommendations in top_recommendations.items():
        print(f"\nTop 5 recommendations for User {user}: {recommendations}")

if __name__ == "__main__":
    main()
