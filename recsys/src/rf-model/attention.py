# Recommender System Using Attention-mechanism
#
# Igor Mol <igor.mol@makes.ai>
#
# The following code implements a deep learning approach to a recommender system
# with an attention mechanism, using the RFM (Recency, Frequency, Monetary)
# paradigm. 

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.callbacks import EarlyStopping
from tabulate import tabulate

# The preprocess_data function prepares the input data for collaborative 
# filtering, creating necessary features and mappings for users and items.
# The primary objectives include normalizing temporal, frequency, and monetary
# features, creating a unified RFM score, and generating user and item indices.
#
# Parameters:
#   - file_path: The file path to the input dataset in CSV format.
#
# Returns:
#   - data: A DataFrame containing the preprocessed data with additional 
#     features such as recency, frequency, monetary, unified RFM score, and 
#     indices for users and items.
#   - user_mapping: A dictionary mapping customer IDs to corresponding indices.
#   - item_mapping: A dictionary mapping item IDs to corresponding indices.
#
# Preprocessing Steps:
#   - Read the input dataset from the specified file path.
#   - Convert the 'purchase_date' column to datetime format.
#   - Calculate recency as the normalized difference between the maximum 
#     purchase date and each purchase date.
#   - Compute frequency as the normalized count of purchases for each customer.
#   - Normalize the monetary feature by dividing it by the maximum amount.
#   - Create a unified RFM score using specified weights for recency, frequency,
#     and monetary.
#   - Generate user and item mappings with unique indices.
#   - Map customer and item IDs to their corresponding indices in the dataset.
#   - Return the preprocessed data, user mapping, and item mapping.

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['recency'] = (data['purchase_date'].max() - data['purchase_date']).dt.days
    data['frequency'] = data.groupby('customer_id')['purchase_date'].transform('count')
    data['recency'] = data['recency'] / data['recency'].max()
    data['frequency'] = data['frequency'] / data['frequency'].max()
    data['monetary'] = data['amount'] / data['amount'].max()  # Normalize monetary

    # Create a unified RFM score
    data['unified_rfm'] = 0.4 * data['recency'] + 0.4 * data['frequency'] + 0.2 * data['monetary']

    user_mapping = {user: idx for idx, user in enumerate(data['customer_id'].unique())}
    item_mapping = {item: idx for idx, item in enumerate(data['item_id'].unique())}
    data['user_idx'] = data['customer_id'].map(user_mapping)
    data['item_idx'] = data['item_id'].map(item_mapping)

    return data, user_mapping, item_mapping

# The build_model function creates a neural collaborative filtering model
# with three inputs: user, item, and unified RFM (Recency, Frequency, Monetary) scores.
# The objective is to learn the latent representations of users and items and predict
# user-item interactions using the unified RFM scores.
#
# Parameters:
#   - user_mapping: A mapping of user IDs to integer indices.
#   - item_mapping: A mapping of item IDs to integer indices.
#
# Returns:
#   - A compiled Keras model for collaborative filtering with unified RFM scores.
#
# Model Architecture:
#   - Embedding layers are used to transform user and item indices into dense vectors.
#   - The user and item embeddings are flattened and concatenated with the unified RFM scores.
#   - The concatenated features are passed through dense layers with ReLU activation.
#   - The final layer outputs a single value representing the predicted interaction.


def build_model(user_mapping, item_mapping):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    unified_rfm_input = Input(shape=(1,), name='unified_rfm_input')

    user_embedding = Embedding(input_dim=len(user_mapping), output_dim=50, input_length=1)(user_input)
    item_embedding = Embedding(input_dim=len(item_mapping), output_dim=50, input_length=1)(item_input)

    user_flat = Flatten()(user_embedding)
    item_flat = Flatten()(item_embedding)

    concat = Concatenate()([user_flat, item_flat, unified_rfm_input])
    dense = Dense(100, activation='relu')(concat)
    output = Dense(1)(dense)

    model = Model(inputs=[user_input, item_input, unified_rfm_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# The train_model function trains a collaborative filtering model using the 
# provided training data, which includes user indices, item indices, unified 
# RFM scores, and corresponding interaction amounts (e.g., purchase amounts).
#
# Parameters:
#   - model: The collaborative filtering model to be trained.
#   - train_data: A DataFrame containing training data with columns 'user_idx',
#     'item_idx', 'unified_rfm', and 'amount'.
#   - epochs: The number of training epochs (default is 20).
#   - validation_split: The fraction of training data to be used for validation
#     during training (default is 0.2).
#
# Returns:
#   - None (The function modifies the provided model in place).
#
# Training Procedure:
#   - The model is trained using the provided training data, including user 
#     indices, item indices, unified RFM scores, and interaction amounts.
#   - The training process may stop early if the validation performance does not
#     improve within a specified patience period (early stopping).
#   - The function modifies the provided model with the learned weights.

def train_model(model, train_data, epochs=20, validation_split=0.2):
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(
        [train_data['user_idx'], train_data['item_idx'], train_data['unified_rfm']],
        train_data['amount'],
        epochs=epochs, validation_split=validation_split, callbacks=[early_stopping]
    )

# The predict_recommendations function generates predictions for the test data
# using the collaborative filtering model. The main goal is to predict amounts 
# of customer-item interactions based on the learned model.
#
# Parameters:
#   - model: The trained collaborative filtering model.
#   - test_data: DataFrame containing test data with user and item indices,
#     as well as unified RFM scores.
#
# Returns:
#   - A DataFrame with predictions appended as 'predicted_amount'. The output
#     includes the top 3 predicted recommendations for each customer, sorted by
#     the predicted amount in descending order.
#
# Prediction Steps:
#   - Utilize the trained model to predict amounts for the test data.
#   - Append the predicted amounts to the test data DataFrame.
#   - Group the data by customer ID and select the top 3 predictions for each
#     customer based on the predicted amount.
#   - Return the resulting DataFrame with predicted recommendations.

def predict_recommendations(model, test_data):
    predictions = model.predict([test_data['user_idx'], test_data['item_idx'], test_data['unified_rfm']])
    test_data['predicted_amount'] = predictions
    return test_data.groupby('customer_id').apply(lambda group: group.nlargest(3, 'predicted_amount')).reset_index(drop=True)

# The print_recommendations function displays the top recommendations for all
# users based on predicted amounts from the collaborative filtering model.
# The primary purpose is to format and print the recommendations in a tabular
# grid for better readability.
#
# Parameters:
#   - recommendations: DataFrame containing predicted recommendations, including
#     customer IDs, item IDs, and corresponding predicted amounts.
#
# Returns:
#   - None (Prints the formatted table of top recommendations for all users).
#
# Display Steps:
#   - Use the 'tabulate' function to format the recommendations DataFrame in a
#     grid with specified headers.
#   - Print an informative header indicating that the displayed table represents
#     the top recommendations for all users.
#   - Print the formatted table containing customer IDs, item IDs, and predicted
#     amounts for each recommendation.

def print_recommendations(recommendations):
    table = tabulate(recommendations[['customer_id', 'item_id', 'predicted_amount']],
                     headers=['Customer ID', 'Item ID', 'Predicted Amount'], tablefmt='grid', showindex=False)
    print("Top Recommendations for All Users:")
    print(table)

def main():
    file_path = "./generated_data.csv"

    # Data Preprocessing
    data, user_mapping, item_mapping = preprocess_data(file_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Build and Train the Model
    model = build_model(user_mapping, item_mapping)
    train_model(model, train_data)

    # Make Predictions and Print Recommendations
    recommendations = predict_recommendations(model, test_data)
    print_recommendations(recommendations)

if __name__ == "__main__":
    main()
