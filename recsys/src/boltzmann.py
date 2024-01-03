# Deep learning Recommendation System using a restricted Boltzmann machine
#
# Igor Mol <igor.mol@makes.ai>
#
# A restricted Boltzmann machine (RBM) is a type of neural network used in
# machine learning for unsupervised learning tasks. RBMs consist of visible and
# hidden layers; the connections between layers have no internal connections.
# They are capable of learning probability distributions over their set of
# inputs.
# 	This program implements a recommendation system using a deep learning
# approach with a restricted Boltzmann machine. The recommendation process is
# encapsulated within a set of functions and a main program. The functions
# include loading and preprocessing data, building and training the
# recommendation model, making predictions, and extracting top N
# recommendations. The main program orchestrates the entire recommendation
# process by creating an instance of the RecommendationModel class, loading and
# preprocessing the data, building and training the model, making predictions,
# and printing the top 5 recommendations for each user.

# load_data():
# Define a function to load data from a CSV file into a Pandas DataFrame.

def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.

    Parameters:
    - file_path: The path to the CSV file.

    Returns:
    - df: The Pandas DataFrame containing the loaded data.
    """
    # Use Pandas read_csv function to load data into a DataFrame
    df = pd.read_csv(file_path)

    # Return the loaded DataFrame
    return df


# preprocess_data():
# Define a function to preprocess user-item interaction data.

def preprocess_data(df):
    """
    Encode user and article IDs and split the data into training and testing sets.

    Parameters:
    - df: The Pandas DataFrame containing user-item interaction data.

    Returns:
    - train_df: The preprocessed training DataFrame.
    """
    # Initialize LabelEncoders for user and article IDs
    user_encoder = LabelEncoder()
    article_encoder = LabelEncoder()

    # Encode user IDs and article IDs using LabelEncoders
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['click_article_id'] = article_encoder.fit_transform(df['click_article_id'])

    # Split the data into training and testing sets
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    # Return the preprocessed training DataFrame
    return train_df

# k_nearest_neighbors():
# Define a function to calculate k-NN recommendations for a user using adjusted 
# cosine similarity and L1 norm.

def k_nearest_neighbors(user_id, df, k=5):
    """
    Calculate k-NN recommendations for a user using adjusted cosine similarity and L1 norm.

    Parameters:
    - user_id: The ID of the target user for whom recommendations are calculated.
    - df: The Pandas DataFrame containing user-item interaction data.
    - k: The number of nearest neighbors to consider for recommendations (default is 5).

    Returns:
    - recommendations: A list of top N recommendations for the specified user.
    """
    # Extract the click timestamps for the target user
    user_clicks = df[df['user_id'] == user_id].set_index('click_article_id')['click_timestamp'].to_dict()

    # Calculate average rating for the user
    user_ratings = {item: np.mean(df[df['user_id'] == user_id]['click_timestamp']) for item in user_clicks}

    # Find unique user IDs in the dataset
    unique_users = df['user_id'].unique()
    similarity_scores = []

    # Calculate adjusted cosine similarity with other users
    for other_user_id in unique_users:
        if other_user_id != user_id:
            other_user_clicks = df[df['user_id'] == other_user_id].set_index('click_article_id')['click_timestamp'].to_dict()

            similarity = adjusted_cosine_similarity(user_clicks, other_user_clicks, user_ratings)
            similarity_scores.append((other_user_id, similarity))

    # Sort similarity scores in descending order and select top k users
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_users = similarity_scores[:k]

    recommendations = set()

    # Extract top items from the top k similar users
    for user, _ in top_k_users:
        user_clicks = df[df['user_id'] == user].set_index('click_article_id')['click_timestamp'].to_dict()
        recommendations.update(user_clicks)

    # Filter out items already clicked by the target user
    recommendations = [item for item in recommendations if item not in user_clicks]

    # Sort recommendations based on the predicted engagement timestamp
    recommendations = sorted(recommendations, key=lambda x: user_clicks.get(x, 0), reverse=True)[:5]

    return recommendations

# train_model():
# Define a function to train a recommendation model using user-item interaction 
# data.

def train_model(model, train_df, epochs=10, batch_size=64, validation_split=0.2):
    """
    Train the recommendation model.

    Parameters:
    - model: The recommendation model to be trained.
    - train_df: The Pandas DataFrame containing the training data with user-item interactions.
    - epochs: The number of training epochs (default is 10).
    - batch_size: The batch size used during training (default is 64).
    - validation_split: The fraction of the training data to use for validation (default is 0.2).

    Returns:
    - None
    """
    # Train the model using user IDs, article IDs, and click timestamps
    model.fit(
        [train_df['user_id'], train_df['click_article_id']],
        train_df['click_timestamp'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )


# make_predictions():
# Define a function to make predictions on the entire dataset using a given 
# recommendation model.

def make_predictions(model, df):
    """
    Make predictions on the entire dataset using a recommendation model.

    Parameters:
    - model: The trained recommendation model.
    - df: The Pandas DataFrame containing user-item interaction data.

    Returns:
    - predictions: Predicted click timestamps for user-item interactions.
    """
    # Use the model to predict click timestamps based on user and item interactions
    predictions = model.predict([df['user_id'], df['click_article_id']])
    
    # Return the predictions for user-item interactions
    return predictions


# get_top_n_recommendations():
# Define a function to get the top N recommendations for each user based on 
# model predictions.

def get_top_n_recommendations(predictions, df, user_col, item_col, n=5):
    """
    Get top N recommendations for each user.

    Parameters:
    - predictions: Model predictions for user-item interactions.
    - df: The Pandas DataFrame containing user-item interaction data.
    - user_col: The column name representing user IDs.
    - item_col: The column name representing item IDs.
    - n: The number of recommendations to retrieve for each user (default is 5).

    Returns:
    - top_n: A dictionary where keys are user IDs, and values are lists of top N recommended item IDs.
    """
    # Initialize an empty dictionary to store top N recommendations for each user
    top_n = {}

    # Iterate through unique user IDs in the DataFrame
    for i, user_id in enumerate(df[user_col].unique()):
        # Create a boolean mask to filter interactions for the current user
        user_mask = (df[user_col] == user_id)
        
        # Flatten the predictions for the current user
        user_predictions = predictions[user_mask].flatten()
        
        # Get the indices of the top N items based on predictions
        top_items = np.argsort(user_predictions)[::-1][:n]
        
        # Inverse transform item indices to item IDs
        top_item_ids = recommendation_model.article_encoder.inverse_transform(top_items)
        
        # Store the top N recommendations for the current user in the dictionary
        top_n[user_id] = top_item_ids
    
    # Return the dictionary containing top N recommendations for each user
    return top_n


if __name__ == "__main__":
    # Main program
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"

    # Load data
    df = load_data(file_path)

    # Preprocess data
    train_df = preprocess_data(df)

    # Build model
    num_users = df['user_id'].nunique()
    num_articles = df['click_article_id'].nunique()
    model = build_model(num_users, num_articles)

    # Train model
    train_model(model, train_df)

    # Make predictions
    all_predictions = make_predictions(model, df)

    # Get and print top 5 recommendations for all users
    top_recommendations = get_top_n_recommendations(all_predictions, df, 'user_id', 'click_article_id', n=5)
    for user, recommendations in top_recommendations.items():
        print(f"\nTop 5 recommendations for User {user}: {recommendations}")
