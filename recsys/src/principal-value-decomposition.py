# RecSys Using a Principal Value Decomposition Approach
#
# Igor Mol <igor.mol@makes.ai>
#
# Principal Value Decomposition (PVD) is an algorithm used in collaborative
# filtering recommender systems to analyze and decompose a user-item interaction
# matrix. This technique breaks down the matrix into three matrices, representing
# users, items, and singular values. Singular values indicate the importance of
# each component. PVD helps uncover latent factors in the data, allowing the
# system to make predictions by approximating the original matrix using a reduced
# number of dimensions. This method is effective in handling sparse data and
# revealing hidden patterns in user preferences, facilitating accurate
# recommendations.

import numpy as np
import pandas as pd

# Load the CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# create_user_item_matrix(): This function takes a DataFrame representing
# user-item interactions and converts it into a user-item matrix. This matrix
# is structured with users as rows, items as columns, and the values
# indicating the strength of interaction (e.g., session size or ratings).
#
# Parameters:
#   - df: Pandas DataFrame
#     The input DataFrame containing user-item interaction data, with columns
#     'user_id', 'click_article_id', and 'session_size'.
#
# Returns:
#   A Pandas DataFrame representing the user-item matrix, where rows correspond
#   to users, columns to items, and values to the strength of interactions. The
#   matrix is filled with zeros for user-item pairs with no recorded interaction.

def create_user_item_matrix(df):
    user_item_matrix = {}

    # Iterate over rows in the input DataFrame
    for _, row in df.iterrows():
        user_id = row['user_id']
        article_id = row['click_article_id']
        session_size = row['session_size']

        # Check if the user is already in the matrix, create if not
        if user_id not in user_item_matrix:
            user_item_matrix[user_id] = {}

        # Assign the session size to the corresponding user-item pair
        user_item_matrix[user_id][article_id] = session_size

    # Convert the dictionary to a DataFrame and fill missing values with zeros
    return pd.DataFrame(user_item_matrix).fillna(0)


# matrix_decomposition_pca(): This function uses Principal Component Analysis
# (PCA) to perform matrix decomposition on a user-item matrix. Matrix decomposition
# is a technique to break down a matrix into simpler components, and PCA is a
# specific method to achieve this by extracting the principal components.
#
# Parameters:
#   - user_item_matrix: 2D NumPy array
#     The user-item interaction matrix where rows represent users, columns
#     represent items, and the values represent the strength of interaction.
#   - num_components: Integer (default: 10)
#     The number of principal components to retain during PCA.
#
# Returns:
#   A 2D NumPy array representing the user matrix obtained through matrix
#   decomposition using PCA. Rows correspond to users, and columns represent the
#   retained principal components.

def matrix_decomposition_pca(user_item_matrix, num_components=10):
    # Perform Principal Component Analysis (PCA) on the user-item matrix
    _, _, vt = np.linalg.svd(user_item_matrix, full_matrices=False)
    
    # Extract the user matrix using the first 'num_components' components
    user_matrix = vt[:num_components, :].T

    # Return the user matrix obtained through PCA
    return user_matrix

# predict_ratings() and get_top_recommendations():
# The first function, 'predict_ratings', computes predicted ratings for all
# users and items based on a user matrix. The second function, 'get_top_recom-
# mendations' retrieves the top recommendations for a specified user using the 
# predicted ratings.
#
# Parameters:
#   - user_matrix: 2D NumPy array
#     The user matrix obtained through matrix decomposition or other collaborative
#     filtering techniques.
#
# Returns:
#   - For 'predict_ratings': A 2D NumPy array representing the predicted ratings
#     for all users and items.
#   - For 'get_top_recommendations': A 1D NumPy array containing the indices of the
#     top recommended items for the specified user.

def predict_ratings(user_matrix):
    # Compute predicted ratings using the dot product of the user matrix
    return np.dot(user_matrix, user_matrix.T)

def get_top_recommendations(predicted_ratings, user):
    # Retrieve the indices of the top recommended items for the specified user
    return np.argsort(predicted_ratings[user, :])[::-1][:5]


# Main function to run the recommendation system
def main():
    # Load data
    file_path = "/Volumes/KINGSTON/archive2/clicks_sample.csv"
    df = load_data(file_path)

    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(df)

    # Matrix Decomposition with PCA
    user_matrix = matrix_decomposition_pca(user_item_matrix)

    # Predict ratings
    predicted_ratings = predict_ratings(user_matrix)

    # Print top 5 recommendations for all users
    for user in user_item_matrix.columns:
        top_recommendations = get_top_recommendations(predicted_ratings, user)
        print(f"Top 5 recommendations for User {user}: {top_recommendations}")

# Run the recommendation system
if __name__ == "__main__":
    main()

