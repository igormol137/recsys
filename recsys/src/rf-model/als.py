# ALS-RFM Recommender Engine
#
# Igor Mol <igor.mol@makes.ai>
#
# In building a recommender system, we employ the ALS (Alternating Least Squares)
# algorithm within the framework of the RFM (Recency, Frequency, Monetary) model.
# The ALS algorithm is a collaborative filtering technique that aids in
# understanding user preferences by iteratively optimizing a matrix
# factorization model. Integrating it with the RFM model, which analyzes user
# behavior based on recency, frequency, and monetary value, enhances the
# recommender system's accuracy. By utilizing ALS to learn patterns from the RFM
# metrics, we can predict user preferences more effectively. This integration
# ensures that the recommender system tailors recommendations to individual
# users, considering not only the popularity of items but also the unique
# historical engagement patterns of each user.

import pandas as pd
import numpy as np

def calculate_item_factors(relevant_user_factors, user_item_interactions, rfm_scores, lambda_reg, num_factors):
    """
    Update item factors using ALS.

    Parameters:
    - relevant_user_factors: User factors corresponding to items the user has interacted with
    - user_item_interactions: User-item interactions matrix
    - rfm_scores: Unified RFM scores for the items
    - lambda_reg: Regularization parameter
    - num_factors: Number of latent factors

    Returns:
    - Updated item factors
    """
    weighted_interactions = rfm_scores * user_item_interactions
    return np.linalg.solve(
        np.dot(relevant_user_factors.T, relevant_user_factors) + lambda_reg * np.eye(num_factors),
        np.dot(relevant_user_factors.T, weighted_interactions)
    )

def als_with_rfm(df, num_factors=50, num_iterations=50, lambda_reg=0.1):
    """
    ALS algorithm with unified RFM scores.

    Parameters:
    - df: DataFrame containing user-item interactions and RFM scores
    - num_factors: Number of latent factors
    - num_iterations: Number of iterations for ALS training
    - lambda_reg: Regularization parameter

    Returns:
    - User factors and item factors
    """
    user_item_matrix_with_rfm = pd.pivot_table(df, values=['amount', 'rfm_unified'], index='customer_id', columns='item_id', fill_value=0)
    user_item_array_with_rfm = user_item_matrix_with_rfm.values

    num_users, num_items = user_item_array_with_rfm.shape
    user_factors_with_rfm = np.random.rand(num_users, num_factors)
    item_factors_with_rfm = np.random.rand(num_items, num_factors)

    for _ in range(num_iterations):
        for i in range(num_users):
            user_rfm_scores = user_item_array_with_rfm[i, user_item_array_with_rfm[i, :] > 0]
            relevant_item_factors = item_factors_with_rfm[user_item_array_with_rfm[i, :] > 0, :]

            user_factors_with_rfm[i, :] = np.linalg.solve(
                np.dot(relevant_item_factors.T, relevant_item_factors) + lambda_reg * np.eye(num_factors),
                np.dot(relevant_item_factors.T, user_rfm_scores)
            )

        for j in range(num_items):
            relevant_user_factors = user_factors_with_rfm[user_item_array_with_rfm[:, j] > 0, :]

            # Update: Corrected indexing for rfm_scores
            rfm_scores = user_item_array_with_rfm[user_item_array_with_rfm[:, j] > 0, j]
            
            item_factors_with_rfm[j, :] = calculate_item_factors(
                relevant_user_factors,
                user_item_array_with_rfm[user_item_array_with_rfm[:, j] > 0, j],
                rfm_scores,
                lambda_reg,
                num_factors
            )

    return user_factors_with_rfm, item_factors_with_rfm

def generate_recommendations(user_factors, item_factors, num_recommendations=5):
    """
    Generate recommendations using ALS factors.

    Parameters:
    - user_factors: User factors matrix
    - item_factors: Item factors matrix
    - num_recommendations: Number of recommendations to generate

    Returns:
    - Top recommendations for each user
    """
    num_users = user_factors.shape[0]
    top_recommendations = {}

    for i in range(num_users):
        user_recommendations = np.argsort(np.dot(user_factors[i, :], item_factors.T))[::-1][:num_recommendations]
        top_recommendations[i + 1] = user_recommendations + 1  # Adjust indices to start from 1

    return top_recommendations

def main():
    # Load data and perform ALS with RFM
    csv_file_path = "./generated_data.csv"
    df = pd.read_csv(csv_file_path)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    max_purchase_date = df['purchase_date'].max()
    df['recency'] = (max_purchase_date - df['purchase_date']).dt.days
    rfm_df = df.groupby('customer_id').agg({
        'recency': 'min',
        'purchase_date': 'count',
        'amount': 'sum'
    }).reset_index()
    rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    rfm_df[['recency', 'frequency', 'monetary']] = (
        rfm_df[['recency', 'frequency', 'monetary']] - rfm_df[['recency', 'frequency', 'monetary']].min()
    ) / (rfm_df[['recency', 'frequency', 'monetary']].max() - rfm_df[['recency', 'frequency', 'monetary']].min())
    rfm_df['rfm_unified'] = rfm_df[['recency', 'frequency', 'monetary']].apply(np.prod, axis=1) ** (1/3)
    df = pd.merge(df, rfm_df[['customer_id', 'rfm_unified']], on='customer_id')

    user_factors_with_rfm, item_factors_with_rfm = als_with_rfm(df)

    # Generate and print recommendations
    top_recommendations_with_rfm = generate_recommendations(user_factors_with_rfm, item_factors_with_rfm)
    for user_id, recommendations in top_recommendations_with_rfm.items():
        print(f"Top recommendations for user {user_id}: {recommendations}")

if __name__ == "__main__":
    main()

