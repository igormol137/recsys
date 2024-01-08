import pandas as pd
import numpy as np

# The load_data function reads a CSV file containing customer transaction data
# and performs preprocessing on the DataFrame.
# The main objectives are to load the data, convert the 'purchase_date' column
# to datetime format, calculate the recency in days, and return the processed
# DataFrame.
# Parameters:
#   - file_path: The file path to the CSV file containing customer transaction
#     data.
# Returns:
#   - df: A DataFrame with the loaded and processed customer transaction data,
#     including the 'purchase_date' column converted to datetime and the
#     calculated 'recency' column in days.
def load_data(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Convert the 'purchase_date' column to datetime format
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    # Calculate the recency in days relative to today
    df['recency'] = (pd.to_datetime('today') - df['purchase_date']).dt.days
    
    # Return the processed DataFrame
    return df

# The calculate_rfm_scores function computes RFM (Recency, Frequency,
# Monetary) scores for each customer in the provided dataset.
# The main objectives are to aggregate customer-specific information and
# derive a unified RFM score.
# Parameters:
#   - data: The input dataset containing customer transactions and
#     information.
# Returns:
#   - rfm_df: A DataFrame containing customer-specific RFM scores,
#     including recency, monetary, frequency, and the unified RFM score.

def calculate_rfm_scores(data):
    # Group the data by customer_id and aggregate relevant metrics
    rfm_df = data.groupby('customer_id').agg({
        'recency': 'min',           # Minimum recency (days since last purchase)
        'amount': 'sum',            # Total monetary value of purchases
        'purchase_date': 'count'    # Count of purchases (frequency)
    }).reset_index()
    
    # Rename columns for clarity
    rfm_df.columns = ['customer_id', 'recency', 'monetary', 'frequency']
    
    # Calculate the unified RFM score by summing individual scores
    rfm_df['rfm_score'] = rfm_df['recency'] + rfm_df['monetary'] + rfm_df['frequency']
    
    # Return the DataFrame containing RFM scores
    return rfm_df

def standardize_features(data):
    return (data - data.mean()) / data.std()

# The kmeans function performs k-means clustering on a given dataset.
# The main objectives are to assign data points to clusters and determine
# cluster centroids.
# Parameters:
#   - X: The dataset, a numpy array with shape (n_samples, n_features).
#   - n_clusters: The number of clusters to form.
#   - max_iters: The maximum number of iterations for the k-means algorithm.
#     Default is 100.
#   - random_state: Seed for the random number generator to ensure
#     reproducibility. Default is None.
# Returns:
#   - labels: An array containing the cluster labels assigned to each data
#     point.
#   - centroids: An array representing the final cluster centroids.

def kmeans(X, n_clusters, max_iters=100, random_state=None):
    # Set the seed for the random number generator if a random_state is provided
    if random_state:
        np.random.seed(random_state)
    
    # Initialize centroids by randomly choosing data points without replacement
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False), :]
    
    # Iterate for a maximum of max_iters
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.abs(X - centroids.reshape(1, -1)), axis=1)
        
        # Update centroids based on the mean of the assigned points in each cluster
        centroids = np.array([X[labels == i].mean() for i in range(n_clusters)]).reshape(-1, 1)
    
    # Return the final cluster labels and centroids
    return labels, centroids

def main():
    file_path = "./generated_data.csv"
    
    # Load data
    data = load_data(file_path)
    
    # Calculate RFM scores
    rfm_data = calculate_rfm_scores(data)
    
    # Standardize RFM scores
    rfm_scaled = standardize_features(rfm_data['rfm_score'])
    
    # Convert the scaled RFM scores to a numpy array
    X = np.array(rfm_scaled).reshape(-1, 1)
    
    # Apply k-means clustering
    n_clusters = 3
    labels, centroids = kmeans(X, n_clusters, random_state=42)
    
    # Assign cluster labels to the DataFrame
    rfm_data['cluster'] = labels
    
    # Print the customers in a ranking based on the assigned clusters
    print("Customer Ranking:")
    ranked_customers = rfm_data.sort_values(by='cluster').reset_index(drop=True)
    print(ranked_customers[['customer_id', 'rfm_score', 'cluster']])

if __name__ == "__main__":
    main()

