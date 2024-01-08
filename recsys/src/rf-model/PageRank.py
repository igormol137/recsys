# PageRank Recommender System
#
# Igor Mol <igor.mol@makes.ai>
#
# This code implements a recommender system using customer transaction data.
# Key tasks include calculating RFM (Recency, Frequency, Monetary) scores,
# creating a bipartite graph connecting customers and items with RFM scores,
# computing PageRank scores for node importance, and printing top recommendations.
#
# calculate_rfm_scores Function:
# - Calculates RFM scores for customers in the dataset.
# - Defines functions for recency, frequency, and monetary components.
# - Computes individual RFM components, normalizes them, and combines into a single score.
#
# create_bipartite_graph Function:
# - Generates a bipartite graph from the dataset.
# - Constructs a dictionary representing the graph, connecting customers and items
#   based on RFM scores. Nodes have types ('customer' or 'item') and neighbors.
#
# compute_pagerank Function:
# - Calculates PageRank scores for nodes in a given graph.
# - Initializes and iterates PageRank scores, updating based on neighbors until
#   convergence or a specified maximum number of iterations.
#
# print_top_recommendations Function:
# - Prints top recommendations for all users based on PageRank scores.
# - Takes a list of sorted customers and the bipartite graph as input.
#
# main Function:
# - Entry point of the program.
# - Loads dataset, calculates RFM scores, creates a bipartite graph,
#   computes PageRank scores, sorts customers, and prints top recommendations.


import pandas as pd
import numpy as np

# The calculate_rfm_scores function computes RFM (Recency, Frequency, Monetary)
# scores for each customer in the provided dataset and normalizes them.
# The main objectives are to calculate individual RFM components, normalize the
# scores, and combine them into a single measure.
# Parameters:
#   - data: The input dataset containing customer transactions and information.
# Returns:
#   - None (The function modifies the input DataFrame in place).

def calculate_rfm_scores(data):
    # Define functions to calculate recency, frequency, and monetary components
    def recency(date):
        today = pd.to_datetime('today')
        return (today - pd.to_datetime(date)).days

    def frequency(customer_id):
        return data[data['customer_id'] == customer_id].shape[0]

    def monetary(customer_id):
        return data[data['customer_id'] == customer_id]['amount'].sum()

    # Calculate individual RFM components for each customer
    data['recency'] = data['purchase_date'].apply(recency)
    data['frequency'] = data['customer_id'].apply(frequency)
    data['monetary'] = data['customer_id'].apply(monetary)

    # Normalize the RFM scores
    data['recency_normalized'] = (data['recency'] - data['recency'].min()) / (data['recency'].max() - data['recency'].min())
    data['frequency_normalized'] = (data['frequency'] - data['frequency'].min()) / (data['frequency'].max() - data['frequency'].min())
    data['monetary_normalized'] = (data['monetary'] - data['monetary'].min()) / (data['monetary'].max() - data['monetary'].min())

    # Combine normalized RFM scores into a single measure using specified weights
    data['rfm_score'] = (
        0.4 * data['recency_normalized'] +
        0.4 * data['frequency_normalized'] +
        0.2 * data['monetary_normalized']
    )

# The create_bipartite_graph function generates a bipartite graph from the
# provided dataset, where customers and items form two disjoint sets.
# The main objectives are to construct a bipartite graph with customer and
# item nodes, connecting them based on RFM scores.
# Parameters:
#   - data: The input dataset containing customer-item relationships and RFM
#     scores.
# Returns:
#   - G: A dictionary representing the bipartite graph. Each node has a 'type'
#     (either 'customer' or 'item') and 'neighbors' containing connected nodes
#     with corresponding weights.
def create_bipartite_graph(data):
    # Create an empty dictionary to represent the bipartite graph
    G = {}

    # Iterate over rows in the input dataset to construct the graph
    for _, row in data.iterrows():
        customer_id = row['customer_id']
        item_id = row['item_id']
        weight = row['rfm_score']

        # If the customer node does not exist, add it to the graph
        if customer_id not in G:
            G[customer_id] = {'type': 'customer', 'neighbors': {}}
        
        # If the item node does not exist, add it to the graph
        if item_id not in G:
            G[item_id] = {'type': 'item', 'neighbors': {}}

        # Connect customer and item nodes with corresponding RFM score
        G[customer_id]['neighbors'][item_id] = weight
        G[item_id]['neighbors'][customer_id] = weight

    # Return the generated bipartite graph
    return G

# The compute_pagerank function calculates PageRank scores for nodes in a given
# graph, representing the importance of each node based on its connections.
# The main objectives are to initialize and iterate PageRank scores until
# convergence.
# Parameters:
#   - G: The input graph represented as a dictionary with nodes, their types,
#     and neighbors with corresponding weights.
#   - alpha: Damping factor for PageRank calculation (default is 0.85).
#   - max_iter: Maximum number of iterations for PageRank convergence
#     (default is 100).
#   - tol: Tolerance level for convergence (default is 1e-6).
# Returns:
#   - pagerank_scores: A dictionary containing PageRank scores for each node in
#     the graph.
def compute_pagerank(G, alpha=0.85, max_iter=100, tol=1e-6):
    # Initialize PageRank scores for each node
    pagerank_scores = {node: 1.0 for node in G}

    # Perform PageRank iterations
    for _ in range(max_iter):
        prev_pagerank_scores = pagerank_scores.copy()

        # Update PageRank scores for each node based on neighbors
        for node in G:
            sum_weighted_pageranks = sum(prev_pagerank_scores[neighbor] * G[node]['neighbors'][neighbor] for neighbor in G[node]['neighbors'])
            pagerank_scores[node] = (1 - alpha) + alpha * sum_weighted_pageranks

        # Check for convergence using the tolerance level
        if np.sum(np.abs(np.array(list(pagerank_scores.values())) - np.array(list(prev_pagerank_scores.values())))) < tol:
            break

    # Return the final PageRank scores
    return pagerank_scores

def print_top_recommendations(sorted_customers, G):
    # Print top recommendations for all users
    for customer_id, score in sorted_customers:
        recommended_items = [neighbor for neighbor in G[customer_id]['neighbors'] if G[neighbor]['type'] == 'item']
        print(f"Customer {customer_id}: Top Recommendations - {recommended_items[:5]}")

def main():
    # Load the CSV file into a pandas DataFrame
    csv_file_path = "./generated_data.csv"
    data = pd.read_csv(csv_file_path)

    # Calculate RFM scores
    calculate_rfm_scores(data)

    # Create a bipartite graph
    G = create_bipartite_graph(data)

    # Compute PageRank scores
    pagerank_scores = compute_pagerank(G)

    # Sort customers based on PageRank scores
    sorted_customers = sorted(((k, v) for k, v in pagerank_scores.items() if G[k]['type'] == 'customer'), key=lambda x: x[1], reverse=True)

    # Print top recommendations
    print_top_recommendations(sorted_customers, G)

if __name__ == "__main__":
    main()

