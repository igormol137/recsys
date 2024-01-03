# content-based-filtering.py
#
# Igor Mol <igor.mol@makes.ai>
#
# The program implements a recommendation system using TF-IDF (Term Frequency-Inverse Document
# Frequency) and cosine similarity calculations. It starts by preprocessing a DataFrame with
# user-article interaction data, extracting unique user-article pairs, ensuring no duplicates,
# and converting 'click_article_id' to strings. The subsequent functions perform a basic TF-IDF
# calculation from scratch, covering term frequency (TF) and inverse document frequency (IDF).
# The resulting TF-IDF matrix represents word importance in a document relative to the entire
# dataset. Additionally, the program computes cosine similarity from the TF-IDF matrix, measuring
# document similarity. Finally, a function is provided to obtain top recommendations for a user
# based on their similarity scores.


import pandas as pd
import numpy as np

# preprocess_data():
# Function to preprocess data for a collaborative filtering recommendation system.
# Parameters:
#   - df: Pandas DataFrame, the input DataFrame containing user interactions with articles.
# Objective:
#   This function extracts unique user-article pairs from the input DataFrame,
#   and converts the 'click_article_id' column to strings for compatibility with
#   collaborative filtering algorithms.
# Returns:
#   A Pandas DataFrame with columns 'user_id' and 'click_article_id' containing
#   unique user-article pairs with article IDs represented as strings.

def preprocess_data(df):
    # Extract unique user-article pairs
    article_data = df[['user_id', 'click_article_id']].drop_duplicates()

    # Convert 'click_article_id' to string
    article_data['click_article_id'] = article_data['click_article_id'].astype(str)

    return article_data

# The function `simple_tokenizer(text)' performs simple tokenization for TF-IDF.
# Objective:
#   This function takes a text input and converts it to lowercase, splitting
#   it into a list of tokens.
# Parameters:
#   - text: str, the input text to be tokenized.
# Returns:
#   A list of tokens obtained by converting the input text to lowercase and
#   splitting it.

def simple_tokenizer(text):
    # Simple tokenizer for TF-IDF
    return text.lower().split()

# The function `tfidf_from_scratch' calculate the TF-IDF matrix.
# Objective:
#   This function takes a corpus of documents and computes the TF-IDF matrix
#   using the term frequency (TF) and inverse document frequency (IDF) values.
# Parameters:
#   - corpus: list of str, where each string is a document in the corpus.
# Returns:
#   A NumPy array representing the TF-IDF matrix for the given corpus.

def tfidf_from_scratch(corpus):
    # Calculate TF-IDF matrix from scratch
    
    # Calculate term frequency (TF)
    tf = {}
    for doc in corpus:
        tokens = simple_tokenizer(doc)
        for token in tokens:
            if token not in tf:
                tf[token] = 1
            else:
                tf[token] += 1
    
    # Calculate inverse document frequency (IDF)
    idf = {}
    total_docs = len(corpus)
    for doc in corpus:
        tokens = set(simple_tokenizer(doc))
        for token in tokens:
            if token not in idf:
                idf[token] = 1
            else:
                idf[token] += 1
    
    for token in idf:
        idf[token] = np.log(total_docs / idf[token])
    
    # Calculate TF-IDF
    tfidf_matrix = np.zeros((total_docs, len(tf)))
    
    for i, doc in enumerate(corpus):
        tokens = simple_tokenizer(doc)
        for j, token in enumerate(tf.keys()):
            tfidf_matrix[i, j] = tf.get(token, 0) * idf.get(token, 0)
    
    return tfidf_matrix

# The function `cosine_similarity' computes the cosine similarity matrix.
# Objective:
#   This function calculates the cosine similarity matrix for a given input matrix.
# Parameters:
#   - matrix: 2D NumPy array, where each row represents a vector.
# Returns:
#   A 2D NumPy array representing the cosine similarity matrix.

def cosine_similarity_from_scratch(matrix):
    # Compute cosine similarity matrix from scratch
    
    if matrix.ndim > 1:
        matrix = matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    
    dot_product = np.dot(matrix, matrix.T)
    
    return dot_product

# Function to get top k recommendations for a user.
# Objective:
#   This function retrieves the top k article recommendations for a given user based on
#   cosine similarity scores.
# Parameters:
#   - user_id: User identifier for whom recommendations are generated.
#   - cosine_sim: Cosine similarity matrix between users and articles.
#   - article_data: DataFrame containing user-article interactions.
#   - k: Number of recommendations to retrieve (default is 5).
# Returns:
#   A pandas Series containing the article_ids of the top k recommended articles.

def get_top_recommendations(user_id, cosine_sim, article_data, k=5):
    # Get top k recommendations for a user
    
    # Find the index of the user in the dataframe
    user_index = article_data[article_data['user_id'] == user_id].index[0]
    
    # Get the cosine similarity scores for the user
    user_similarity = cosine_sim[user_index]
    
    # Get indices of articles sorted by similarity (excluding the user's own article)
    sim_scores = list(enumerate(user_similarity))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
    article_indices = [i[0] for i in sim_scores]
    
    # Get the article_ids corresponding to the indices
    recommendations = article_data.loc[article_indices, 'click_article_id']
    
    return recommendations


def main():
    # Load your data into a DataFrame (replace this with your actual data path)
    df = pd.read_csv("/Volumes/KINGSTON/archive2/clicks_sample.csv")
    
    # Preprocess data
    article_data = preprocess_data(df)
    
    # Calculate TF-IDF matrix from scratch
    corpus = article_data['click_article_id'].tolist()
    tfidf_matrix_scratch = tfidf_from_scratch(corpus)
    
    # Compute cosine similarity matrix from scratch
    cosine_sim = cosine_similarity_from_scratch(tfidf_matrix_scratch)
    
    # Print top 5 recommendations for all users
    all_users = df['user_id'].unique()
    
    for user_id in all_users:
        recommendations = get_top_recommendations(user_id, cosine_sim, article_data, k=5)
        print(f"User {user_id}: {recommendations.tolist()}")

if __name__ == "__main__":
    main()

