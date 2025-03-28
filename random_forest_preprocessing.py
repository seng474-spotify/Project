# Data Mining - Winter 2025 
# Random Forests Preprocessing Implementation
# author: Austin Goertz
# student number: V00987134

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

def load_dataset(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def display_num_songs_each_genre(songs: pd.DataFrame) -> None:
    genre_counts = songs['genre'].value_counts()
    print("Number of Songs Per Genre:")
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")

def drop_features(songs: pd.DataFrame) -> pd.DataFrame:
    return songs.drop(['artist_name', 'track_name', 'track_id', 'key'], axis=1)

# Functiom simply converts data from pandas dataframe to numpy arrays so they behave nicer with scikit learn.
def format_dataset(songs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Separate features and target
    X = songs.drop(columns=['genre']).to_numpy()
    y = songs['genre'].to_numpy()
    return X, y

def split_dataset(X: np.ndarray, y: np.ndarray, training_set_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Validate training set size
    if not (0 < training_set_size < 1):
        raise ValueError("Training set size must be a float between 0 and 1.")
    
    return train_test_split(X, y, train_size=training_set_size)

def select_most_dissimilar_genres(songs: pd.DataFrame, n=8):
    """
    Selects the n most dissimilar genres based on the centroids of their feature vectors.
    
    Parameters:
        X (np.array or pd.DataFrame): Feature matrix.
        y (np.array or pd.Series): Genre labels.
        n (int): Number of genres to select.
        metric (str): Distance metric to use (default is 'euclidean').
        
    Returns:
        selected_genres (list): List of selected genre labels.
    """
    
    # So we don't modify the songs dataframe by accident. 
    df = songs.copy()

    # Calculate the centroid for each genre
    centroids = df.groupby('genre').mean()
    
    # Compute the pairwise distance matrix between genre centroids
    distance_matrix = pairwise_distances(centroids, metric='euclidean')
    genres = list(centroids.index)
    
    # Select genres that are dissimilar
    # Start with the two genres that are furthest apart
    max_distance_index = np.unravel_index(np.argmax(distance_matrix, axis=None), distance_matrix.shape)
    selected_indices = list(max_distance_index)
    remaining_indices = list(set(range(len(genres))) - set(selected_indices))
    
    while len(selected_indices) < n and remaining_indices:
        # For each remaining genre, compute its distance to the closest already selected genre
        min_distances = []
        for ri in remaining_indices:
            dists = [distance_matrix[ri, si] for si in selected_indices]
            min_distances.append(min(dists))
        # Select the genre that maximizes this minimum distance
        next_index = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)
    
    # Get the corresponding genre names
    selected_genres = [genres[i] for i in selected_indices]
    
    return selected_genres

def filter_dataset_by_genre(X: np.ndarray, y: np.ndarray, genres: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # Create a boolean mask where True indicates that the element in y is in the genres list.
    mask = np.isin(y, genres)
    
    # Use the mask to filter both X and y
    filtered_X = X[mask]
    filtered_y = y[mask]
    
    return filtered_X, filtered_y

def normalize(X: np.ndarray) -> np.ndarray:
    # Compute the mean and standard deviation for each feature (column)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    
    # To avoid division by zero in case any feature has zero variance
    stds[stds == 0] = 1
    
    # Perform normalization
    X_normalized = (X - means) / stds
    return X_normalized


def preprocess_data(filepath: str) -> tuple:
    '''
    Returns preproccessed data given file handle.
    '''
    songs = load_dataset("spotify_data.csv")
    songs = drop_features(songs)
    # Pass the DataFrame to the genre selection function.
    dissimilar_genres = select_most_dissimilar_genres(songs, 12) # 12 = number of genres
    print(dissimilar_genres)
    X, y = format_dataset(songs)
    X, y = filter_dataset_by_genre(X, y, dissimilar_genres)
    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.8)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    return X_train, X_test, y_train, y_test






