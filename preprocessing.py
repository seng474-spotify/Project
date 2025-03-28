import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split


def normalize_feature(data,column_number, min, max):
    for i in range(len(data)):
        data[i][column_number] = (data[i][column_number] - min) / (max-min)

def find_genre(file_path, data_destination):

    df = pd.read_csv(file_path, usecols=["genre"])
    unique_genres = df['genre'].unique()

    with open(data_destination, "w") as f:

        for genre in unique_genres:
            f.write(genre + '\n')

    f.close()

def label_genre(genre_file_path, data):
    with open(genre_file_path, "r") as f:
        for i, line in enumerate(f):
            data[i] = line.strip()
    f.close()

def reduce_data(num_rows, X, Y, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))[:num_rows]
    return X.iloc[indices], Y.iloc[indices]

def preprocess_data(filepath: str, num_train_rows: int = 50000, num_genres: int = None):

    # Load the dataset
    df = pd.read_csv(filepath)
    
    # If num_genres is specified, filter to include only the top num_genres by frequency.
    if num_genres is not None:
        genre_counts = df['genre'].value_counts()
        top_genres = genre_counts.nlargest(num_genres).index.tolist()
        df = df[df['genre'].isin(top_genres)]
        # Save the filtered genres to a file for later use (e.g., label encoding)
        with open("genre_list.txt", "w") as f:
            for genre in top_genres:
                f.write(f"{genre}\n")
    else:
        find_genre(filepath, "genre_list.txt")
    
    # Separate features and target
    X = df.drop(columns=['genre'])
    Y = df['genre']
    
    # First, split data into training and testing sets
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Now, reduce the training set to the specified number of rows (keeping X and Y aligned)
    X_train, Y_train = reduce_data(num_train_rows, X_train_full, Y_train_full)
    
    # Convert training and test sets to numpy arrays
    np_x_train = X_train.to_numpy()
    np_x_test = X_test.to_numpy()
    np_y_train = Y_train.to_numpy()
    np_y_test = Y_test.to_numpy()
    
    # Normalization on training set features
    normalize_feature(np_x_train, 4, 0, 100)    # Normalize 'popularity' (0 to 100)
    normalize_feature(np_x_train, 5, 2000, 2023)  # Normalize 'year' (2000 to 2023)
    normalize_feature(np_x_train, 8, -1, 11)      # Normalize 'key' (-1 to 11)
    normalize_feature(np_x_train, 9, -60, 0)       # Normalize 'loudness' (-60 to 0)
    normalize_feature(np_x_train, 16, 0, 250)      # Normalize 'tempo' (0 to 250)
    normalize_feature(np_x_train, 17, 3, 7)        # Normalize 'time signature' (3 to 7)
    
    # Apply the same normalization to the test set
    normalize_feature(np_x_test, 4, 0, 100)
    normalize_feature(np_x_test, 5, 2000, 2023)
    normalize_feature(np_x_test, 8, -1, 11)
    normalize_feature(np_x_test, 9, -60, 0)
    normalize_feature(np_x_test, 16, 0, 250)
    normalize_feature(np_x_test, 17, 3, 7)
    
    # Remove unwanted columns (e.g., columns 1, 2, and 3) from both training and test sets
    np_x_train = np.delete(np_x_train, [1, 2, 3], axis=1)
    np_x_test = np.delete(np_x_test, [1, 2, 3], axis=1)
    
    # Label encode the genres using the list in genre_list.txt
    label_genre("genre_list.txt", np_y_train)
    label_genre("genre_list.txt", np_y_test)
    
    return np_x_train, np_y_train, np_x_test, np_y_test



"""
file_path = "spotify_data.csv"  # Update this path to your actual file
find_genre(file_path, "genre_list.txt")

df = pd.read_csv(file_path)


X = df.drop(columns=['genre'])
Y = df['genre']

#x = df.iloc[:,:-1] #features
#y = df.iloc[:,-1] #label/last col

np_x = X.to_numpy()
np_y = Y.to_numpy()


#popularity 0 to 100
normalize_feature(np_x, 4, 0, 100)
#year 2000 to 2023 -> doesnt really matter
normalize_feature(np_x, 5, 2000, 2023)
#key -1 to 11
normalize_feature(np_x, 8, -1, 11)
#loudness -60 to 0
normalize_feature(np_x, 9, -60, 0)
#tempo 0 to 250
normalize_feature(np_x, 16, 0, 250)
#time signature 3 to 7
normalize_feature(np_x, 17, 3, 7)

np_x = np.delete(np_x, [1,2,3], axis=1)

label_genre("genre_list.txt", np_y)
"""

