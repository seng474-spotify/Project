import pandas as pd
import numpy as np
import sklearn as sk


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


file_path = "Data/spotify_data.csv"  # Update this path to your actual file
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
