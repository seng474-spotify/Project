#T. Pires - knn.py - K-NN python script for SENG474 project.
#Created 2025-03-08
#Last Updated 2025-03-08

#Dependencies
import kagglehub #pip install kagglehub
import numpy as np #pip install numpy
import pandas as pd #pip install pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Control Switches
debug = 0 #Switch for debugging

def main():

    print("\nStarted execution of knn.py...\n")

    #https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks
    #Ensure the latest version of the dataset is downloaded.
    path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks") #If this is failing for some reason, look in: C:\Users\(insert user name here)\.cache\kagglehub\datasets\amitanshjoshi\spotify-1million-tracks\versions\1
    path += r"\spotify_data.csv" #Ensure path points to the CSV file instead of the directory

    if debug == 1:
        print("Path to dataset files:", path)
        
    print("Dataset updated.")

    data = pd.read_csv(path)
    
    #Filter to include only the 8 highest occurring genres - classification granularity metric 
    topEightGenres = data["genre"].value_counts().index[:8]
    data = data[data["genre"].isin(topEightGenres)]

    #Randomly select a % of the dataset (62500 (0.0625) at a 80/20 split ensures the training set is of size 50000) - performance (training set size) metric
    data = data.sample(frac = 0.0625, random_state = 8) #Tested on random_state=8

    #Get root(n) to use as k
    dataInstances = len(data)
    rootN = int(np.sqrt(dataInstances)) #Ensure the value is a whole number

    #Split data into feature matrix and target/label vector
    x = data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"]) #Remove non-features. "Unnamed: 0" is the enumeration column. 
    y = data["genre"]
    
    if debug == 1:
        print(x.columns.tolist())
        print(y)

    print("Dataset split into feature matrix and target/label vector.")

    #Normalize all relevant features using StandardScaler
    scaler = StandardScaler()
    xNormalized = scaler.fit_transform(x.to_numpy())  # Convert DataFrame to NumPy array

    #Convert the normalized NumPy array back to a DataFrame
    xNormalized = pd.DataFrame(xNormalized, columns = x.columns)

    if debug == 1:
        print(xNormalized)

    xTrain, xTest, yTrain, yTest = train_test_split(xNormalized, y, test_size = 0.2, random_state = 8) #Tested on random_state=8
    print("Training and testing sets created from dataset.")

    print("\nTraining KNN Classifier...")

    testClassifier = trainKNNClassifier(xTrain, yTrain, rootN)

    print("KNN Classifier trained.")
    print("Training accuracy: " + str(testClassifier.score(xTrain, yTrain) * 100) + "%.")
    print("Testing accuracy: " + str(testClassifier.score(xTest, yTest) * 100) + "%.")
    
    print("\nFinished execution of knn.py.")

def trainKNNClassifier(xTrain, yTrain, k):
    
    KNNClassifier = KNeighborsClassifier(n_neighbors = k)
    KNNClassifier.fit(xTrain, yTrain)

    return KNNClassifier

if __name__ == "__main__":
    main()
