#T. Pires - knn.py - K-NN python script for SENG474 project.
#Created 2025-03-08
#Last Updated 2025-03-10

#Dependencies
import kagglehub #pip install kagglehub
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np #pip install numpy
import pandas as pd #pip install pandas
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Control Switches
debug = 0 #Switch for debugging

#Experiment Switches
initialExperimentSwitch = 0 #Initialization
tuneNeighboursSwitch = 0 #n_neighbors
#weights
#algorithm
#leaf_size
#p
tuneDissimilarityMeasureSwitch = 1 #metric (dissimilarity measure)

#Ideal Hyperparameter Values (calculated from experiments)
idealNeighbors = 40

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

    if(initialExperimentSwitch == 1):
        initialExperiment(data)
    
    if(tuneNeighboursSwitch) == 1:
        tuneNeighbours(data)

    if tuneDissimilarityMeasureSwitch == 1:
        tuneDissimilarityMeasure(data)

    print("\nFinished execution of knn.py.")

def tuneDissimilarityMeasure(data):
    print("\nRunning \"tune dissimilarity measure \" experiment...")

    #Filter to include only the 8 highest occurring genres - classification granularity metric 
    topEightGenres = data["genre"].value_counts().index[:8]
    data = data[data["genre"].isin(topEightGenres)]
    print("Reduced dataset to only include highest 8 occurring genres.")

    #Randomly select a % of the dataset - performance (training set size) metric
    #Set to 0.375 (37.5%) because top eight genres are composed of 168309 instances; ensures a training set size slightly greater than 50000 at 80/20 split
    data = data.sample(frac = 0.375) #Tested on random_state = 8
    
    if debug == 1:
        print(data.shape)

    #Split data into feature matrix and target/label vector
    x = data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"]) #Remove non-features. "Unnamed: 0" is the enumeration column. 
    y = data["genre"]
    
    if debug == 1:
        print(x.columns.tolist())

    print("Dataset split into feature matrix and target/label vector.")

    #Normalize all relevant features using StandardScaler
    scaler = StandardScaler()
    xNormalized = scaler.fit_transform(x.to_numpy())  # Convert DataFrame to NumPy array

    #Convert the normalized NumPy array back to a DataFrame
    xNormalized = pd.DataFrame(xNormalized, columns = x.columns)

    if debug == 1:
        print(xNormalized)

    print("Dataset features normalized.")

    xTrain, xTest, yTrain, yTest = train_test_split(xNormalized, y, test_size = 0.2) #Tested on random_state = 8
    print("Training and testing sets created from dataset.")

    #All metrics supported by KNeighborsClassifier
    #haversine invalid in non-binary classification tasks
    #cityblock, manhattan, l1 all the same measure
    #euclidean and l2 same measure (nan euclidean accounts for missing data points, none present in data set)
    metricsGrid = {"metrics": ["cosine", "euclidean", "manhattan"]}
    
    trainAccuracies = []
    testAccuracies = []
    metrics = metricsGrid["metrics"]
    
    print("Performing grid search...")

    for metric in metrics:
        KNNClassifier = trainKNNClassifier(xTrain = xTrain, yTrain = yTrain, metric = metric)
        trainingAccuracy = KNNClassifier.score(xTrain, yTrain) * 100
        testingAccuracy = KNNClassifier.score(xTest, yTest) * 100
        trainAccuracies.append(trainingAccuracy)
        testAccuracies.append(testingAccuracy)
    
    print("Train accuracies:", trainAccuracies)
    print("Test accuracies:", testAccuracies)
    
    #Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(metrics, trainAccuracies, label = "Training Accuracy")
    plt.plot(metrics, testAccuracies, label = "Testing Accuracy")
    plt.xlabel("Dissimilarity Measure")
    plt.ylabel("Accuracy")
    plt.title("Dissimilarity Measure vs. Training and Testing Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Finished execution of \"tune dissimilarity measure\" experiment.")

def tuneNeighbours(data):
    print("\nRunning \"tune neighbours\" experiment...")
    
    #Filter to include only the 8 highest occurring genres - classification granularity metric 
    topEightGenres = data["genre"].value_counts().index[:8]
    data = data[data["genre"].isin(topEightGenres)]
    print("Reduced dataset to only include highest 8 occurring genres.")

    #Randomly select a % of the dataset - performance (training set size) metric
    #Set to 0.375 (37.5%) because top eight genres are composed of 168309 instances; ensures a training set size slightly greater than 50000 at 80/20 split
    data = data.sample(frac = 0.375) #Tested on random_state = 8
    
    if debug == 1:
        print(data.shape)

    #Split data into feature matrix and target/label vector
    x = data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"]) #Remove non-features. "Unnamed: 0" is the enumeration column. 
    y = data["genre"]
    
    if debug == 1:
        print(x.columns.tolist())

    print("Dataset split into feature matrix and target/label vector.")

    #Normalize all relevant features using StandardScaler
    scaler = StandardScaler()
    xNormalized = scaler.fit_transform(x.to_numpy())  # Convert DataFrame to NumPy array

    #Convert the normalized NumPy array back to a DataFrame
    xNormalized = pd.DataFrame(xNormalized, columns = x.columns)

    if debug == 1:
        print(xNormalized)

    print("Dataset features normalized.")

    xTrain, xTest, yTrain, yTest = train_test_split(xNormalized, y, test_size = 0.2) #Tested on random_state = 8
    print("Training and testing sets created from dataset.")

    #Square root of n (assuming n is ~50000) is ~224
    #Use spacing around there to tune neighbours hyperparameter
    neighboursGrid = {"n_neighbors": [10, 40, 70, 100, 130, 160, 190, 220, 250]}
    
    trainAccuracies = []
    testAccuracies = []
    neighbours = neighboursGrid["n_neighbors"]
    
    print("Performing grid search...")

    for k in neighbours:
        KNNClassifier = trainKNNClassifier(xTrain, yTrain, k)
        trainingAccuracy = KNNClassifier.score(xTrain, yTrain) * 100
        testingAccuracy = KNNClassifier.score(xTest, yTest) * 100
        trainAccuracies.append(trainingAccuracy)
        testAccuracies.append(testingAccuracy)
    
    print("Train accuracies:", trainAccuracies)
    print("Test accuracies:", testAccuracies)
    
    #Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(neighbours, trainAccuracies, label = "Training Accuracy")
    plt.plot(neighbours, testAccuracies, label = "Testing Accuracy")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.title("Number of Neighbors vs. Training and Testing Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Finished execution of \"tune neighbours\" experiment.")

def initialExperiment(data):
    print("\nRunning initial experiment...")    

    #Filter to include only the 8 highest occurring genres - classification granularity metric 
    topEightGenres = data["genre"].value_counts().index[:8]
    data = data[data["genre"].isin(topEightGenres)]
    print("Reduced dataset to only include highest 8 occurring genres.")

    #Randomly select a % of the dataset - performance (training set size) metric
    #Set to 0.375 (37.5%) because top eight genres are composed of 168309 instances; ensures a training set size slightly greater than 50000 at 80/20 split
    data = data.sample(frac = 0.375) #Tested on random_state = 8
    
    if debug == 1:
        print(data.shape)

    #Get root(n) to use as k
    dataInstances = len(data)
    rootN = int(np.sqrt(dataInstances)) #Ensure the value is a whole number

    #Split data into feature matrix and target/label vector
    x = data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"]) #Remove non-features. "Unnamed: 0" is the enumeration column. 
    y = data["genre"]
    
    if debug == 1:
        print(x.columns.tolist())

    print("Dataset split into feature matrix and target/label vector.")

    #Normalize all relevant features using StandardScaler
    scaler = StandardScaler()
    xNormalized = scaler.fit_transform(x.to_numpy())  # Convert DataFrame to NumPy array

    #Convert the normalized NumPy array back to a DataFrame
    xNormalized = pd.DataFrame(xNormalized, columns = x.columns)

    if debug == 1:
        print(xNormalized)

    print("Dataset features normalized.")

    xTrain, xTest, yTrain, yTest = train_test_split(xNormalized, y, test_size = 0.2) #Tested on random_state = 8
    print("Training and testing sets created from dataset.")

    print("Training KNN Classifier...")

    testClassifier = trainKNNClassifier(xTrain, yTrain, rootN)

    print("KNN Classifier trained.")
    print("Training accuracy: " + str(testClassifier.score(xTrain, yTrain) * 100) + "%.")
    print("Testing accuracy: " + str(testClassifier.score(xTest, yTest) * 100) + "%.")
    
    print("Finished execution of initial experiment.")

def trainKNNClassifier(xTrain, yTrain, k = idealNeighbors, metric = "minkowski"):
    
    KNNClassifier = KNeighborsClassifier(n_neighbors = k, metric = metric, n_jobs = -1)
    KNNClassifier.fit(xTrain, yTrain)

    return KNNClassifier


if __name__ == "__main__":
    main()
