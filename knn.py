#T. Pires - knn.py - K-NN python script for SENG474 project.
#Created 2025-03-08
#Last Updated 2025-03-26

#Dependencies
import kagglehub #pip install kagglehub
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np #pip install numpy
import pandas as pd #pip install pandas
from sklearn.cluster import AgglomerativeClustering #pip install scikit-learn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Control Switches
debug = 0 #Switch for debugging

#Experiment Switches
initialExperimentSwitch = 0 #Initialization
tuneNeighboursSwitch = 0 #n_neighbors
tuneWeightsSwitch = 0 #weights
tuneAlgorithmSwitch = 0 #algorithm - leaf-size just affects the speed at which the construction takes place.
tuneDissimilarityMeasureSwitch = 0 #metric (dissimilarity measure) - p affects Minkowski only...
tunedKNNSwitch = 0 #Using tuned hyperparameters

testSwitch = 1

#Ideal Hyperparameter Values (calculated from experiments)
idealNeighbors = 40 
idealWeighting = "uniform"
idealAlgorithm = "auto"
idealMetric = "manhattan"

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

    if tuneWeightsSwitch == 1:
        tuneWeights(data)

    if tuneAlgorithmSwitch == 1:
        tuneAlgorithm(data)

    if tunedKNNSwitch == 1:
        tunedKNNExperiment(data)

    if testSwitch == 1:

        #Filter to include only the 8 highest occurring genres - classification granularity metric 
        topEightGenres = data["genre"].value_counts().index[:8] #black-metal, gospel, ambient, acoustic, alt-rock, emo, indian, k-pop
        data = data[data["genre"].isin(topEightGenres)]
        print("Reduced dataset to only include highest 8 occurring genres.")

        blackMetalData = data[data["genre"].isin(["black-metal"])]
        gospelData = data[data["genre"].isin(["gospel"])]
        ambientData = data[data["genre"].isin(["ambient"])]
        acousticData = data[data["genre"].isin(["acoustic"])]
        altRockData = data[data["genre"].isin(["alt-rock"])]
        emoData = data[data["genre"].isin(["emo"])]
        indianData = data[data["genre"].isin(["indian"])]
        KPopData = data[data["genre"].isin(["k-pop"])]
        print("Separated dataset into instances of each label in top 8 genres.")

        if debug == 1:
            print(blackMetalData)
            print(gospelData)
            print(ambientData)
            print(acousticData)
            print(altRockData)
            print(emoData)
            print(indianData)
            print(KPopData)

        blackMetalCenter = calculateClusterCenter(blackMetalData)
        gospelCenter = calculateClusterCenter(gospelData)
        ambientCenter = calculateClusterCenter(ambientData)
        acousticCenter = calculateClusterCenter(acousticData)
        altRockCenter = calculateClusterCenter(altRockData)
        emoCenter = calculateClusterCenter(emoData)
        indianCenter = calculateClusterCenter(indianData)
        KPopCenter = calculateClusterCenter(KPopData)
        print("Calculated cluster centers.") #(using standard scaler).")

        if debug == 1:
            print(blackMetalCenter)
            print(gospelCenter)
            print(ambientCenter)
            print(acousticCenter)
            print(altRockCenter)
            print(emoCenter)
            print(indianCenter)
            print(KPopCenter)

        #Group 1 kpop/altrock
        group1Data = pd.concat([KPopData, altRockData])
        
        xGroup1 = group1Data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"])
        yGroup1 = group1Data["genre"]
        
        scaler = StandardScaler()
        xGroup1Normalized = scaler.fit_transform(xGroup1.to_numpy())  #Convert DataFrame to NumPy array

        #Convert the normalized NumPy array back to a DataFrame
        xGroup1Normalized = pd.DataFrame(xGroup1Normalized, columns = xGroup1.columns)
        
        xGroup1Train, xGroup1Test, yGroup1Train, yGroup1Test = train_test_split(xGroup1Normalized, yGroup1, test_size = 0.2)
        
        xGroup1TrainCopy = xGroup1Train.copy()
        xGroup1TestCopy = xGroup1Test.copy()
        yGroup1TrainCopy = yGroup1Train.copy()
        yGroup1TestCopy = yGroup1Test.copy()

        yGroup1TrainCopy[:] = "group1"
        yGroup1TestCopy[:] = "group1"

        #Group 2 - acoustic/emo
        group2Data = pd.concat([acousticData, emoData])
        
        xGroup2 = group2Data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"])
        yGroup2 = group2Data["genre"]
        
        scaler = StandardScaler()
        xGroup2Normalized = scaler.fit_transform(xGroup2.to_numpy())  #Convert DataFrame to NumPy array

        #Convert the normalized NumPy array back to a DataFrame
        xGroup2Normalized = pd.DataFrame(xGroup2Normalized, columns = xGroup2.columns)
        
        xGroup2Train, xGroup2Test, yGroup2Train, yGroup2Test = train_test_split(xGroup2Normalized, yGroup2, test_size = 0.2)
        
        xGroup2TrainCopy = xGroup2Train.copy()
        xGroup2TestCopy = xGroup2Test.copy()
        yGroup2TrainCopy = yGroup2Train.copy()
        yGroup2TestCopy = yGroup2Test.copy()

        yGroup2TrainCopy[:] = "group2"
        yGroup2TestCopy[:] = "group2"

        #Group 3 - gospel/ambient
        group3Data = pd.concat([gospelData, gospelData])
        
        xGroup3 = group3Data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"])
        yGroup3 = group3Data["genre"]
        
        scaler = StandardScaler()
        xGroup3Normalized = scaler.fit_transform(xGroup3.to_numpy())  #Convert DataFrame to NumPy array

        #Convert the normalized NumPy array back to a DataFrame
        xGroup3Normalized = pd.DataFrame(xGroup3Normalized, columns = xGroup3.columns)
        
        xGroup3Train, xGroup3Test, yGroup3Train, yGroup3Test = train_test_split(xGroup3Normalized, yGroup3, test_size = 0.2)
        
        xGroup3TrainCopy = xGroup3Train.copy()
        xGroup3TestCopy = xGroup3Test.copy()
        yGroup3TrainCopy = yGroup3Train.copy()
        yGroup3TestCopy = yGroup3Test.copy()

        yGroup3TrainCopy[:] = "group3"
        yGroup3TestCopy[:] = "group3"

        #Group 4 - blackmetal/indian
        group4Data = pd.concat([blackMetalData, indianData])
        
        xGroup4 = group4Data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"])
        yGroup4 = group4Data["genre"]
        
        scaler = StandardScaler()
        xGroup4Normalized = scaler.fit_transform(xGroup4.to_numpy())  #Convert DataFrame to NumPy array

        #Convert the normalized NumPy array back to a DataFrame
        xGroup4Normalized = pd.DataFrame(xGroup4Normalized, columns = xGroup4.columns)
        
        xGroup4Train, xGroup4Test, yGroup4Train, yGroup4Test = train_test_split(xGroup4Normalized, yGroup4, test_size = 0.2)
        
        xGroup4TrainCopy = xGroup4Train.copy()
        xGroup4TestCopy = xGroup4Test.copy()
        yGroup4TrainCopy = yGroup4Train.copy()
        yGroup4TestCopy = yGroup4Test.copy()

        yGroup4TrainCopy[:] = "group4"
        yGroup4TestCopy[:] = "group4"

        print("Datasets created for each grouping.")

        yTestCombined = pd.concat([yGroup1Test, yGroup2Test, yGroup3Test, yGroup4Test])

        xTrainCopyCombined = pd.concat([xGroup1TrainCopy, xGroup2TrainCopy, xGroup3TrainCopy, xGroup4TrainCopy])
        yTrainCopyCombined = pd.concat([yGroup1TrainCopy, yGroup2TrainCopy, yGroup3TrainCopy, yGroup4TrainCopy])
        
        xTestCopyCombined = pd.concat([xGroup1TestCopy, xGroup2TestCopy, xGroup3TestCopy, xGroup4TestCopy])
        yTestCopyCombined = pd.concat([yGroup1TestCopy, yGroup2TestCopy, yGroup3TestCopy, yGroup4TestCopy])

        KNNTopLevelClassifier = trainKNNClassifier(xTrain = xTrainCopyCombined, yTrain = yTrainCopyCombined)
        print("Trained top-level KNN Classifier (groups).")
        
        topLevelPredictions = KNNTopLevelClassifier.predict(xTestCopyCombined)
        group1Indices = np.where(topLevelPredictions == "group1")[0].tolist()
        group2Indices = np.where(topLevelPredictions == "group2")[0].tolist()
        group3Indices = np.where(topLevelPredictions == "group3")[0].tolist()
        group4Indices = np.where(topLevelPredictions == "group4")[0].tolist()

        print("Finished classification into groups.")

        KNNGroup1Classifier = trainKNNClassifier(xTrain = xGroup1Train, yTrain = yGroup1Train)
        print("Trained bottom-level KNN Classifier for group 1.")
        xGroup1Data = xTestCopyCombined.iloc[group1Indices]
        yGroup1Data = yTestCombined.iloc[group1Indices]
        group1DataPredictions = KNNGroup1Classifier.predict(xGroup1Data)
        
        KNNGroup2Classifier = trainKNNClassifier(xTrain = xGroup2Train, yTrain = yGroup2Train)
        print("Trained bottom-level KNN Classifier for group 2.")
        xGroup2Data = xTestCopyCombined.iloc[group2Indices]
        yGroup2Data = yTestCombined.iloc[group2Indices]
        group2DataPredictions = KNNGroup2Classifier.predict(xGroup2Data)        

        KNNGroup3Classifier = trainKNNClassifier(xTrain = xGroup3Train, yTrain = yGroup3Train)
        print("Trained bottom-level KNN Classifier for group 3.")
        xGroup3Data = xTestCopyCombined.iloc[group3Indices]
        yGroup3Data = yTestCombined.iloc[group3Indices]
        group3DataPredictions = KNNGroup3Classifier.predict(xGroup3Data)        

        KNNGroup4Classifier = trainKNNClassifier(xTrain = xGroup3Train, yTrain = yGroup3Train)
        print("Trained bottom-level KNN Classifier for group 4.")
        xGroup4Data = xTestCopyCombined.iloc[group4Indices]
        yGroup4Data = yTestCombined.iloc[group4Indices]
        group4DataPredictions = KNNGroup1Classifier.predict(xGroup4Data)

        finalPredictions = np.concatenate((group1DataPredictions, group2DataPredictions, group3DataPredictions, group4DataPredictions))
        finalPredictions = pd.DataFrame(finalPredictions)
        orderedLabels = pd.concat([yGroup1Data, yGroup2Data, yGroup3Data, yGroup4Data])
        print("Final testing score: " + str(accuracy_score(finalPredictions, orderedLabels) * 100) + "%.")

    print("\nFinished execution of knn.py.")

def calculateClusterCenter(data):

    x = data.drop(columns = ["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"]) #Remove non-features. "Unnamed: 0" is the enumeration column. 

    clusterCenter = x.mean()
    
    return clusterCenter

def tunedKNNExperiment(data):
    print("\nRunning experiment on tuned KNN Classifier...")

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
    xNormalized = scaler.fit_transform(x.to_numpy())  #Convert DataFrame to NumPy array

    #Convert the normalized NumPy array back to a DataFrame
    xNormalized = pd.DataFrame(xNormalized, columns = x.columns)

    if debug == 1:
        print(xNormalized)

    print("Dataset features normalized.")

    xTrain, xTest, yTrain, yTest = train_test_split(xNormalized, y, test_size = 0.2) #Tested on random_state = 8
    print("Training and testing sets created from dataset.")
    
    KNNClassifier = trainKNNClassifier(xTrain = xTrain, yTrain = yTrain)
    print("Trained KNN Classifier.")

    print("Training accuracy: " + str(KNNClassifier.score(xTrain, yTrain) * 100) + "%.")
    print("Testing accuracy: " + str(KNNClassifier.score(xTest, yTest) * 100) + "%.")

    print("Finished execution of experiment on tuned KNN Classifier.")

def tuneAlgorithm(data): #All algorithms yielded the same training & testing accuracies; keeping auto.
    print("\nRunning \"tune algorithm\" experiment...")

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

    algorithmsGrid = {"algorithms": ["auto", "ball_tree", "kd_tree", "brute"]}
    
    trainAccuracies = []
    testAccuracies = []
    algorithms = algorithmsGrid["algorithms"]
    
    print("Performing grid search...")

    for algorithm in algorithms:
        KNNClassifier = trainKNNClassifier(xTrain = xTrain, yTrain = yTrain, algorithm = algorithm)
        trainingAccuracy = KNNClassifier.score(xTrain, yTrain) * 100
        testingAccuracy = KNNClassifier.score(xTest, yTest) * 100
        trainAccuracies.append(trainingAccuracy)
        testAccuracies.append(testingAccuracy)
    
    print("Train accuracies:", trainAccuracies)
    print("Test accuracies:", testAccuracies)
    
    #Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(algorithms, trainAccuracies, label = "Training Accuracy")
    plt.plot(algorithms, testAccuracies, label = "Testing Accuracy")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.title("Algorithm vs. Training and Testing Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Finished execution of \"tune algorithm\" experiment.")

def tuneWeights(data): #Weighting only changes the training accuracy; overfits using distance.
    print("\nRunning \"tune weights\" experiment...")

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
    xNormalized = scaler.fit_transform(x.to_numpy())  #Convert DataFrame to NumPy array

    #Convert the normalized NumPy array back to a DataFrame
    xNormalized = pd.DataFrame(xNormalized, columns = x.columns)

    if debug == 1:
        print(xNormalized)

    print("Dataset features normalized.")

    xTrain, xTest, yTrain, yTest = train_test_split(xNormalized, y, test_size = 0.2) #Tested on random_state = 8
    print("Training and testing sets created from dataset.")

    weightsGrid = {"weights": ["uniform", "distance"]}
    
    trainAccuracies = []
    testAccuracies = []
    weights = weightsGrid["weights"]
    
    print("Performing grid search...")

    for weight in weights:
        KNNClassifier = trainKNNClassifier(xTrain = xTrain, yTrain = yTrain, weighting = weight)
        trainingAccuracy = KNNClassifier.score(xTrain, yTrain) * 100
        testingAccuracy = KNNClassifier.score(xTest, yTest) * 100
        trainAccuracies.append(trainingAccuracy)
        testAccuracies.append(testingAccuracy)
    
    print("Train accuracies:", trainAccuracies)
    print("Test accuracies:", testAccuracies)
    
    #Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(weights, trainAccuracies, label = "Training Accuracy")
    plt.plot(weights, testAccuracies, label = "Testing Accuracy")
    plt.xlabel("Weighting")
    plt.ylabel("Accuracy")
    plt.title("Weighting vs. Training and Testing Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Finished execution of \"tune weights\" experiment.")

def tuneDissimilarityMeasure(data):
    print("\nRunning \"tune dissimilarity measure\" experiment...")

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
    xNormalized = scaler.fit_transform(x.to_numpy())  #Convert DataFrame to NumPy array

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
    xNormalized = scaler.fit_transform(x.to_numpy())  #Convert DataFrame to NumPy array

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
    xNormalized = scaler.fit_transform(x.to_numpy())  #Convert DataFrame to NumPy array

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

def trainKNNClassifier(xTrain, yTrain, k = idealNeighbors, weighting = idealWeighting, algorithm = idealAlgorithm, metric = idealMetric):
    
    KNNClassifier = KNeighborsClassifier(n_neighbors = k, weights = weighting, algorithm = algorithm, metric = metric, n_jobs = -1)
    KNNClassifier.fit(xTrain, yTrain)

    return KNNClassifier


if __name__ == "__main__":
    main()
