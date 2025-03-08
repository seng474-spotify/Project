#T. Pires - knn.py - K-NN python script for SENG474 project.
#Created 2025-03-08
#Last Updated 2025-03-08

#Dependencies
import kagglehub #pip install kagglehub
import pandas as pd #pip install pandas

#Control Switches
debug = 0 #Switch for debugging

def main():

    print("\nStarted execution of knn.py...\n")

    #https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks
    #Ensure the latest version of the dataset is downloaded.
    path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks") #If this is failing for some reason, look in :
                                                                               #C:\Users\(insert user name here)\.cache\kagglehub\datasets\amitanshjoshi\spotify-1million-tracks\versions\1
    
    #Ensure path points to the CSV file instead of the directory
    path += r"\spotify_data.csv"

    if debug == 1:
        print("Path to dataset files:", path)
        
    print("Dataset updated.")

    data = pd.read_csv(path)
    
    #Split data into feature matrix and target/label vector
    x = data.drop(columns=["Unnamed: 0", "artist_name", "track_name", "track_id", "genre"]) #"Unnamed: 0" is the enumeration column
    y = data["genre"]
    
    if debug == 1:
        print(x.columns.tolist())
        print(y)

    print("Dataset split into feature matrix and target/label vector.")
    
    print("\nFinished execution of knn.py.")

if __name__ == "__main__":
    main()