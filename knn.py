#T. Pires - knn.py - K-NN python script for SENG474 project.
#Created 2025-03-08
#Last Updated 2025-03-08

#Dependencies
import kagglehub #pip install kagglehub

#Control Switches
debug = 1 #Switch for debugging

def main():

    #https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks
    #Ensure the latest version of the dataset is downloaded.
    path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks") #If this is failing for some reason, look in :
                                                                               #C:\Users\(insert user name here)\.cache\kagglehub\datasets\amitanshjoshi\spotify-1million-tracks\versions\1
    if debug == 1:
        print("Path to dataset files:", path)

    print("Hello world")
    
if __name__ == "__main__":
    main()