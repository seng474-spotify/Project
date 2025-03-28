# Data Mining - Winter 2025 
# Random Forests Implementation
# author: Austin Goertz
# student number: V00987134

#IMPORTSimport argparse
import argparse
import numpy as np
from math import sqrt
from math import floor
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import preprocessing
import random_forest_preprocessing
from error import calculate_errors
from graph import plot_errors

def parse_cl_args() -> str:
    '''
    Takes in no input, simply parses the command line arguments and returns:
    criterion(gini, informationGain, pruning_method(mcc, reduced error)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='training_size', required=False, choices=['training_size', 
                                                                                                  'forest_size', 
                                                                                                  'random_features', 
                                                                                                  'bootstrap_sample_size',
                                                                                                  'random_forest',
                                                                                                  'num_genres'])
    args = parser.parse_args()
    return args.experiment

def train_random_forest_model(X: np.ndarray, Y: np.ndarray, M=100, random_features=None, bootstrap_sample_size=None):

    # Assign Caluclated Defaults:
    if random_features == None:
        random_features = floor(sqrt(X.shape[1]))
    if bootstrap_sample_size == None:
       bootstrap_sample_size = X.shape[0]

    # Debugging print statements
    # print(f"{bootstrap_sample_size}, {random_features}")

    classifier: ensemble.RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=M,  
                                                                                  max_samples=bootstrap_sample_size, 
                                                                                  max_features=random_features,
                                                                                  criterion='gini')
    classifier = classifier.fit(X, Y)
    return classifier

def training_size_experiment() -> None:
    
    sample_sizes = [i / 10 for i in range(1, 10, 1)]
    training_errors: list[float] = []
    test_errors: list[float] = []

    for sample_size in sample_sizes:
        X_train, Y_train, X_test, Y_test = preprocessing.preprocess_data('spambase_augmented.csv', sample_size)
        model = train_random_forest_model(X_train, Y_train)
        training_error, test_error = calculate_errors(model, X_train, Y_train, X_test, Y_test)

        training_errors.append(training_error)
        test_errors.append(test_error)
    
    plot_errors(sample_sizes, training_errors, test_errors, "Effect of Training Sample Size on Error for Random Forest Error", "Sample Size %")

def random_forest_size_experiment():
    
    sample_size = 0.8
    X_train, Y_train, X_test, Y_test = preprocessing.preprocess_data('spambase_augmented.csv', sample_size)
    forest_sizes = [i for i in range(1, 201)]
    training_errors: list[float] = []
    test_errors: list[float] = []

    for forest_size in forest_sizes:
        model = train_random_forest_model(X_train, Y_train, M=forest_size)
        training_error, test_error = calculate_errors(model, X_train, Y_train, X_test, Y_test)

        training_errors.append(training_error)
        test_errors.append(test_error)
    
    plot_errors(forest_sizes, training_errors, test_errors, "Effect of Number of Trees on Error for Random Forests", "Random Forest Size")

def random_features_experiment():

    sample_size = 0.8
    X_train, Y_train, X_test, Y_test = preprocessing.preprocess_data('spambase_augmented.csv', sample_size)
    random_features_initial = floor(sqrt(X_train.shape[1]))
    print(X_train.shape[1])

    random_feature_sizes = [i for i in range(1, X_train.shape[1], 10)]
    training_errors: list[float] = []
    test_errors: list[float] = []

    for random_feature_size in random_feature_sizes:
        model = train_random_forest_model(X_train, Y_train, random_features=random_feature_size)
        training_error, test_error = calculate_errors(model, X_train, Y_train, X_test, Y_test)

        training_errors.append(training_error)
        test_errors.append(test_error)
    
    plot_errors(random_feature_sizes, training_errors, test_errors, "Effect of Number of Random Features on Error for Random Forests", "Number of Random Features")

def bootstrap_size_experiment():

    sample_size = 0.8
    X_train, Y_train, X_test, Y_test = preprocessing.preprocess_data('spambase_augmented.csv', sample_size)
    bootstrap_size_max = X_train.shape[0]

    bootstrap_sample_sizes = [i for i in range(1, bootstrap_size_max, 5)]
    training_errors: list[float] = []
    test_errors: list[float] = []

    for bootstrap_sample_size in bootstrap_sample_sizes:
        model = train_random_forest_model(X_train, Y_train, bootstrap_sample_size=bootstrap_sample_size)
        training_error, test_error = calculate_errors(model, X_train, Y_train, X_test, Y_test)

        training_errors.append(training_error)
        test_errors.append(test_error)
    
    plot_errors(bootstrap_sample_sizes, training_errors, test_errors, "Effect of Number of Bootstrap Sample Size on Error for Random Forests", "Bootstrap Sample Size")

def random_forest_experiment() -> None:

    
    X_train, X_test, Y_train, Y_test = random_forest_preprocessing.preprocess_data('spotify_data.csv') # Defaults to a test set size of 20%, and a training set of 80%

    # Tune the model using a grid search
    # Define the parameter grid
    param_grid = {
    'n_estimators': [50, 100, 200, 300],          # Number of trees
    'max_depth': [10, 20, None],             # Depth of each tree
    'min_samples_split': [2, 5, 10],         # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],           # Min samples per leaf node
    'max_features': ['sqrt', 'log2']         # Number of features to consider for splits
    }

    # Define the random forest
    random_forest_model: ensemble.RandomForestClassifier = ensemble.RandomForestClassifier()

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the grid search
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-validation Score:", grid_search.best_score_)

    # Best Parameters: Best Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
    # Test Error: 13.190059665179865%

    # Use the best model from grid search
    best_model = grid_search.best_estimator_

    # Calculate errors using the best model
    training_error, test_error = calculate_errors(best_model, X_train, Y_train, X_test, Y_test)

    # Show the results
    print("Random Forest Experiment Results:")
    print(f"Training Error: {training_error * 100}%")
    print(f"Test Error: {test_error * 100}%")

    return

def num_genres_experiment() -> None:

    genres_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    training_errors = []
    test_errors = []
    
    for num_genres in genres_list:
        print(f"Testing {num_genres} genres")
        # Get the data given the correct number of features
        X_train, Y_train, X_test, Y_test = preprocessing.preprocess_data('spotify_data.csv', 50000, num_genres) # Defaults to a test set size of 20%, and a training set of 80%
        # Create a model given the optimal tuning found in random_forest_experiment().
        model = ensemble.RandomForestClassifier(max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=200)
        model.fit(X_train, Y_train)
        
        # Calculate Errors
        training_error, test_error = calculate_errors(model, X_train, Y_train, X_test, Y_test)
        training_errors.append(training_error)
        test_errors.append(test_error)
    
    plot_errors(genres_list, training_errors, test_errors, "Effect of Number of Number of Genres used on Error for Random Forests", "Number of Genres")
    
    # Print Results Nicely
    print(f"{'Number of Genres':<20}{'Training Error':<20}{'Test Error':<20}")
    print("-" * 60)  # Separator line
    for genre, train_err, test_err in zip(genres_list, training_errors, test_errors):
        print(f"{genre:<20}{train_err:<20.4f}{test_err:<20.4f}")
    
    return


def main():
    experiment: str = parse_cl_args()
   
    if experiment == 'training_size':
        training_size_experiment()
    elif experiment == 'forest_size':
        random_forest_size_experiment()
    elif experiment == 'random_features':
        random_features_experiment()
    elif experiment == 'bootstrap_sample_size':
        bootstrap_size_experiment()
    elif experiment == 'random_forest':
        random_forest_experiment()
    elif experiment == 'num_genres':
        num_genres_experiment()
    else:
        raise ValueError(f"Invalid experiment '{experiment}'. Valid options are: 'training_size', 'forest_size', 'random_features', 'bootstrap_sample_size'.")

if __name__ == '__main__':
    main()
