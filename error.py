'''
Data Mining - SENG 474, Winter 2025
error.py - Caclculates a model's training a test scores
Author: Austin Goertz, V00987134
'''

def calculate_errors(model, X_train, Y_train, X_test, Y_test):
    """Calculate training and test errors"""
    training_accuracy = model.score(X_train, Y_train)
    test_accuracy = model.score(X_test, Y_test)
    return 1 - training_accuracy, 1 - test_accuracy