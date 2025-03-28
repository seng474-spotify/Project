'''
Data Mining - SENG 474, Winter 2025
graph.py
Author: Austin Goertz, V00987134
'''

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import datetime
import numpy as np

def plot_errors(x_axis, training_errors, test_errors, title, x_label):
    np.random.seed(42)

    """Plot errors and save the figure"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, training_errors, label="Training Error", marker='o')
    plt.plot(x_axis, test_errors, label="Test Error", marker='o')
    plt.xlabel(x_label)
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Graphs/{title} {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()

def plot_errors_exp_scale(x_axis, training_errors, test_errors, title, x_label):
    np.random.seed(42)

    """Plot errors with an exponential x-axis scale and save the figure."""
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, training_errors, label="Training Error", marker='o')
    plt.plot(x_axis, test_errors, label="Test Error", marker='o')
    
    # Set x-axis to logarithmic scale
    plt.xscale('log')

    # Adjust the ticks to avoid the plot being too compact
    # Automatically adjust ticks for the logarithmic scale
    # plt.xticks(np.logspace(np.log10(min(x_axis)), np.log10(max(x_axis)), num=10))
    
    plt.xlabel(x_label)
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    # Save the figure
    plt.savefig(f'Graphs/{title} {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Show the plot
    plt.show()

def display_decision_tree(model, title: str = "Decision Tree Visualization"):
    """
    Plots the decision tree model using scikit-learn's plot_tree and displays it.
    
    Parameters:
        model: A fitted decision tree model from scikit-learn (e.g., DecisionTreeClassifier or DecisionTreeRegressor).
        title (str): The title to display on the plot. Default is "Decision Tree Visualization".
    """
    # Plot the tree
    plt.figure(figsize=(12, 8))
    plot_tree(model)
    
    # Set the title
    plt.title(title)

    # Save the figure
    plt.savefig(f'Photos/{title} {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Show the plot
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_bargraph(labels, train_errors, val_errors, title, x_label):
    # Set the bar width and position
    bar_width = 0.35
    index = np.arange(len(labels))
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars for train and validation errors
    bars1 = ax.bar(index - bar_width/2, train_errors, bar_width, label='Train Error', color='skyblue', edgecolor='black')
    bars2 = ax.bar(index + bar_width/2, val_errors, bar_width, label='Validation Error', color='lightcoral', edgecolor='black')

    # Add labels to each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset the label slightly above the bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    # Add title and axis labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    
    # Add x-ticks, set the position, and rotate for clarity
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add legend
    ax.legend()

    # Add gridlines for easier reading of the values
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'Graphs/{title} {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Show the plot
    plt.show()

