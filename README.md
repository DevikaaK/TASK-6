# TASK_6

KNN Classification Analysis on the Iris Dataset

This project performs a comprehensive K-Nearest Neighbors (KNN) classification analysis using the classic Iris flower dataset. It is designed as a complete pipeline that includes data exploration, normalization, K-value experimentation, model evaluation, decision boundary visualization, and feature importance analysis.

The analysis is wrapped inside a single Python class: KNNClassificationAnalysis, with modular methods for each stage of the machine learning workflow.Key Steps:

1.Load and Explore Dataset

* Uses the built-in Iris dataset from scikit-learn

* Displays summary statistics, feature descriptions, and target class distribution

2.Data Preprocessing

* Splits the data into training and testing sets (70/30 split)

* Applies standard normalization using StandardScaler

3.K-value Optimization

* Tests values of K from 1 to 20

* Compares test set accuracy and 5-fold cross-validation accuracy

* Selects the best K based on cross-validation performance

4.Model Training

* Trains the final KNN model using the optimal K

* Predicts class labels and probabilities

5.Evaluation

* Calculates test accuracy and prints a classification report

* Displays both raw and normalized confusion matrices

* Lists misclassified samples 

6.Decision Boundary Visualization

* Plots 2D decision boundaries for selected feature pairs

7.Feature Analysis

* Shows feature distributions, correlations, and scatter plots

* Estimates feature importance using permutation technique 
