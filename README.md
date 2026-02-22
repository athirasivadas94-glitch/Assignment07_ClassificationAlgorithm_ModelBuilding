# Assignment07_ClassificationAlgorithm_ModelBuilding
Classification Algorithm_Model Building
>>Project Overview

This project focuses on applying multiple supervised learning classification algorithms to the Breast Cancer dataset available in the sklearn library. The objective is to build, compare, and evaluate different machine learning models to accurately classify tumors.
This assessment demonstrates data preprocessing, model implementation, and performance comparison using standard evaluation metrics. Dataset

>>Dataset: Breast Cancer Wisconsin Dataset (from sklearn)

Total Samples: 569

Features: 30 numerical features

The dataset contains medical measurements such as radius, texture, perimeter, area, and smoothness of tumors.

>> Project Workflow
1. Loading and Preprocessing

Loaded the dataset using sklearn.datasets

Converted the dataset into a Pandas DataFrame

Checked for missing values (no missing values found)

Performed feature scaling using StandardScaler

Why Preprocessing is Necessary

Features have different ranges (e.g., area vs smoothness)

Scaling improves model performance and convergence

Essential for distance-based and gradient-based algorithms like SVM, KNN, and Logistic Regression

>>Classification Algorithms Implemented

The following five supervised learning algorithms were implemented:

1. Logistic Regression

A linear model that predicts class probabilities using a sigmoid function.

Suitable because the dataset is binary and relatively well-separated.

2. Decision Tree Classifier

A tree-based model that splits data based on feature importance.

Easy to interpret and handles feature interactions well.

3. Random Forest Classifier

An ensemble of multiple decision trees.

Reduces overfitting and improves prediction accuracy.

4. Support Vector Machine (SVM)

Finds the optimal hyperplane that maximizes the margin between classes.

Performs well on high-dimensional datasets like this one.

5. k-Nearest Neighbors (k-NN)

Classifies data based on the majority class of nearest neighbors.

Effective for structured and scaled datasets.

>> Model Performance Comparison
Algorithm	Accuracy
Logistic Regression	0.9737
Decision Tree	0.9474
Random Forest	0.9649
SVM	0.9561
k-NN	0.9474

>>Best and Worst Performing Models

Best Model: Logistic Regression (97.37% accuracy)

Indicates the dataset is fairly linearly separable.

Worst Models: Decision Tree and k-NN (94.74% accuracy)

Decision Tree may overfit.

k-NN is sensitive to distance and parameter selection.

>> Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (for visualization)

>> Conclusion

All models performed well with accuracy above 94%, showing that the dataset is well-suited for classification tasks. Logistic Regression achieved the highest accuracy, making it the most effective model for this dataset. Ensemble methods like Random Forest and margin-based models like SVM also showed strong performance.

This project highlights the importance of preprocessing, model selection, and evaluation metrics in building reliable machine learning models for real-world healthcare applications.
