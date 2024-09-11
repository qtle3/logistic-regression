# Social Network Ads Classification using Logistic Regression

This project demonstrates the use of **Logistic Regression** to classify whether a user on a social network will purchase a product based on their age and estimated salary. Using a dataset that includes these features, the Logistic Regression model is trained and evaluated, with visualizations for both the training and test set results.

## Dataset

The dataset (`Social_Network_Ads.csv`) contains user information from a social network, including:
- **Age**
- **Estimated Salary**
- **Purchased (Target Variable)** - Whether the user purchased the product (1) or did not (0).

## Detailed Summary

The script performs classification using **Logistic Regression** to predict whether a user will purchase a product based on their age and salary. It involves splitting the dataset into training and test sets, scaling the features, training the model, and making predictions. The performance of the model is evaluated using a **confusion matrix** and **accuracy score**, and the results are visualized using decision boundaries.

The script performs the following steps:

1. **Data Loading:**
   - Loads the `Social_Network_Ads.csv` dataset and splits it into features (`Age`, `Estimated Salary`) and the target variable (`Purchased`).

2. **Data Splitting and Scaling:**
   - The dataset is split into **training** and **test sets** (75% training, 25% testing).
   - **StandardScaler** is used to scale the features to ensure optimal performance for the Logistic Regression model.

3. **Model Training:**
   - A **Logistic Regression** model is trained on the scaled training set to classify whether users will purchase a product.

4. **Prediction and Evaluation:**
   - The model is used to make predictions on the test set, and the results are evaluated using:
     - **Confusion Matrix**: Shows the number of correct and incorrect predictions.
     - **Accuracy Score**: Provides a percentage of correct predictions.

5. **Visualization:**
   - The decision boundaries for the training and test set results are visualized using **contour plots** to show how well the model classifies users based on their age and estimated salary.
