# House Price Prediction

## Project Description

This project aims to create a predictive model that can accurately forecast rental prices of residential properties in Kuala Lumpur and Selangor. This model can be used by property owners, investors, and potential renters seeking reliable and precise rental price estimates. Renters who have trouble finding suitable rental properties and landlords who want to optimize their pricing strategy and attract potential renters can benefit from this model.

## Objectives

1. **Identify the main contributing factors that affect house rental prices in Kuala Lumpur.**
2. **Determine the relationship between the contributing factors and the house rental prices.**
3. **Train a model that allows the user to predict the house rental price based on their inputs using predictive modeling.**

## Data Modeling

### Importing Model and Dependencies

We import the necessary libraries: Pandas, NumPy, Matplotlib, and Seaborn. Additionally, we configure Pandas to display a maximum width of 200 characters per column to handle long entries effectively.

### Extracting Input and Output

This function separates the input data and output data from the dataset, returning the input data as a DataFrame and the output data as a Series. This preparation is crucial for training machine learning models where the input data trains the models, and the output data represents the target variable.

### Data Standardization

We standardize the data using `StandardScaler` from scikit-learn. This function returns a standardized DataFrame and a scaler object to transform new data consistently. Standardization normalizes the scale of features, aiding model performance.

### Splitting Train-Test Data

To evaluate model performance, we split the data into training and testing sets. Using an 80/20 split and a random state of 42 ensures reproducibility. The training set (`X_train`, `y_train`) is used to train the model, while the testing set (`X_test`, `y_test`) evaluates its performance.

### Training Machine Learning Models

We use regression models: linear regression, gradient boosting, and random forest. We evaluate model performance using R² score and Mean Absolute Error (MAE).

#### Baseline Average Value

Using the average value of the target variable as a baseline prediction, we establish a benchmark. The baseline's R² score and MAE help us compare the performance of more complex models.

#### Linear Regression Model

Linear regression assumes a linear relationship between features and the target variable. The model explains approximately 62.36% of the variance in monthly rent, with an MAE of RM298.47.

#### Gradient Boosting Model

Gradient boosting combines multiple weak prediction models to form a strong predictive model. It explains approximately 69.91% of the variance in monthly rent, with an MAE of RM267.04. Using grid search, we optimized hyperparameters, achieving an R² score of 78.50% and an MAE of RM224.75.

#### Random Forest Model

Random forest uses an ensemble of decision trees to improve accuracy. Initially, it explained 94.89% of the variance with an MAE of RM102.44. After hyperparameter optimization via grid search, it achieved an R² score of 95.00% and an MAE of RM101.69.

### Summary of Model Performance

A summary table shows the R² score and MAE for each model:

- **Random Forest**: Best performance with the lowest MAE and highest R² score.
- **Gradient Boosting**: Moderate performance.
- **Linear Regression**: Basic performance.
- **Baseline Model**: Poorest performance.

### Pre-Processing Test Data

The same preprocessing steps applied to the training data are applied to the test data. The random forest model, when tested, showed an R² score of 71.9% and an MAE of RM247.75, indicating good performance on unseen data. The results suggest the model fits well to the training set but has slightly lower performance on the test set.
