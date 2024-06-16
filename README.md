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

## Data Interpretation

### Property Prices

![Imgur](https://i.imgur.com/mHVQhNB.png)

- **Bar Chart**: Indicates higher average property prices in Kuala Lumpur (mean value: RM1900) compared to Selangor (mean value: RM1350).
  - **Kuala Lumpur**: Higher prices due to greater employment opportunities, better transportation networks, lifestyle attractions, higher population density, and urbanization.
  - **Selangor**: Lower prices attributed to larger geographic size and more competitive rental market.

### Property Types

![Imgur](https://i.imgur.com/8VXpm6X.png)

- **Stacked Bar Chart**: Condominiums are the most common property type in both Kuala Lumpur and Selangor.
  - **Kuala Lumpur**: Service residences are the second most prevalent, indicating demand for luxury and convenience.
  - **Selangor**: Apartments are the second most prevalent, reflecting affordability and suitability for individuals or small families.

### Property Size Distribution

![Imgur](https://i.imgur.com/ewjkCvV.png)

- **Distribution Graph**: Selangor properties tend to be smaller, while Kuala Lumpur properties tend to be larger.
  - **Kuala Lumpur**: Attracts higher-income individuals or businesses needing larger spaces.
  - **Selangor**: More diverse economic landscape leading to smaller, more affordable properties.

### Model Performance

![Imgur](https://i.imgur.com/3qsRl1i.png)

- **Scatter Plot (Actual vs Predicted Rent)**: 
  - The Random Forest model shows good performance with points close to the ideal prediction line.
  - Better accuracy for lower rent values, with higher variability for higher rent values, indicating a limitation in predicting higher rents.

### Feature Importance

![Imgur](https://i.imgur.com/zfclMde.png)

- **Top 10 Features**:
  - **Size**: Most significant influence on rent (importance value: 0.246770).
  - **Furnishing**: Second most important, indicating high value on convenience.
  - **Bathrooms and Rooms**: Higher rents for properties with more bathrooms and rooms.
  - **Parking**: Significant impact on rent, especially in urban areas.
  - **Location and Property Type**: Specific locations like "Mont Kiara" and property types like "Service Residence" associated with higher rents.

### Model Residuals

![Imgur](https://i.imgur.com/YDS1phl.png)

- **Residual Histogram**: 
  - Most residuals are normally distributed around 0, indicating unbiased and accurate predictions by the Random Forest model.

![Imgur](https://i.imgur.com/doY1LzQ.png)

- **Residual Scatter Plot**: 
  - Uniform distribution around y=0, suggesting consistent accuracy.
  - Wider spread for higher predicted values indicates heteroscedasticity (variability of errors).

### Categorical Feature Encoding

- **One-Hot Encoding**:
  - Utilized for categorical features such as location, region, and furnishing, resulting in high-dimensional data.
  - The Random Forest model effectively handles this high-dimensional dataset.

### Decision Tree Visualization

![Imgur](https://i.imgur.com/wdT0hbp.png)

- **Tree Structure**: 
  - Visualizes how the model partitions the feature space based on different conditions, capturing complex relationships.
  - Helps manage computational complexity while retaining relevant information.
