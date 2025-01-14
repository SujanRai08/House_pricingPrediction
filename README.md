### House Price Prediction Project
# Overview
The House Price Prediction project aims to estimate housing prices based on various features such as area, bedrooms, bathrooms, location, and other attributes. This project demonstrates proficiency in data preprocessing, exploratory data analysis, machine learning model training, and evaluation. It also involves hyperparameter tuning and saving the trained model for deployment purposes.

### Objective:
To build a machine learning model capable of accurately predicting house prices using structured data, implementing best practices in data preprocessing, feature scaling, and model evaluation.

*Key Technologies and Tools:*

Languages: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Machine Learning Models: Linear Regression, Random Forest Regressor
Techniques: Data encoding, feature scaling, hyperparameter tuning
Deployment: Pickle for saving the trained model

Project Workflow
Data Loading and Exploration:

Used pandas to load and inspect the dataset (Housing.csv) to understand its structure.
Why: Understanding the data is essential to identify missing values, outliers, and the relationship between features.
Exploratory Data Analysis (EDA):

Visualized the distribution of house prices using seaborn's histogram plot.
Why: To analyze price trends and detect any skewness in the data, aiding in model selection and preprocessing decisions.
Data Preprocessing:

Applied pd.get_dummies() to encode categorical variables into numerical format.
Why: Machine learning models require numerical inputs; encoding ensures categorical data is represented in a format the model can interpret.
Feature Scaling:

Standardized features using StandardScaler to ensure all variables contribute equally to the model.
Why: Scaling improves the performance and convergence speed of gradient-based models like Linear Regression.
Data Splitting:

Split the dataset into training and testing sets using train_test_split.
Why: To evaluate model performance on unseen data and prevent overfitting.
Model Training and Evaluation:

Trained a Linear Regression model and evaluated its performance using the Root Mean Squared Error (RMSE) metric.
Trained a Random Forest Regressor and compared its RMSE to determine which model performed better.
Why: Multiple models help determine the most suitable one for the dataset. RMSE was used as it penalizes large errors, making it ideal for regression problems.
Hyperparameter Tuning:

Used GridSearchCV to optimize the Random Forest model’s parameters (n_estimators and max_depth).
Why: Hyperparameter tuning improves model accuracy and prevents overfitting.
Model Saving:

Saved the trained Random Forest model using pickle.
Why: To reuse the model without retraining, enabling deployment in real-world applications.
