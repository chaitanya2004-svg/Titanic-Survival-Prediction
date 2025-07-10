🚢 Titanic Survival Prediction
📄 Project Overview
This project builds an end-to-end machine learning pipeline to predict whether a passenger survived the Titanic disaster based on their demographic and travel information.
We use the famous Titanic dataset and demonstrate the full workflow: data cleaning, visualization, feature engineering, model training, and evaluation.

🎯 Objectives
Understand and explore the Titanic dataset.

Clean the data and handle missing values.

Encode categorical variables and scale numerical ones.

Train a Random Forest Classifier to predict survival.

Evaluate the model using various metrics.

Identify important features influencing survival.

🗃 Dataset
Source: Kaggle Titanic Dataset

Features used:

Pclass - Passenger Class (1, 2, 3)

Sex - Gender

Age - Age in years

SibSp - Number of siblings/spouses aboard

Parch - Number of parents/children aboard

Fare - Ticket fare

Embarked - Port of Embarkation (C, Q, S)

Target:

Survived (1 = Yes, 0 = No)

🔷 Technologies & Libraries
Programming Language: Python

Libraries:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

📋 Steps Performed
✅ Data Loading: Loaded dataset directly from a public URL.
✅ Exploratory Data Analysis (EDA): Visualized survival distribution & feature correlations.
✅ Data Cleaning: Filled missing values and removed irrelevant columns.
✅ Feature Engineering: Encoded categorical variables & created scaled feature set.
✅ Model Training: Used Random Forest Classifier.
✅ Model Evaluation: Measured performance with accuracy, confusion matrix, and classification report.
✅ Feature Importance: Identified key features contributing to predictions.

📈 Results
The model performed well on the test set and highlighted that gender, passenger class, and fare were among the most important predictors of survival.
