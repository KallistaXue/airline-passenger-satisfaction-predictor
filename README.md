# Airline-Passenger-Satisfaction-Predictor 
## Overview
This is a final project for course COMS 4995 Applied Machine Learning in Columbia University finished together with my classmates Tanisha Aggrawal and Vishal Bhardwaj. This repository contains a Jupyter notebook that predicts airline passenger satisfaction based on 22 features such as age, type of travel, class, and more. The model uses Logistic Regression, Random Forest, XGBoost and SVM to make predictions.

## Course Project Description 
The project assignment gives students an opportunity to apply different aspects of Machine Learning covered in the classroom to a real-world application. Through this project, students will get hands-on experience solving a Machine Learning problem including data analysis, visualization and applying machine learning models to develop actionable insights. 

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dataset](#dataset)
3. [Model](#model)
4. Results
5. Contributing
6. License

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.9
- Jupyter Notebook

### Libraries Used
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- time (built-in Python library, not included in 'requirements.txt')

### Installation Steps
1. Clone this repository

```
git clone https://github.com/KallistaXue/airline-passenger-satisfaction-predictor.git
```

2. Navigate into the repository
```
cd airline-passenger-satisfaction-predictor
```

3. Install Jupyter
```
pip install jupyter
```

4. Install required libraries
```
pip install -r requirements.txt
```
5. Run Jupyter Notebook
```
jupyter notebook
```

## Dataset
 ['airline-passenger-satisfaction']: (https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

In this project, we are using ['airline-passenger-satisfaction'] dataset to train the model.

The [development set](train_and_validation.csv) consists of 103904 samples, and the [testing set](test.csv) consists of 25976 samples

### Features
Both datasets consist of 22 features used to predict the target variable.

| Feature                          | Description                                       |
|----------------------------------|---------------------------------------------------|
| Gender                           | Gender of the passengers (Female, Male)           |
| Customer Type                    | The customer type (Loyal customer, disloyal customer)|
| Age                              | The actual age of the passengers                  |
| Type of Travel                   | Purpose of the flight of the passengers (Personal Travel, Business Travel)|
| Class                            | Travel class in the plane of the passengers (Business, Eco, Eco Plus)|
| Flight Distance                  | The flight distance of this journey               |
| Inflight Wifi Service            | Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)|
| Departure/Arrival Time Convenient| Satisfaction level of Departure/Arrival time convenient|
| Ease of Online Booking           | Satisfaction level of online booking              |
| Gate Location                    | Satisfaction level of Gate location               |
| Food and Drink                   | Satisfaction level of Food and drink              |
| Online Boarding                  | Satisfaction level of online boarding             |
| Seat Comfort                     | Satisfaction level of Seat comfort                |
| Inflight Entertainment           | Satisfaction level of inflight entertainment      |
| On-board Service                 | Satisfaction level of On-board service            |
| Leg Room Service                 | Satisfaction level of Leg room service            |
| Baggage Handling                 | Satisfaction level of baggage handling            |
| Check-in Service                 | Satisfaction level of Check-in service            |
| Inflight Service                 | Satisfaction level of inflight service            |
| Cleanliness                      | Satisfaction level of Cleanliness                 |
| Departure Delay in Minutes       | Minutes delayed when departure                    |
| Arrival Delay in Minutes         | Minutes delayed when Arrival                      |

### Target Variable
| Variable name       |  Description    |
|---------------------|-----------------|
|Satisfaction         |Indicates whether the passenger is Satisfied or Dissatisfied with the service|

### Data Preprocessing 
#### Missing Value Analysis
We start by analyzing the missing values in the data and remove the information about the passenger (the whole row) if part of their data is missing. 310 samples have been removed from the dev set and 83 samples have been removed from the test set.

#### Distribution of the Target Variable 
Later we observe the distribution of the target variable in the dataset, revealing a slightly imbalanced dataset with ‘satisfied’: ‘neutral or dissatisfied’ = 0.43: 0.57. 

#### Drop Irrelevant Columns
Next, we drop the columns 'id' and 'satisfaction' for both the development and test set because these information are irrelevant to the target output.

#### Encode Categorical Features
Then we encode the 4 categorical features with String values ('Gender', 'Customer Type', 'Type of Travel', 'Class') using Ordinal Encoding. 

#### Scale the Dataset
Finally, we scale the data using StandardScaler().

### Exploratory Data Analysis
[Final Report]:(Final%20Report_Group%202.pdf)
[Data Analysis and Visualization slides]:(Data%20Analysis%20and%20Visualization.pptx)

EDA is performed in two different directions - the features which are categorical and the features that are continuous. The analysis of the categorical features is distributed into the binary categorical features and the features containing more than 2 ordinal categorical features. For more information about our insights on the EDA please see our [Data Analysis and Visualization slides] or [Final Report].

## Model
For modeling customer satisfaction, we tried multiple classification techniques. For all the techniques we are using the processed scaled data and comparing the model performance on the test data using multiple classification metrics. To tune the hyperparameters we are using k-fold cross-validation and using GridSearchCV for performing the analysis.

### Logistic Regression
- param_grid = {'C': [0.1, 1.0, 10.0, 100.0, 1000.0], 'penalty': ['12']}
- Best hyperparameters: {'C': 100.0, 'penalty': '12'}
- Best cross-validation score: 0.840
- Test accuracy of the best model: 0.819

### Random Forest
- param_grid = {'max_features': ['sqrt', 'log2', 0.3, 0.6, 0.9],'n_estimators': [30, 60, 90, 120, 150]}
- Best hyperparameters: {'max_features': 0.3, 'n_estimators': 150}
- Best cross-validation score: 0.963
- Test accuracy of the best model: 0.964

### XGBoost
- param_grid = {'learning_rate': [0.001, 0.01, 0.1, 1], 'max_depth': [6, 9, 12, 15], 'n_estimators': [50, 100, 150]}
- Best hyperparameters: {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 150}
- Best cross-validation score: 0.964
- Test accuracy of the best model: 0.964

### Support Vector Machines
- param_grid = {'loss':['hinge', 'squared_hinge'] ,  'alpha':[1e-3, 1e-4, 1e-5]}
- Best hyperparameters: {'alpha': 0.001, 'loss': 'squared_hinge'}
- Best cross-validation score: 0.815
- Test accuracy of the best model: 0.829




