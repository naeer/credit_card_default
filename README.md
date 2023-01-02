# Credit Card Default Prediction

## Overview

A classification model was built to determine the probability of default on credit card payments for the upcoming months. Two different Machine learning models, namely Random Forest and XGBoost, were used to classify customers into two groups - likely to default and not likely to default. 

## How to Run the program
- Clone the repository from the following link: https://github.com/naeer/credit_card_default
- Open the modelling_official.R file in RStudio
- Run the whole program to see the results of both the models

## Data Understanding
Two separate datasets were used to train and evaluate the models. The train set was under the name: 'AT2_credit_train.csv' and the test set (unseen data) was under: 'AT2_credit_test.csv.'

The training set had 25 variables and 23101 observations, while the testing set (unseen data) had 24 variables and 6899 observations. 

| Column | Description |
| --- | --- |
| ID | ID of each client |
| LIMIT_BAL | Amount of given credit in dollars (includes individual and family/supplementary credit |
| SEX | Gender (1=male, 2=female) |
| EDUCATION | (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) |
| MARRIAGE | Marital status (1=married, 2=single, 3=others) |
| AGE | Age in years |
| PAY_X | Repayment status for the past X months. -2 = No credit to pay; -1=paid on time; 0 = the use of revolving credit; 1=payment delay for one month; 2=payment delay for two months etc | 
| BILL_AMTX | Bill amount for past X months |
| PAY_AMTX | Payment amount for last X months |
| default | Default payment next month (1=yes, 0=no). This is the Target variable |

## Data Preparation
- Invalid values in Sex column were removed
- Unusually high Age values of more than 94 were replaced with the median age
- Rows with negative Limit Balance were dropped
- The target variable, 'default', was balanced with the help of SMOTE technique

## Modelling

### Random Forest
Random Forest was chosen as one of the modelling techniques as it works well with big databases and due to the ensemble nature of their structure, they do not require a significant amount of supervised feature selection in order to function effectively.

Initial iteration of the model consisted of 500 trees, each having 4 features while splitting. This yielded a model with an OOB (out-of-bag) rate of 20.05%. After further iterations, the lowest OOB score was obtained when the number of features per split was set to 8. It is important to note that a low OOB score is not a clear indication of a well performing model, it is only letting one know how the current model has correctly predicted samples outside of the bag. A high OOB score does however indicate a class imbalance which is apparent in the dataset. Performing SMOTE brought down the OOB rate to 4% but drastically reduced the AUC score. As a resut, a different technique known as XGBoost was used to model the data

### XGBoost
XGBoost is a tree ensemble learning method that uses weak learners, otherwise known as shallow trees, to make predictions. These trees are grown sequentially, where each tree learns from the previously grown tree, making it an algorithm with high predictive power. XGBoost was chosen over the regular gradient boosting algorithm because it has regularisation parameters that help to reduce overfitting to the training data.

To ensure high predictive accuracy of the model, certain feature engineering decisions were taken. These were in addition to the data preparation steps mentioned earlier. A new attribute called proportion of bill to credit limit paid (bill_limit_ratio_paid) was derived. It was calculated as the bill amount (BILL_AMTX) divided by the credit limit (LIMIT) and multiplied by the status of payment (pay_x) for each month. This was done to capture variations between users that had a high proportion of bill to credit limit and did not pay from those that had a low bill to credit limit ratio and had paid.

A rolling average of paid amounts (PAY_AMTX) for every 2 months was calculated and included in the model instead of the paid amount each month to capture the long-term trend in the monthly payments. Finally, all the categorical variables were one-hot encoded as XGBoost only accepts numerical variables.In the final modelling of the data, all the features were included except for the bill amount from April to August 2005. Only the bill amount of the previous month (September 2005) was included as it seemed to have a high correlation with the bill amounts of all the other months. The final model was built using 2000 trees with a maximum depth of 9 and a learning rate of 0.035. The model assumed that the data about all the customers were independent and similarly distributed. It also assumed that there were no order in the data.

### Evalution Summary

| Model | Accuracy | F1 score | AUC |
| --- | --- | --- | --- |
| Random Forest | 80.7% | 0.873 | 0.78146 |
| XGBoost | 78.30% | 0.633 | 0.79407 |
