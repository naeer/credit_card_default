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


