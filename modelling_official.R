################ Clearing everything ################

rm(list = ls())

################ Loading Packages ################

library(tidyr)
library(ggplot2)
library(dplyr)
library(forcats)
library(forecast)
library(lubridate)
library(stringr)
library(dlookr)
library(corrplot)
library(rpart)
library(rpart.plot)
library(mlbench)
library(randomForest)
library(caret)
library(pROC)
library(parallel)
library(ROCR)
library(pdp)
library(doParallel)
library(dygraphs)
library(xts)
library(glmnet)
library(magrittr)
library(ROSE)
library(sqldf)
library(scales)
library(ggrepel)
library(xgboost)

################ Reading dataset ################

dset <- read.csv("./data/AT2_credit_train.csv")
head(dset)

################ Data Cleaning ################

#Dropping the ID column
dset <- dset[-1]

## Replacing invalid age

range(dset$AGE)
median(dset$AGE)
dset[c(which(dset$AGE>=94)),5] = 34 #need to include >= to include 94 as well.


##Omitting negative balances
table(dset$LIMIT_BAL)
dset <- dset[!(dset$LIMIT_BAL < 0),]

#Converting Sex column into factors

unique(dset$SEX)

#Dropping values of Sex besides 1 & 2
dset <- dset[!(dset$SEX=="martian" | dset$SEX=="2113" | dset$SEX=="RYDE" | dset$SEX=="orthodontist"),]

## Labeled sex column as a Gender

colnames(dset)[2] <- "Gender"

#Converting Gender into factors

dset$Gender <- as.factor(ifelse(dset$Gender == 1 , 1,
                                ifelse(dset$Gender == 2, 2, "NULL")))



#Converting tagret variable into 1 or 0 and as factors
dset$default <- (ifelse(dset$default == "Y", 1, 0))
dset$default <- as.factor(dset$default)

#Converting Marriage into factors
unique(dset$MARRIAGE)
dset$MARRIAGE <- as.factor(dset$MARRIAGE)

#################### MODELLING FOR RANDOM FOREST 1############################

#Merging values of 0,4 and 5,6 for education and converting into factor
unique(dset$EDUCATION)

#CREATING COPY OF ORIGINAL DATASET
data_sample <- dset

data_sample <- data_sample %>% mutate (EduLevel = case_when(EDUCATION == 0 ~ 4, 
                                                            EDUCATION == 1 ~ 1, 
                                                            EDUCATION == 2 ~ 2, 
                                                            EDUCATION == 3 ~ 3, 
                                                            EDUCATION == 4 ~ 4,
                                                            EDUCATION == 5 ~ 5,
                                                            EDUCATION == 6 ~ 5)) 


data_sample$EduLevel <- as.factor(data_sample$EduLevel)


data_sample <- dplyr::select(data_sample, -c('EDUCATION'))
summary(data_sample)

#Splitting x_dataset into trainging and testing
indxTrain <- createDataPartition(y=data_sample$default, p=0.7, list = FALSE)
training <- data_sample[indxTrain,]
testing <- data_sample[-indxTrain,]

prop.table(table(data_sample$default)) * 100

head(training)

x = training[,-23]
y = training$default

model1 <- randomForest(y ~ ., data = x, importance = TRUE)

model1 #500 for 4

#Fine tuning parameters 
model2 <- randomForest(y ~ ., data = x, ntree = 500, mtry = 8, importance = TRUE)
model2

Predict <- predict(model2, newdata = testing)

confusionMatrix(Predict, testing$default)

#Plotting Variable Performance
importance(model2)
varImpPlot(model2)


#Unseen Test Data for Classifier
vset <- read.csv("./data/AT2_credit_test.csv")
head(vset)


colnames(vset)[3] <- "Gender"

vset <- vset %>% mutate (EduLevel = case_when(EDUCATION == 0 ~ 4, 
                                              EDUCATION == 1 ~ 1, 
                                              EDUCATION == 2 ~ 2, 
                                              EDUCATION == 3 ~ 3, 
                                              EDUCATION == 4 ~ 4,
                                              EDUCATION == 5 ~ 5,
                                              EDUCATION == 6 ~ 5)) 


vset$EduLevel <- as.factor(vset$EduLevel)

vset$MARRIAGE <- as.factor(vset$MARRIAGE)

vset$Gender <- as.factor(ifelse(vset$Gender == 1 , 1,
                                ifelse(vset$Gender == 2, 2, "NULL")))


vset <- dplyr::select(vset, -c('EDUCATION'))
head(vset)
new_prediction <- predict(model2, newdata = vset[,-1], type="prob")
pred_x <- as.data.frame(new_prediction)

vset$default <- pred_x$'1'

final_df <- subset(vset, select = c(ID,default))

final_df

#Writing out to CSV
write.csv(final_df,"./submissions/rf21_prob_scores.csv", row.names = FALSE)


############################## MODELLING FOR XGBOOST ##############################


# Data preparation for XGBoost --------------------------------------------

# Taking a copy of initially prepared data
dset_prep <- dset

## Set Education as factors
dset_prep <- dset_prep %>% mutate (EduLevel = case_when(EDUCATION == 0 ~ 4, 
                                              EDUCATION == 1 ~ 1, 
                                              EDUCATION == 2 ~ 2, 
                                              EDUCATION == 3 ~ 3, 
                                              EDUCATION == 4 ~ 4,
                                              EDUCATION == 5 ~ 5,
                                              EDUCATION == 6 ~ 5))

## Set EduLevel as factors
dset_prep$EduLevel <- as.factor(dset_prep$EduLevel)

# Removing education from data set
dset_prep$EDUCATION <- NULL

## Default proportions
prop.table(table(dset_prep$default))

## Graphing default proportions 
ggplot(dset_prep, aes(x=default)) +
  geom_bar(stat = "count") +
  labs(title = "Distribution of Customers who Default")

# One-hot encoding of categorical variables
dummy <- dummyVars("~.", data = dset_prep)
dset_prep <- data.frame(predict(dummy, newdata = dset_prep))
# Reference: https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/

# Drop default.0 column
dset_prep$default.0 <- NULL
dset_prep$Gender.NULL <- NULL
colnames(dset_prep)[27] <- "default"

# Calculate rolling average of PAY_AMTX columns
dset_prep$PAY_AMT12 <- (dset_prep$PAY_AMT1 + dset_prep$PAY_AMT2)/2
dset_prep$PAY_AMT23 <- (dset_prep$PAY_AMT2 + dset_prep$PAY_AMT3)/2
dset_prep$PAY_AMT34 <- (dset_prep$PAY_AMT3 + dset_prep$PAY_AMT4)/2
dset_prep$PAY_AMT45 <- (dset_prep$PAY_AMT4 + dset_prep$PAY_AMT5)/2
dset_prep$PAY_AMT56 <- (dset_prep$PAY_AMT5 + dset_prep$PAY_AMT6)/2

# Drop the PAY_AMTX columns from the data set
dset_prep$PAY_AMT1 <- NULL
dset_prep$PAY_AMT2 <- NULL
dset_prep$PAY_AMT3 <- NULL
dset_prep$PAY_AMT4 <- NULL
dset_prep$PAY_AMT5 <- NULL
dset_prep$PAY_AMT6 <- NULL

# Calculate the proportion of bill amount to the credit limit multiplied by payment status
dset_prep$bill_limit_ratio_paid1 <- (dset_prep$BILL_AMT1/dset_prep$LIMIT_BAL)*dset_prep$PAY_0
dset_prep$bill_limit_ratio_paid2 <- (dset_prep$BILL_AMT2/dset_prep$LIMIT_BAL)*dset_prep$PAY_2
dset_prep$bill_limit_ratio_paid3 <- (dset_prep$BILL_AMT3/dset_prep$LIMIT_BAL)*dset_prep$PAY_3
dset_prep$bill_limit_ratio_paid4 <- (dset_prep$BILL_AMT4/dset_prep$LIMIT_BAL)*dset_prep$PAY_4
dset_prep$bill_limit_ratio_paid5 <- (dset_prep$BILL_AMT5/dset_prep$LIMIT_BAL)*dset_prep$PAY_5
dset_prep$bill_limit_ratio_paid6 <- (dset_prep$BILL_AMT6/dset_prep$LIMIT_BAL)*dset_prep$PAY_6

# Drop the BILL_AMTX columns
# dset_prep$BILL_AMT1 <- NULL
dset_prep$BILL_AMT2 <- NULL
dset_prep$BILL_AMT3 <- NULL
dset_prep$BILL_AMT4 <- NULL
dset_prep$BILL_AMT5 <- NULL
dset_prep$BILL_AMT6 <- NULL

# Check for collinearity in the data set
cor(dset_prep)


# Creating train and test partitions --------------------------------------

# Find the 70% of row numbers for the data
trainset_size <- floor(0.7 * nrow(dset_prep))
# Reference: Code taken from Lecture 2 Exercise 2. R file (lecture exercise)

# Set seed for reproducibility
set.seed(40)
# Reference: Code taken from Lecture 2 Exercise 2. R file (lecture exercise)

# Get indices of rows to be put in the training set (randomly)
trainset_indices <- sample(seq_len(nrow(dset_prep)), size = trainset_size)
# Reference: Code taken from Lecture 2 Exercise 2. R file (lecture exercise)

# Split the train and test sets
trainset <- dset_prep[trainset_indices,]
testset <- dset_prep[-trainset_indices,]

# Create separate integer list for train and test sets
train.label <- as.integer(trainset$default)
test.label <- as.integer(testset$default)

# Drop the default variable from train and test sets
trainset$default <- NULL
testset$default <- NULL

# Normalize the data set (Z-score normalization)
# Calculate mean and variance parameters for normalization
processed <- preProcess(trainset, method = c("center", "scale"))

# Apply normalization to train set
trainset <- predict(processed, trainset)

# Apply normalization to test set
testset <- predict(processed, testset)


# XGboost Modelling -------------------------------------------------------

# Transform the data set into xgb.Matrix
xgb.train <- xgb.DMatrix(data = as.matrix(trainset), label = train.label)
xgb.test <- xgb.DMatrix(data = as.matrix(testset), label = test.label)
# Reference: https://stackoverflow.com/questions/42743460/r-xgboost-error-building-dmatrix

# Finding out the total number of negative class values by the positive class values
class_count <- as.data.frame(table(train.label))
neg_by_pos_instances <- class_count$Freq[1]/class_count$Freq[2]

# Define the parameters
params <- list(
  booster="gbtree",
  eta=0.035,
  max_depth=9,
  subsample=0.7,
  colsample_bytree=0.8,
  objective="binary:logistic",
  eval_metric = "auc",
  eval_metric = "error",
  scale_pos_weight = neg_by_pos_instances,
  lambda = 12,
  min_child_weight = 14,
  gamma = 22
)

# Set watchlist to both train and test sets
watchlist = list(train = xgb.train, test = xgb.test)

# Set seed for reproducibility
set.seed(43)

# Fit the xgboost model
m <- xgb.train(params = params,
               data = xgb.train,
               nrounds = 2000,
               verbose = 1,
               watchlist = watchlist
)
# Reference: https://xgboost.readthedocs.io/en/stable/R-package/xgboostPresentation.html

# Check the summary of the model
summary(m)

# Check the feature importance for xgboost model
importance_matrix <- xgb.importance(colnames(xgb.train), model = m)
importance_matrix

# Plot the feature importance in a graph
xgb.plot.importance(importance_matrix = importance_matrix)
# Reference: https://www.projectpro.io/recipes/visualise-xgboost-feature-importance-r#mcetoc_1g2k9tqogh

# Lets get the probability estimates for train and test sets
trainset$probability = predict(m, xgb.train, type = "response")
testset$probability = predict(m, xgb.test, type = "response")
testset$prediction = 0

# Make predictions according to threshold
testset[testset$probability >= 0.5, "prediction"] = 1
# Reference: Code taken from 05-GBM.R (lecture exercise)

# Add train set label to train_bal data
trainset$default <- train.label

# Add test set label to testset data
testset$default <- test.label

# Evaluation
# Confusion matrix
confusion_matrix <- table(pred=testset$prediction,true=testset$default)
confusion_matrix
# Reference: Code taken from 05-GBM.R (lecture exercise)

# Precision = TP/(TP+FP)
precision <- confusion_matrix[2,2]/(confusion_matrix[2,1]+confusion_matrix[2,2])
precision

#Recall = TP/(TP+FN)
recall <- confusion_matrix[2,2]/(confusion_matrix[1,2]+confusion_matrix[2,2])
recall

#F1
f1 <- 2*(precision*recall/(precision+recall))
f1

# Accuracy
accuracy <- (confusion_matrix[2,2] + confusion_matrix[1,1])/(confusion_matrix[2,2] + confusion_matrix[1,1] +
                                                               confusion_matrix[2,1] + confusion_matrix[1,2])
accuracy

# Calculate the prediction objects for train and test sets
ROCRpred_train_gbm <- prediction(trainset$probability, trainset$default)
ROCRpred_test_gbm <- prediction(testset$probability, testset$default)
# Reference: Code taken from 07-ROC.R file (lecture exercise)

# tpr and fpr for our training
train_tpr_fpr_gbm = performance(ROCRpred_train_gbm, "tpr","fpr")
train_auc_gbm = performance(ROCRpred_train_gbm, "auc")
train_auc_gbm@y.values[[1]]
# Reference: Code taken from 07-ROC.R file (lecture exercise)

#tpr and fpr for our testing
test_tpr_fpr_gbm = performance(ROCRpred_test_gbm, "tpr","fpr")
test_auc_gbm = performance(ROCRpred_test_gbm, "auc")
test_auc_gbm@y.values[[1]]
# Reference: Code taken from 07-ROC.R file (lecture exercise)

# Plotting the tpr and fpr gains chart ROC for both testing and training data
plot(test_tpr_fpr_gbm, main="Testing and Training ROC Curves", col = "blue")
plot(train_tpr_fpr_gbm, add = T, col = "red")
legend("bottomright", legend = c("Training","Testing"), col = c("red","blue"), lty = 1, lwd = 2)
abline(0,1, col = "darkgray")
grid()
# Reference: Code taken from 07-ROC.R file (lecture exercise)



# Validation of the model -------------------------------------------------

# Load the unseen test data
test_data <- read.csv("./data/AT2_credit_test.csv")

# Summarizing data set
head(test_data)
summary(test_data)
str(test_data)
colnames(test_data)
sum(is.na(test_data))

# Cleaning data set

## Labeled sex column as a Gender

colnames(test_data)[3] <- "Gender"

## Replacing invalid age

range(test_data$AGE)
median(test_data$AGE)
test_data[c(which(test_data$AGE>94)),5] = 34

## Checking gender column 

unique(test_data$Gender)

## Set Gender as factors 

test_data$Gender <- as.factor(ifelse(test_data$Gender == 1 , 1,
                                     ifelse(test_data$Gender == 2, 2, "NULL")))

## Checking gender column again
unique(test_data$Gender)

## Set Education as factors
test_data <- test_data %>% mutate (EduLevel = case_when(EDUCATION == 0 ~ 4, 
                                                        EDUCATION == 1 ~ 1, 
                                                        EDUCATION == 2 ~ 2, 
                                                        EDUCATION == 3 ~ 3, 
                                                        EDUCATION == 4 ~ 4,
                                                        EDUCATION == 5 ~ 5,
                                                        EDUCATION == 6 ~ 5))
# test_data$EDUCATION <- as.factor(test_data$EDUCATION)

## Set Marriage as factors
test_data$MARRIAGE <- as.factor(test_data$MARRIAGE)

## Remove negative values in Limit balance column
test_data <- test_data %>% 
  filter(LIMIT_BAL>=0)

## Set EduLevel as factors
test_data$EduLevel <- as.factor(test_data$EduLevel)

# Removing education from data set
test_data$EDUCATION <- NULL

# One-hot encoding of categorical variables
dummy <- dummyVars("~.", data = test_data)
test_data <- data.frame(predict(dummy, newdata = test_data))
# Reference: https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/

# Drop Gender.NULL column
test_data$Gender.NULL <- NULL

# Calculate rolling average of PAY_AMTX columns
test_data$PAY_AMT12 <- (test_data$PAY_AMT1 + test_data$PAY_AMT2)/2
test_data$PAY_AMT23 <- (test_data$PAY_AMT2 + test_data$PAY_AMT3)/2
test_data$PAY_AMT34 <- (test_data$PAY_AMT3 + test_data$PAY_AMT4)/2
test_data$PAY_AMT45 <- (test_data$PAY_AMT4 + test_data$PAY_AMT5)/2
test_data$PAY_AMT56 <- (test_data$PAY_AMT5 + test_data$PAY_AMT6)/2

# Drop the PAY_AMTX columns from the data set
test_data$PAY_AMT1 <- NULL
test_data$PAY_AMT2 <- NULL
test_data$PAY_AMT3 <- NULL
test_data$PAY_AMT4 <- NULL
test_data$PAY_AMT5 <- NULL
test_data$PAY_AMT6 <- NULL

# Calculate the proportion of bill amount to the credit limit paid multiplied by payment status
test_data$bill_limit_ratio_paid1 <- (test_data$BILL_AMT1/test_data$LIMIT_BAL)*test_data$PAY_0
test_data$bill_limit_ratio_paid2 <- (test_data$BILL_AMT2/test_data$LIMIT_BAL)*test_data$PAY_2
test_data$bill_limit_ratio_paid3 <- (test_data$BILL_AMT3/test_data$LIMIT_BAL)*test_data$PAY_3
test_data$bill_limit_ratio_paid4 <- (test_data$BILL_AMT4/test_data$LIMIT_BAL)*test_data$PAY_4
test_data$bill_limit_ratio_paid5 <- (test_data$BILL_AMT5/test_data$LIMIT_BAL)*test_data$PAY_5
test_data$bill_limit_ratio_paid6 <- (test_data$BILL_AMT6/test_data$LIMIT_BAL)*test_data$PAY_6

# Drop the BILL_AMTX columns
# test_data$BILL_AMT1 <- NULL
test_data$BILL_AMT2 <- NULL
test_data$BILL_AMT3 <- NULL
test_data$BILL_AMT4 <- NULL
test_data$BILL_AMT5 <- NULL
test_data$BILL_AMT6 <- NULL

# Make data frame without ID
test_data_without_ID <- test_data[,-1]
# Apply normalization to unseen test data
test_data_without_ID <- predict(processed, test_data_without_ID)

# Transforming test_data to xgb.DMatrix
test_data_matrix <- xgb.DMatrix(data = as.matrix(test_data_without_ID))

# Probability estimates on validation set
test_data$default <- predict(m, test_data_matrix, type = "response")
# Reference: Code taken from 05-GBM.R (lecture exercise)

# Select ID and default probability
output_df <- test_data %>% select(ID, default)
# Refernece: https://stackoverflow.com/questions/10085806/extracting-specific-columns-from-a-data-frame 

# Write the output_df to a csv file
write.csv(output_df, 
          "./submissions/xgboost_prob_iter4.csv", 
          row.names = FALSE)
# Reference: https://datatofish.com/export-dataframe-to-csv-in-r/

