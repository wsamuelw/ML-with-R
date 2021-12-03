# a simple binary classification ML example

library(caret)
library(tidyverse)
library(mlbench)

data(PimaIndiansDiabetes)
df <- PimaIndiansDiabetes

head(df)

# pregnant glucose pressure triceps insulin mass pedigree age diabetes
# 1        6     148       72      35       0 33.6    0.627  50      pos
# 2        1      85       66      29       0 26.6    0.351  31      neg
# 3        8     183       64       0       0 23.3    0.672  32      pos
# 4        1      89       66      23      94 28.1    0.167  21      neg
# 5        0     137       40      35     168 43.1    2.288  33      pos
# 6        5     116       74       0       0 25.6    0.201  30      neg

# the distribution of the response variable

table(df$diabetes)
# neg pos 
# 500 268 

set.seed(123)

training.samples <- df$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)

train  <- df[training.samples, ]
test <- df[-training.samples, ]

# create a generalized linear model
# response = diabetes
# predictors = all features
model <- glm(diabetes ~., data = train, family = binomial)
summary(model)

# Call:
# glm(formula = diabetes ~ ., family = binomial, data = train)
# 
# Deviance Residuals: 
# Min       1Q   Median       3Q      Max  
# -2.4719  -0.7674  -0.4402   0.7776   2.9436  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -7.8116450  0.7694301 -10.153  < 2e-16 ***
# pregnant     0.0998300  0.0358381   2.786  0.00534 ** 
# glucose      0.0342306  0.0040533   8.445  < 2e-16 ***
# pressure    -0.0148671  0.0055567  -2.676  0.00746 ** 
# triceps     -0.0006103  0.0076247  -0.080  0.93621    
# insulin     -0.0007117  0.0009565  -0.744  0.45681    
# mass         0.0806695  0.0165458   4.876 1.09e-06 ***
# pedigree     0.9355556  0.3388561   2.761  0.00576 ** 
# age          0.0154356  0.0102718   1.503  0.13291    
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 796.05  on 614  degrees of freedom
# Residual deviance: 598.41  on 606  degrees of freedom
# AIC: 616.41
# 
# Number of Fisher Scoring iterations: 5

# make predictions
probability <- model %>% predict(test, type = "response")
predicted <- ifelse(probability > 0.5, "pos", "neg")

test$proba_Y <- probability
test$proba_N <- 1 - test$proba_Y
test$predicted <- predicted
test$predicted_numeric <- ifelse(probability > 0.5, 1, 0)

head(test)

#    pregnant glucose pressure triceps insulin mass pedigree age diabetes predicted predicted_numeric    proba_Y    proba_N
# 2         1      85       66      29       0 26.6    0.351  31      neg       neg                 0 0.05476158 0.94523842
# 5         0     137       40      35     168 43.1    2.288  33      pos       pos                 1 0.90628701 0.09371299
# 17        0     118       84      47     230 45.8    0.551  31      pos       neg                 0 0.37168638 0.62831362
# 25       11     143       94      33     146 36.6    0.254  51      pos       pos                 1 0.65413262 0.34586738
# 27        7     147       76       0       0 39.4    0.257  43      pos       pos                 1 0.70510173 0.29489827
# 34        6      92       92       0       0 19.9    0.188  28      neg       neg                 0 0.03850063 0.96149937

# create evaluation metrics
# https://towardsdatascience.com/top-5-metrics-for-evaluating-classification-model-83ede24c7584

# Confusion Matrix

# need to be converted to factor
caret::confusionMatrix(as.factor(test$predicted), as.factor(test$diabetes))

# Confusion Matrix and Statistics
# 
# Reference
# Prediction neg pos
# neg  91  21
# pos   9  32
# 
# Accuracy : 0.8039          
# 95% CI : (0.7321, 0.8636)
# No Information Rate : 0.6536          
# P-Value [Acc > NIR] : 3.3e-05         
# 
# Kappa : 0.5426          
# 
# Mcnemar's Test P-Value : 0.04461         
#                                           
#             Sensitivity : 0.9100          
#             Specificity : 0.6038          
#          Pos Pred Value : 0.8125          
#          Neg Pred Value : 0.7805          
#              Prevalence : 0.6536          
#          Detection Rate : 0.5948          
#    Detection Prevalence : 0.7320          
#       Balanced Accuracy : 0.7569          
#                                           
#        'Positive' Class : neg             


test %>% 
  group_by(predicted, diabetes) %>% 
  rename(actual = diabetes) %>% 
  summarise(n = n ())

# predicted actual       n
# neg       neg         91 # True Negative
# neg       pos         21 # False Negative
# pos       neg          9 # False Positive
# pos       pos         32 # True Positive

# True Positive: Case when you correctly predict the positive result
# True Negative: Case when you correctly predict negative result

# False Positive: Case when you predict the result to be positive but it is actually negative
# False Negative: Case then you predict the result to be negative but it is actually positive

TN = 91
FN = 21
FP = 9
TP = 32

# Accuracy
# measure of how the overall model performs
# it is the sum of accurate predictions / total 

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Accuracy # 0.8039216

# Recall
# it is used when the aim is to capture a maximum number of positive cases
# TP & FN = actually positive
# the formula is TP/(TP+FN)
# suppose we have 32 / (32+2), the metric is 0.9411765
# the model is doing well in terms of predicting positive cases as it only misses 2

Recall = TP/(TP+FN)
Recall # 0.6037736

# Precision
Precision = TP/(TP+FP)
Precision # 0.7804878

# F1 Score
F1.Score = 2*(Precision*Recall) / (Precision+Recall)
F1.Score # 0.6808511

# AUC
library(pROC)
auc(test$diabetes, probability) # Area under the curve: 0.8925

# Log Loss
library(MLmetrics)
LogLoss(y_pred = model$fitted.values, y_true = test$predicted_numeric) # not sure how to make it work

# AUC
PRAUC(y_pred = model$fitted.values, y_true = train$diabetes) # 0.7141541

Accuracy(y_pred = predicted, y_true = test$diabetes) # 0.8039216

F1_Score(y_pred = predicted, y_true = test$diabetes, positive = "0")







