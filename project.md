---
title: "Practical Machine Learning: Predicting Exercise Manner"
author: "Pratham Yogeesh Nayak"
date: "2026-09-11"
output: 
  html_document:
    keep_md: true
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible 
to collect large amounts of data about personal activity. In this project, 
data from accelerometers on the belt, forearm, arm, and dumbbell of 6 
participants is used to predict the manner in which they performed barbell 
lifts (the `classe` variable), which can be one of 5 classes (A-E), 
corresponding to correct form (A) and four common mistakes (B-E).

## Loading and Cleaning the Data


```r
library(caret)
library(randomForest)

training <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
testing  <- read.csv("pml-testing.csv",  na.strings = c("NA", "", "#DIV/0!"))
```

The raw dataset contains 160 variables, many of which are summary statistics 
that are mostly missing, along with bookkeeping columns not useful for 
prediction. These were removed:


```r
na_frac <- colMeans(is.na(training))
keep_cols <- names(na_frac[na_frac < 0.5])
training <- training[, keep_cols]

training <- training[, !(names(training) %in% 
    c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2",
      "cvtd_timestamp","new_window","num_window"))]

nzv <- nearZeroVar(training)
if(length(nzv) > 0) training <- training[, -nzv]

dim(training)
```

```
## [1] 19622    53
```

This left 52 predictors plus the outcome `classe`.

## Cross-Validation Strategy

The training data was split into a training set (70%) and a validation set 
(30%). The model itself was also trained using 5-fold cross-validation, so 
that the reported accuracy reflects performance on data not used to fit 
each fold.


```r
set.seed(1234)
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
trainSet <- training[inTrain, ]
validSet <- training[-inTrain, ]
```

## Model Building

A random forest model was chosen for its strong performance on 
high-dimensional sensor data without requiring manual feature engineering.


```r
modFit <- readRDS("modFit.rds")
modFit
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10991, 10989, 10991, 10988 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9909732  0.9885808
##   27    0.9910464  0.9886728
##   52    0.9842034  0.9800155
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

## Out-of-Sample Error Estimate

The model was evaluated on the held-out validation set (data not used in 
training or cross-validation):


```r
pred <- predict(modFit, validSet)
confusionMatrix(pred, factor(validSet$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    5    0    0    0
##          B    0 1129   14    0    0
##          C    1    5 1010    8    0
##          D    0    0    2  955    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9915, 0.9957)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9912   0.9844   0.9907   1.0000
## Specificity            0.9988   0.9971   0.9971   0.9996   0.9998
## Pos Pred Value         0.9970   0.9878   0.9863   0.9979   0.9991
## Neg Pred Value         0.9998   0.9979   0.9967   0.9982   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1918   0.1716   0.1623   0.1839
## Detection Prevalence   0.2851   0.1942   0.1740   0.1626   0.1840
## Balanced Accuracy      0.9991   0.9941   0.9908   0.9951   0.9999
```

The model achieved **99.39% accuracy** on the validation set, giving an 
estimated **out-of-sample error of approximately 0.61%** (1 − accuracy).

## Predicting the 20 Test Cases


```r
predictions <- predict(modFit, testing)
predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion

A random forest model trained with 5-fold cross-validation achieved 
approximately 99.4% accuracy on a held-out validation set, corresponding 
to an estimated out-of-sample error rate of about 0.6%. This model was 
used to generate predictions for the 20 test cases above.
