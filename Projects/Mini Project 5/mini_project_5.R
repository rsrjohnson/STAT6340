#Packages
library(ggplot2) #Used for graphics and visual representations

library(MASS) #Used for LD and QD analysis
library(caret) #Used to handle LOOCV training

library(leaps) #Used for best-subset and forward and backward stepwise selection
#with linear models 
library(bestglm) #Used for best-subset and forward and backward stepwise selection
#with generalized linear models 
library(glmnet) #Use for Ridge Regression and Lasso
library(broom) #To create tidy objects for ggplot visualization

library(ISLR)

rdseed=8466 #Seed to replicate results


#Experiment 1
print("Experiment 1")

#Importing data and removing NA values
Hitters=na.omit(Hitters)


####Question 1.a####

#Exploring the data
summary(Hitters) #Different scales observed, standardized the data is recommended

#Separating features and creating dummy variables
y = Hitters$Salary
x = model.matrix(Salary ~ ., Hitters)[, -1]

#Standardizing the predictors
x[,-c(14,15,19)]=scale(x[,-c(14,15,19)])


