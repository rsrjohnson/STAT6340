#Packages
library(ggplot2) #Used for graphics and visual representations
library(ggfortify)  #Used to generate biplot ggplot object 
library(caret) #Used to handle LOOCV training
library(glmnet) #Use for Ridge Regression
library(broom) #To create tidy objects for ggplot visualization
library(pls) #Used for PCR and PLS
library(ISLR) #Library for the data

rdseed=8466 #Seed to replicate results


#Experiment 1
print("Experiment 1")

#Importing data and removing NA values
Hitters=na.omit(Hitters)

#Number of observations
n=nrow(Hitters)


#Experiment 2
print("Experiment 2")

#Reading the data
diabetes = read.csv("diabetes.csv")

#Renaming predictors
names(diabetes)=c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin",   "BMI",  
                  "DiabetesPedigreeFunction", "Age","Outcome" )

#Converting Outcome to factor
diabetes$Outcome=as.factor(diabetes$Outcome)