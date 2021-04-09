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


pca_x=prcomp(x)

scores=cov(pca_x$x)

#biplot
pca_x$rotation
biplot(pca_x, scale=0)

#Percent of Variance explained
diag(scores)/sum(diag(scores))
pc_var=pca_x$sdev^2
pve=pc_var/sum(pc_var)


g_pve=ggplot(data.frame(PC=1:ncol(x),pve=pve),aes(x=PC,y=pve))+geom_line(size=1)+
  geom_point(shape=1,size=2)+ylim (0,1)+
  xlab("Principal Component")+  ylab("Proportion of Variance Explained")
#4 principal components seems to be appropriate

plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = 'b')


g_cum=ggplot(data.frame(PC=1:ncol(x),CumPVE=cumsum(pve)),aes(x=PC,y=CumPVE))+geom_line(size=1)+
  geom_point(shape=1,size=2)+
  xlab("Principal Component")+  ylab("Cumulative Proportion of Variance Explained")
#With 4 principal components we explained around 84% of the variance


plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = 'b')
