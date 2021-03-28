#Packages
library(ggplot2) #Used for graphics and visual representations, 
#ggplots will be graph under the plot tab in rstudio
library(plotly) #Used for 3D graphics,
#plotly graphics will be under the viewer tab in rstudio
library(ggpubr) #Used for graphics handling
library(MASS) #Used for LD and QD analysis
library(caret) #Used to handle LOOCV training
library(boot) #Used for bootstrapping

library(leaps)
library(glmnet)
library(bestglm)


library(e1071) #Used for tuning knn

rdseed=8467 #Seed to replicate results

#Experiment 1
print("Experiment 1")

#Reading the data
wine=read.delim("wine.txt")

#Converting Region to factor
wine$Region=as.factor(wine$Region)

#Error Dataframe to track errors of each model
error.df=data.frame(MSE=rep(0,6))
row.names(error.df)=c("Full","Subset","Forward","Backward","RidgeReg","Lasso")

#Defining the training control for the train method of caret 
control=trainControl(method = "LOOCV")

####Question 1.a####
n=nrow(wine)

#LOOCV on full regression model
m_fullloocv = train(Quality ~ .,
              data = wine,
              method = "lm",
              trControl = control)

print(m_fullloocv)

#Estimated MSE
error.df["Full","MSE"]=m_fullloocv$results$RMSE^2

mfull=lm(Quality~.,data=wine)

#Estimated coefficients
coeff.full=mfull$coefficients


####Question 1.b####

#total predictors
totpred=ncol(wine)-1

#finding best-subset selection models
m.subset=regsubsets(Quality ~ ., wine, nvmax = totpred)
m.subset.summ=summary(m.subset)

#best model according to adjusted r2
kbest=which.max(m.subset.summ$adjr2)

selected_predictors=coef(m.subset, kbest)
print(selected_predictors)

#LOOCV of the best model
msub_LOOCV= train(Quality ~. -Clarity -Aroma -Body,
                data = wine,
                method = "lm",
                trControl = control)

print(msub_LOOCV)

#Estimated MSE
error.df["Subset","MSE"]=msub_LOOCV$results$RMSE^2

m.subset=lm(Quality ~. -Clarity -Aroma -Body,data=wine)
#Estimated coefficients
coeff.full=m.subset$coefficients

####Question 1.c####

#finding forward selection models
m.forward=regsubsets(Quality ~ ., wine, nvmax = totpred,method="forward")
m.forward.summ=summary(m.forward)

#best model according to adjusted r2
k.forward=which.max(m.forward.summ$adjr2)

selected_predictors=coef(m.forward, k.forward)
print(selected_predictors)

#LOOCV of the best model
mforward_LOOCV= train(Quality ~. -Clarity -Aroma -Body,
                  data = wine,
                  method = "lm",
                  trControl = control)

print(mforward_LOOCV)

#Estimated MSE
error.df["Forward","MSE"]=mforward_LOOCV$results$RMSE^2

#Estimated coefficients same as best subset selection


####Question 1.d####

#finding backward selection models
m.backward=regsubsets(Quality ~ ., wine, nvmax = totpred,method="backward")
m.backward.summ=summary(m.backward)

#best model according to adjusted r2
k.backward=which.max(m.backward.summ$adjr2)

selected_predictors=coef(m.backward, k.backward)
print(selected_predictors)

#LOOCV of the best model
mbackward_LOOCV= train(Quality ~. -Clarity -Aroma -Body,
                      data = wine,
                      method = "lm",
                      trControl = control)

print(mbackward_LOOCV)

#Estimated MSE
error.df["Backward","MSE"]=mbackward_LOOCV$results$RMSE^2

#Estimated coefficients same as best subset selection

####Question 1.e####

response=wine[,6]
predictors=model.matrix(Quality ~ ., wine)[, -1]
lambdas = 10^seq(10, -2, length = 100)

m.ridge = glmnet(predictors, response, alpha = 0, lambda = lambdas)

plot(m.ridge, xvar = "lambda")


cv.ridge = cv.glmnet(predictors, response, alpha = 0, nfolds = n, lambda = lambdas, grouped = FALSE)

plot(cv.ridge)

bestlamb_ridge=cv.ridge$lambda.min

sqerrors.ridge=sapply(1:n,function(i){
  
  m.ridge_i=glmnet(predictors[-i,], response[-i], alpha = 0, lambda = bestlamb_ridge)
  
  ridge.pred=predict(m.ridge_i, s = bestlamb_ridge, newx =matrix(predictors[i, ],nrow=1,ncol=7))
  (ridge.pred - response[i])^2
})

#Estimated MSE
error.df["RidgeReg","MSE"]=mean(sqerrors.ridge)

m.ridge_final=glmnet(predictors, response, alpha = 0, lambda = bestlamb_ridge)
#Estimated coefficients
coeff.ridge=predict(m.ridge_final, type = "coefficients", s = bestlamb_ridge)[1:8, ]

####Question 1.f####

m.lasso = glmnet(predictors, response, alpha = 1, lambda = lambdas)

plot(m.lasso, xvar = "lambda")


cv.lasso = cv.glmnet(predictors, response, alpha = 1, nfolds = n, lambda = lambdas, grouped = FALSE)

plot(cv.lasso)

bestlamb_lasso=cv.lasso$lambda.min

sqerrors.lasso=sapply(1:n,function(i){
  
  m.lasso_i=glmnet(predictors[-i,], response[-i], alpha = 1, lambda = bestlamb_lasso)
  
  lasso.pred=predict(m.lasso_i, s = bestlamb_lasso, newx =matrix(predictors[i, ],nrow=1,ncol=7))
  (lasso.pred - response[i])^2
})

#Estimated MSE
error.df["Lasso","MSE"]=mean(sqerrors.lasso)

m.lasso_final=glmnet(predictors, response, alpha = 1, lambda = bestlamb_lasso)
#Estimated coefficients
coeff.lasso=predict(m.lasso_final, type = "coefficients", s = bestlamb_lasso)[1:8, ]

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

diabetes_glm=diabetes
diabetes_glm

#Defining the training control for the train method of caret 
train_control=trainControl(method = "cv", number = 10)


####Question 1.e####

#LOOCV on full model of logistic regression
mloocv = train(
  form = Outcome ~ .,
  data = diabetes,
  trControl = train_control,
  method = "glm",
  family = "binomial"
)





error_mat=matrix(0,n,totpred)

for (j in 1:n) {
  best.fit = regsubsets(Quality ~ ., data = wine[-j,], nvmax = totpred)
  for (i in 1:totpred) {
    pred = predict(best.fit, wine[j, ], id = i)
    error_mat[j, i] = (wine$Quality[j] - pred)^2
  }
}

MSErrors=apply(error_mat,2,mean)

best_num=which.min(MSErrors)
