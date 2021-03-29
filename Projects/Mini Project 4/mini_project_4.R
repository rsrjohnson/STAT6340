#Packages
library(ggplot2) #Used for graphics and visual representations

library(ggpubr) #Used for graphics handling
library(MASS) #Used for LD and QD analysis
library(caret) #Used to handle LOOCV training
library(boot) #Used for bootstrapping

library(leaps)
library(glmnet)
library(bestglm)


rdseed=8466 #Seed to replicate results

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

#Number of observations
n=nrow(wine)

####Question 1.a####


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

plot(m.subset.summ$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", 
     type = "l")
points(kbest, m.subset.summ$adjr2[kbest], col = "red", cex = 2, pch = 20)


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

plot(m.forward.summ$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", 
     type = "l")
points(k.forward, m.forward.summ$adjr2[k.forward], col = "red", cex = 2, pch = 20)


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

plot(m.backward.summ$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", 
     type = "l")
points(k.backward, m.backward.summ$adjr2[k.backward], col = "red", cex = 2, pch = 20)


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

#Defining predictors, response and lambdas
response=wine[,6]
predictors=model.matrix(Quality ~ ., wine)[, -1]
lambdas = 10^seq(10, -2, length = 100)


#Visualizing coefficients values
m.ridge = glmnet(predictors, response, alpha = 0, lambda = lambdas)
plot(m.ridge, xvar = "lambda")

#Applying LOOCV to find best lambda
cv.ridge = cv.glmnet(predictors, response, alpha = 0, nfolds = n, lambda = lambdas, grouped = FALSE, type.measure = "mse")
plot(cv.ridge)
bestlamb_ridge=cv.ridge$lambda.min

#Estimated MSE
min(cv.ridge$cvm)

#Applying LOOCV with best lambda to estimate test error
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

#Visualizing coefficients values
m.lasso = glmnet(predictors, response, alpha = 1, lambda = lambdas)
plot(m.lasso, xvar = "lambda")

#Applying LOOCV to find best lambda
cv.lasso = cv.glmnet(predictors, response, alpha = 1, nfolds = n, lambda = lambdas, grouped = FALSE, type.measure = "mse")
plot(cv.lasso)
bestlamb_lasso=cv.lasso$lambda.min

#Estimated MSE
min(cv.lasso$cvm)

sqerrors.lasso=sapply(1:n,function(i){
  
  m.lasso_i=glmnet(predictors[-i,], response[-i], alpha = 1, lambda = bestlamb_lasso)
  
  lasso.pred=predict(m.lasso_i, s = bestlamb_lasso, newx =matrix(predictors[i, ],nrow=1,ncol=7))
  (lasso.pred - response[i])^2
})

#Estimated MSE
error.df["Lasso","MSE"]=mean(sqerrors.lasso)

#Final model
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

#Defining the training control for the train method of caret 
set.seed(rdseed)
train_control=trainControl(method = "cv", number = 10)


error.df2=data.frame(Error=rep(0,6))
row.names(error.df2)=c("Full","Subset","Forward","Backward","RidgeReg","Lasso")

df.coeff.estimates=as.data.frame(matrix(NA,nrow=6,ncol = 9))

####Question 2.a####

#LOOCV on full model of logistic regression
set.seed(rdseed)
m.full.loocv = train(
  form = Outcome ~ .,
  data = diabetes,
  trControl = train_control,
  method = "glm",
  family = "binomial"
)

error.df2["Full","Error"]=1-m.full.loocv$results$Accuracy

#Estimated coefficients
m.full_glm=glm(Outcome~.,data = diabetes,family = binomial)

names(df.coeff.estimates)=names(m.full_glm$coefficients)
df.coeff.estimates[1,]=m.full_glm$coefficients




####Question 2.b####

#Redifining diabetes data frame to be used on bestglm package
diabetes_glm=diabetes
diabetes_glm$y=diabetes_glm$Outcome
diabetes_glm$Outcome=NULL


fit.subset.logistic =  bestglm(Xy = diabetes_glm, family = binomial, IC = "AIC",
                             method = "exhaustive")

print(fit.subset.logistic$BestModel$coefficients) #SkinThickness was dropped

#Best model
m_log.subset=fit.subset.logistic$BestModel 
#Estimated coefficients
c.subset=m_log.subset$coefficients
df.coeff.estimates[2,-5]=c.subset


#LOOCV on best subset model
set.seed(rdseed)
m.subset.loocv = train(
  form = Outcome ~ .-SkinThickness,
  data = diabetes,
  trControl = train_control,
  method = "glm",
  family = "binomial"
)

error.df2["Subset","Error"]=1-m.subset.loocv$results$Accuracy


####Question 2.c####

fit.best.forward =  bestglm(Xy = diabetes_glm, family = binomial, IC = "AIC",
                             method = "forward")

print(fit.best.forward$BestModel$coefficients) #SkinThickness was dropped

#Best model
m_log.forward=fit.best.forward$BestModel 
#Estimated coefficients
c.forward=m_log.forward$coefficients
df.coeff.estimates[3,-5]=c.forward


#LOOCV on best subset model
set.seed(rdseed)
m.forward.loocv = train(
  form = Outcome ~ .-SkinThickness,
  data = diabetes,
  trControl = train_control,
  method = "glm",
  family = "binomial"
)

error.df2["Forward","Error"]=1-m.forward.loocv$results$Accuracy



####Question 2.d####

fit.best.backward =  bestglm(Xy = diabetes_glm, family = binomial, IC = "AIC",
                            method = "backward")

print(fit.best.backward$BestModel$coefficients) #SkinThickness was dropped

#Best model
m_log.backward=fit.best.backward$BestModel 
#Estimated coefficients
c.backward=m_log.backward$coefficients
df.coeff.estimates[4,-5]=c.backward


#LOOCV on best subset model
set.seed(rdseed)
m.backward.loocv = train(
  form = Outcome ~ .-SkinThickness,
  data = diabetes,
  trControl = train_control,
  method = "glm",
  family = "binomial"
)

error.df2["Backward","Error"]=1-m.backward.loocv$results$Accuracy



####Question 2.e####

response_diab=diabetes[,9]
predictors_diab=model.matrix(Outcome ~ ., diabetes)[, -1]
lambdas = 10^seq(10, -2, length = 100)


#Visualizing coefficients values
mlog.ridge = glmnet(predictors_diab, response_diab, alpha = 0, lambda = lambdas,
                    family = "binomial")
plot(mlog.ridge, xvar = "lambda")

#Applying 10-fold cv to find best lambda
set.seed(rdseed)
cv10.ridge = cv.glmnet(predictors_diab, response_diab, alpha = 0, nfolds = 10, 
                       lambda = lambdas, type.measure = "class", family = "binomial")
plot(cv10.ridge)
loglamb_ridge=cv10.ridge$lambda.min

#Estimated Error
error.df2["RidgeReg","Error"]=min(cv10.ridge$cvm)


#Best model
m_log.ridge=glmnet(predictors_diab, response_diab, alpha = 0, lambda = loglamb_ridge,
                   family = "binomial")
#Estimated coefficients
c.ridge= predict(m_log.ridge, type = "coefficients", s = loglamb_ridge)[1:9, ]
df.coeff.estimates[5,]=c.ridge



####Question 2.f####


#Visualizing coefficients values
mlog.lasso = glmnet(predictors_diab, response_diab, alpha = 1, lambda = lambdas,
                    family = "binomial")
plot(mlog.lasso, xvar = "lambda")

#Applying 10-fold cv to find best lambda
set.seed(rdseed)
cv10.lasso = cv.glmnet(predictors_diab, response_diab, alpha = 1, nfolds = 10, 
                       lambda = lambdas, type.measure = "class", family = "binomial")
plot(cv10.lasso)
loglamb_lasso=cv10.lasso$lambda.min

#Estimated MSE
error.df2["Lasso","Error"]=min(cv10.lasso$cvm)


#Best model
m_log.lasso=glmnet(predictors_diab, response_diab, alpha = 1, lambda = loglamb_lasso,
                   family = "binomial")
#Estimated coefficients
c.lasso = predict(m_log.lasso, type = "coefficients", s = loglamb_lasso)[1:9, ]
df.coeff.estimates[6,]=c.lasso



####Question 2.f####

final.df2=cbind(error.df2,df.coeff.estimates)
