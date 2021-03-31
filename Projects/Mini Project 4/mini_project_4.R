#Packages
library(ggplot2) #Used for graphics and visual representations

library(MASS) #Used for LD and QD analysis
library(caret) #Used to handle LOOCV training

library(leaps) #Used for best-subset and forward and backward stepwise selection with linear models 
library(bestglm) #Used for best-subset and forward and backward stepwise selection with generalized linear models 
library(glmnet) #Use for Ridge Regression and Lasso
library(broom) #To create tidy objects for ggplot visualization

rdseed=8466 #Seed to replicate results

#Experiment 1
print("Experiment 1")

#Reading the data
wine=read.delim("wine.txt")

#Converting Region to factor
wine$Region=as.factor(wine$Region)

#Dataframe to track errors of each model
error.df=data.frame(Full=0,Subset=0,Forward=0,Backward=0,RidgeReg=0,Lasso=0)
row.names(error.df)="MSE"

#Dataframe to track coefficients estimates
df.coeff.estimates=as.data.frame(matrix(NA,nrow=8,ncol = 6))

names(df.coeff.estimates)=c("Full","Subset","Forward","Backward","RidgeReg","Lasso")

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
error.df$Full=m_fullloocv$results$RMSE^2

#Estimated coefficients
mfull=lm(Quality~.,data=wine)
coeff.full=mfull$coefficients

row.names(df.coeff.estimates)=names(coeff.full)
df.coeff.estimates[,1]=coeff.full


####Question 1.b####

#Total predictors
totpred=ncol(wine)-1

#Finding best-subset selection models
m.subset=regsubsets(Quality ~ ., wine, nvmax = totpred)
m.subset.summ=summary(m.subset)

#Best model according to adjusted r2
kbest=which.max(m.subset.summ$adjr2)

#Visualizing adj r2 vs number of predictors
print(ggplot(data.frame(predictors = 1:totpred, adj_R2 = m.subset.summ$adjr2),
       aes(x=predictors,y=adj_R2))+geom_line(size=1)+
  geom_point(aes(x=kbest,y=m.subset.summ$adjr2[kbest]),
             color="red",size=4,shape=20)+
  labs(x="Number of Variables",y="Adjusted RSq")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Best Subset Selection Adjusted RSq"))

#Exploring the coefficients of the best model
selected_predictors=coef(m.subset, kbest)
print(selected_predictors) #Clarity, Aroma and Body were dropped

#LOOCV of the best model
msub_LOOCV= train(Quality ~. -Clarity -Aroma -Body,
                data = wine,
                method = "lm",
                trControl = control)

print(msub_LOOCV)

#Estimated MSE
error.df$Subset=msub_LOOCV$results$RMSE^2

#Final model
m.subset=lm(Quality ~. -Clarity -Aroma -Body,data=wine)
#Estimated coefficients
coeff.subset=m.subset$coefficients

df.coeff.estimates[-(2:4),2]=coeff.subset


####Question 1.c####

#Finding forward selection models
m.forward=regsubsets(Quality ~ ., wine, nvmax = totpred,method="forward")
m.forward.summ=summary(m.forward)

#Best model according to adjusted r2
k.forward=which.max(m.forward.summ$adjr2)

#Visualizing adj r2 vs number of predictors
print(ggplot(data.frame(predictors = 1:totpred, adj_R2 = m.forward.summ$adjr2),
       aes(x=predictors,y=adj_R2))+geom_line(size=1)+
  geom_point(aes(x=k.forward,y=m.forward.summ$adjr2[k.forward]),
             color="red",size=4,shape=20)+
  labs(x="Number of Variables",y="Adjusted RSq")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Forward Stepwise Selection Adjusted RSq"))

#Exploring the coefficients of the best model
selected_predictors=coef(m.forward, k.forward)
print(selected_predictors) 
#Same predictors as best-subset selection, therefore we will have same Estimated MSE and coefficients

#Estimated MSE
error.df$Forward=error.df$Subset

#Estimated coefficients same as best subset selection
df.coeff.estimates[-(2:4),3]=coeff.subset


####Question 1.d####

#Finding backward selection models
m.backward=regsubsets(Quality ~ ., wine, nvmax = totpred,method="backward")
m.backward.summ=summary(m.backward)

#Best model according to adjusted r2
k.backward=which.max(m.backward.summ$adjr2)

#Visualizing adj r2 vs number of predictors
print(ggplot(data.frame(predictors = 1:totpred, adj_R2 = m.backward.summ$adjr2),
       aes(x=predictors,y=adj_R2))+geom_line(size=1)+
  geom_point(aes(x=k.backward,y=m.backward.summ$adjr2[k.backward]),
             color="red",size=4,shape=20)+
  labs(x="Number of Variables",y="Adjusted RSq")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Backward Stepwise Selection Adjusted RSq"))

#Exploring the coefficients of the best model
selected_predictors=coef(m.backward, k.backward)
print(selected_predictors)
#Same predictors as best-subset selection, therefore we will have same Estimated MSE and coefficients

#Estimated MSE
error.df$Backward=error.df$Subset

#Estimated coefficients same as best subset selection
df.coeff.estimates[-(2:4),4]=coeff.subset

####Question 1.e####

#Defining predictors, response and lambdas
response=wine[,6]
predictors=model.matrix(Quality ~ ., wine)[, -1]
lambdas = 10^seq(10, -3, length = 100)


m.ridge = glmnet(predictors, response, alpha = 0, lambda = lambdas)

#Visualizing coefficients values vs lambdas
plot(m.ridge, xvar = "lambda")

#Applying LOOCV to find best lambda
cv.ridge = cv.glmnet(predictors, response, alpha = 0, nfolds = n, 
                     lambda = lambdas, grouped = FALSE, type.measure = "mse")

#Tidy data frames to graph
tidy_cv <- tidy(cv.ridge)
glance_cv <- glance(cv.ridge)

#Plot of MSE as a function of lambda
g.ridge = ggplot(tidy_cv, aes(lambda, estimate)) +
  geom_point(color="red") + scale_x_log10()+
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .25)+
  geom_vline(xintercept = glance_cv$lambda.min) +
  geom_vline(xintercept = glance_cv$lambda.1se, lty = 2)+
  labs(x="Lambda",y="MSE")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("MSE vs Lambda, Method Ridge Regression")

print(g.ridge)

#Best lambda value: 0.3125716
bestlamb_ridge=cv.ridge$lambda.min

#Estimated MSE
error.df$RidgeReg=min(cv.ridge$cvm)

#Final model
m.ridge_final=glmnet(predictors, response, alpha = 0, lambda = bestlamb_ridge)
#Estimated coefficients
coeff.ridge=predict(m.ridge, type = "coefficients", s = bestlamb_ridge)[1:8, ]

df.coeff.estimates[,5]=coeff.ridge

####Question 1.f####

m.lasso = glmnet(predictors, response, alpha = 1, lambda = lambdas)

#Visualizing coefficients values vs lambdas
plot(m.lasso, xvar = "lambda")

#Applying LOOCV to find best lambda
cv.lasso = cv.glmnet(predictors, response, alpha = 1, nfolds = n, 
                     lambda = lambdas, grouped = FALSE, type.measure = "mse")

#Tidy data frames to graph
tidy_cv <- tidy(cv.lasso)
glance_cv <- glance(cv.lasso)

#Plot of MSE as a function of lambda
g.lasso = ggplot(tidy_cv, aes(lambda, estimate)) +
  geom_point(color="red") + scale_x_log10()+
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .25)+
  geom_vline(xintercept = glance_cv$lambda.min) +
  geom_vline(xintercept = glance_cv$lambda.1se, lty = 2)+
  labs(x="Lambda",y="MSE")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("MSE vs Lambda, Method Lasso")

print(g.lasso)

#Best lambda value: 0.1261857
bestlamb_lasso=cv.lasso$lambda.min

#Estimated MSE
error.df$Lasso=min(cv.lasso$cvm)

#Final model
m.lasso_final=glmnet(predictors, response, alpha = 1, lambda = bestlamb_lasso)

#Estimated coefficients
coeff.lasso=predict(m.lasso_final, type = "coefficients", s = bestlamb_lasso)[1:8, ]

df.coeff.estimates[,6]=coeff.lasso

####Question 1.g####

final.df=rbind(error.df,df.coeff.estimates)

print(final.df)

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
#10 folds cross validation
set.seed(rdseed)
train_control=trainControl(method = "cv", number = 10)

#Dataframe to track errors of each model
error.df2=data.frame(Full=0,Subset=0,Forward=0,Backward=0,RidgeReg=0,Lasso=0)
row.names(error.df2)="Error"

#Dataframe to track coefficients estimates
df.coeff.estimates2=as.data.frame(matrix(NA,nrow=9,ncol = 6))

names(df.coeff.estimates2)=c("Full","Subset","Forward","Backward","RidgeReg","Lasso")

####Question 2.a####

#10 fold cv on full model of logistic regression
set.seed(rdseed)
m.full.cv = train(
  form = Outcome ~ .,
  data = diabetes,
  trControl = train_control,
  method = "glm",
  family = "binomial"
)

#Estimated test error rate
error.df2$Full=1-m.full.cv$results$Accuracy

#Estimated coefficients
m.full_glm=glm(Outcome~.,data = diabetes,family = binomial)

row.names(df.coeff.estimates2)=names(m.full_glm$coefficients)
df.coeff.estimates2[,1]=m.full_glm$coefficients



####Question 2.b####

#Redifining diabetes data frame to be used on bestglm package
diabetes_glm=diabetes
diabetes_glm$y=diabetes_glm$Outcome
diabetes_glm$Outcome=NULL

#Best-subset selection model according to AIC
fit.subset.logistic =  bestglm(Xy = diabetes_glm, family = binomial, IC = "AIC",
                             method = "exhaustive")

print(fit.subset.logistic$BestModel$coefficients) #SkinThickness was dropped


#10 fold cv on best subset model
set.seed(rdseed)
m.subset.cv = train(
  form = Outcome ~ .-SkinThickness,
  data = diabetes,
  trControl = train_control,
  method = "glm",
  family = "binomial"
)

#Estimated test error rate
error.df2$Subset=1-m.subset.cv$results$Accuracy

#Best model
m_log.subset=glm(Outcome ~ .-SkinThickness,data=diabetes,family = binomial)
#Estimated coefficients
c.subset=m_log.subset$coefficients
df.coeff.estimates2[-5,2]=c.subset


####Question 2.c####

fit.best.forward =  bestglm(Xy = diabetes_glm, family = binomial, IC = "AIC",
                             method = "forward")

print(fit.best.forward$BestModel$coefficients) #SkinThickness was dropped
#Same model as in best-subset selection method

#Estimated test error rate
error.df2$Forward=error.df2$Subset


#Estimated coefficients
df.coeff.estimates2[-5,3]=c.subset



####Question 2.d####

fit.best.backward =  bestglm(Xy = diabetes_glm, family = binomial, IC = "AIC",
                            method = "backward")

print(fit.best.backward$BestModel$coefficients) #SkinThickness was dropped
#Same model as best-subset selection method

#Estimated test error rate
error.df2$Backward=error.df2$Subset

#Estimated coefficients
df.coeff.estimates2[-5,4]=c.subset




####Question 2.e####

#Defining predictors, response and lambdas
response_diab=diabetes[,9]
predictors_diab=model.matrix(Outcome ~ ., diabetes)[, -1]
lambdas = 10^seq(10, -3, length = 100)



mlog.ridge = glmnet(predictors_diab, response_diab, alpha = 0, lambda = lambdas,
                    family = "binomial")

#Visualizing coefficients values vs lambdas
plot(mlog.ridge, xvar = "lambda")

#Applying 10-fold cv to find best lambda
set.seed(rdseed)
cv10.ridge = cv.glmnet(predictors_diab, response_diab, alpha = 0, nfolds = 10, 
                       lambda = lambdas, type.measure = "class", family = "binomial")

#Tidy data frames to graph
tidy_cv <- tidy(cv10.ridge)
glance_cv <- glance(cv10.ridge)

#Plot of MSE as a function of lambda
g.ridgecv = ggplot(tidy_cv, aes(lambda, estimate)) +
  geom_point(color="red") + scale_x_log10()+
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .25)+
  geom_vline(xintercept = glance_cv$lambda.min) +
  geom_vline(xintercept = glance_cv$lambda.1se, lty = 2)+
  labs(x="Lambda",y="Test Error Rate")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Test Error vs Lambda, Method Ridge Regression")

print(g.ridgecv)

#Lambda for the minimal test error rate
loglamb_ridge=cv10.ridge$lambda.min

#Estimated test error
error.df2$RidgeReg=min(cv10.ridge$cvm)


#Best model
m_log.ridge=glmnet(predictors_diab, response_diab, alpha = 0, lambda = loglamb_ridge,
                   family = "binomial")
#Estimated coefficients
c.ridge= predict(m_log.ridge, type = "coefficients", s = loglamb_ridge)[1:9, ]
df.coeff.estimates2[,5]=c.ridge



####Question 2.f####



mlog.lasso = glmnet(predictors_diab, response_diab, alpha = 1, lambda = lambdas,
                    family = "binomial")

#Visualizing coefficients values vs lambdas
plot(mlog.lasso, xvar = "lambda")

#Applying 10-fold cv to find best lambda
set.seed(rdseed)
cv10.lasso = cv.glmnet(predictors_diab, response_diab, alpha = 1, nfolds = 10, 
                       lambda = lambdas, type.measure = "class", family = "binomial")

#Tidy data frames to graph
tidy_cv <- tidy(cv10.lasso)
glance_cv <- glance(cv10.lasso)

#Plot of MSE as a function of lambda
g.lassocv = ggplot(tidy_cv, aes(lambda, estimate)) +
  geom_point(color="red") + scale_x_log10()+
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .25)+
  geom_vline(xintercept = glance_cv$lambda.min) +
  geom_vline(xintercept = glance_cv$lambda.1se, lty = 2)+
  labs(x="Lambda",y="Test Error Rate")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Test Error vs Lambda, Method Lasso")

print(g.lassocv)


#Estimated test error rate
error.df2$Lasso=min(cv10.lasso$cvm)


#Lambda for the minimal test error rate
loglamb_lasso=cv10.lasso$lambda.min

#Best model
m_log.lasso=glmnet(predictors_diab, response_diab, alpha = 1, lambda = loglamb_lasso,
                   family = "binomial")
#Estimated coefficients
c.lasso = predict(m_log.lasso, type = "coefficients", s = loglamb_lasso)[1:9, ]
df.coeff.estimates2[,6]=c.lasso



####Question 2.g####

final.df2=rbind(error.df2,df.coeff.estimates2)

print(final.df2)