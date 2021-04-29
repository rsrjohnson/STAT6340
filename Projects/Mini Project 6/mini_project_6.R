#Packages
library(ggplot2) #Used for graphics and visual representations
library(ggfortify)  #Used to generate biplot ggplot object 
library(caret) #Used to handle LOOCV training
library(glmnet) #Use for Ridge Regression
library(broom) #To create tidy objects for ggplot visualization
library(pls) #Used for PCR and PLS
library(ISLR) #Library for the data

library(tree)
library(randomForest)
library(gbm)
library(e1071)

rdseed=8466 #Seed to replicate results


#Experiment 1
print("Experiment 1")

#Importing data and removing NA values
Hitters=na.omit(Hitters)

#Number of observations
n=nrow(Hitters)
#Number of predictors
p=ncol(Hitters)-1
#Number of bootstrap samples
B=1000

#Dataframe with log(Salary)
new_hit=Hitters
new_hit$logSal=log(Hitters$Salary)
new_hit$Salary=NULL

#Dataframe to track errors of each model
error.df=data.frame(MSE=rep(0,4))
row.names(error.df)=c("DT","Bagging","RF","Boosting")


####Question 1.a####

#First we grow a full tree with all the data
hitree=tree(logSal ~ ., new_hit)

summary(hitree)

plot(hitree)
text(hitree, pretty = 0)


#LOOCV to estimate MSE
hit.errors=sapply(1:n, function(i){
  
  ti = tree(logSal ~ ., new_hit[-i,])
  
  ti.pred= predict(ti, new_hit[i,])
  
  (ti.pred-new_hit$logSal[i])^2
})

#Estimated MSE

error.df["DT","MSE"]=mean(hit.errors)

print(mean(hit.errors))


#Desciber regions to be done

####Question 1.b####

#Using LOOCV(K=n) to find values of several prunings
cv.hitree = cv.tree(hitree, K=n)

#Best Size = 9, pruning does not help
best_size=which.min(cv.hitree$size)

#Plotting Test MSE vs Size
plot(cv.hitree$size, cv.hitree$dev/n, type = "b")


#Since the unpruned tree is the best we have the same test MSE as before.
print(mean(hit.errors))
min(cv.hitree$dev)/n


prune.hit <- prune.tree(hitree, best = best_size)
plot(prune.hit)
text(prune.hit, pretty = 0)

#Predictors importance to be done



####Question 1.c####
set.seed(rdseed)
bag.hit<- randomForest(logSal ~ ., data = new_hit, 
                           mtry = p, ntree = B, importance = TRUE)
bag.hit


bag.errors=sapply(1:n, function(i){
  
  ti = bag.hit<- randomForest(logSal ~ ., data = new_hit[-i,], 
                              mtry = p, ntree = B, importance = TRUE)
  
  ti.pred= predict(ti, new_hit[i,])
  
  (ti.pred-new_hit$logSal[i])^2
})

error.df["Bagging","MSE"]=mean(bag.errors)


####Question 1.d####

m=round(p/3)

set.seed(rdseed)
rf.hit <- randomForest(logSal ~ ., data = new_hit, 
                        mtry = m, ntree = B, importance = TRUE)
rf.hit


rf.errors=sapply(1:n, function(i){
  
  ti = randomForest(logSal ~ ., data = new_hit[-i,], 
                    mtry = m, ntree = B)
  
  ti.pred= predict(ti, new_hit[i,])
  
  (ti.pred-new_hit$logSal[i])^2
})

error.df["RF","MSE"]=mean(rf.errors)

####Question 1.e####

d=1
lambda=0.01


set.seed(rdseed)
boost.hit <- gbm(logSal ~ ., data = new_hit, distribution = "gaussian", 
                    n.trees = B, interaction.depth = d,shrinkage = lambda)
summary(boost.hit)

par(mfrow = c(1, 2))
plot(boost.hit, i = "CAtBat")
plot(boost.hit, i = "CRuns")

boost.errors=sapply(1:n, function(i){
  
  boost.i=gbm(logSal ~ ., data = new_hit[-i,], distribution = "gaussian", 
              n.trees = B, interaction.depth = d,shrinkage = lambda)
  
  yi <- predict(boost.i, newdata = new_hit[i, ], 
                n.trees = B)
  (yi - new_hit$logSal[i])^2
})


error.df["Boosting","MSE"]=mean(boost.errors)


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

nobs=nrow(diabetes)

class.error=data.frame(Error=rep(0,3))
row.names(class.error)=c("SVC","SVMP","SVMR")


costs=list(cost = c(0.001, 0.01, 0.1, 1, 5, 10))
gamma_val=c(0.5, 1, 2, 3, 4)
kfold=10

####Question 2.a####

set.seed(rdseed)
best.svc=tune(svm, Outcome ~ ., data = diabetes, kernel = "linear", 
              ranges = costs, scale = TRUE)

bestmod = best.svc$best.model
summary(bestmod)

best_cost=bestmod$cost

svc.errors=sapply(1:nobs, function(i){
  
  ti =svm(Outcome ~ ., data = diabetes[-i,], kernel = "linear", cost = best_cost, scale = TRUE)
  
  ti.pred= predict(ti, diabetes[i,])
  
  (ti.pred!=diabetes$Outcome[i])
})


class.error["SVC","Error"]=mean(svc.errors)


####Question 2.b####

set.seed(rdseed)
best.svm2=tune(svm, Outcome ~ ., data = diabetes, kernel = "polynomial", degree=2, 
              ranges = costs, scale = TRUE)

bestmod2 = best.svm2$best.model
summary(bestmod2)

best_cost2=bestmod2$cost

svm.errors2=sapply(1:nobs, function(i){
  
  ti =svm(Outcome ~ ., data = diabetes[-i,], kernel = "polynomial", degree=2,
          cost=best_cost2, scale = TRUE)
  
  ti.pred= predict(ti, diabetes[i,])
  
  (ti.pred!=diabetes$Outcome[i])
})


class.error["SVMP","Error"]=mean(svm.errors2)


####Question 2.c####

set.seed(rdseed)
best.svmr=tune(svm, Outcome ~ ., data = diabetes, kernel = "radial", 
               ranges = costs, gamma=gamma_val,scale = TRUE)

bestmodr = best.svmr$best.model
summary(bestmodr)

best_costr=bestmodr$cost
best_gam=bestmodr$gamma

svm.errorsr=sapply(1:nobs, function(i){
  
  ti =svm(Outcome ~ ., data = diabetes[-i,], kernel = "radial",
          cost=best_costr, gamma=best_gam, scale = TRUE)
  
  ti.pred= predict(ti, diabetes[i,])
  
  (ti.pred!=diabetes$Outcome[i])
})


class.error["SVMR","Error"]=mean(svm.errorsr)