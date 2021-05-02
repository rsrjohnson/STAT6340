#Packages
library(ggplot2) #Used for graphics and visual representations

library(tree) #Used for decision trees
library(randomForest) #Used for bagging and random forest
library(gbm) #Used for boosting
library(e1071) #Used for hyperparameter tuning
library(data.table)

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

#Creating dataframe with log(Salary)
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

#Regions
hitree

#Visualizing the tree 
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

####Question 1.b####

#Using LOOCV(K=n) to find performance of the pruned trees
cv.hitree = cv.tree(hitree, K=n)

#Best Size = 9, pruning does not help
best_size=which.min(cv.hitree$size)

#Plotting Test MSE vs Size
ggplot(data.frame(Tree_Size = cv.hitree$size, MSE= cv.hitree$dev/n ),aes(x=Tree_Size, y=MSE))+
  geom_line()+geom_point(color="red")+scale_x_continuous(breaks=1:9)



#Since the unpruned tree is the best we have the same test MSE as before.
print(mean(hit.errors))


####Question 1.c####

#Fitting a bagging model
set.seed(rdseed)
bag.hit<- randomForest(logSal ~ ., data = new_hit, 
                           mtry = p, ntree = B, importance = TRUE)

#Importance of predictors
print(bag.hit$importance)
varImpPlot(bag.hit, main="Bagging Predictor Importance")
#Career predictors like CAtBat, CRuns, CRBI and CHits are the most important

#LOOCV to estimate MSE
bag.errors=sapply(1:n, function(i){
  
  ti = bag.hit<- randomForest(logSal ~ ., data = new_hit[-i,], 
                              mtry = p, ntree = B)
  
  ti.pred= predict(ti, new_hit[i,])
  
  (ti.pred-new_hit$logSal[i])^2
})

error.df["Bagging","MSE"]=mean(bag.errors)

print(mean(bag.errors))

####Question 1.d####

#Number of predictors sampled per tree
m=round(p/3)

#Fitting a random forest model
set.seed(rdseed)
rf.hit <- randomForest(logSal ~ ., data = new_hit, 
                        mtry = m, ntree = B, importance = TRUE)

#Importance of predictors
print(rf.hit$importance)
varImpPlot(rf.hit, main="Random Forest Predictor Importance")
#Career predictors are the most important predictors

#LOOCV to estimate MSE
rf.errors=sapply(1:n, function(i){
  
  ti = randomForest(logSal ~ ., data = new_hit[-i,], 
                    mtry = m, ntree = B)
  
  ti.pred= predict(ti, new_hit[i,])
  
  (ti.pred-new_hit$logSal[i])^2
})

error.df["RF","MSE"]=mean(rf.errors)

print(rf.errors)

####Question 1.e####

#Depth of the trees
d=1

#Shrinkage value
lambda=0.01


set.seed(rdseed)
boost.hit <- gbm(logSal ~ ., data = new_hit, distribution = "gaussian", 
                    n.trees = B, interaction.depth = d,shrinkage = lambda)
summary(boost.hit)

#Visualizing the effect of some predictors
par(mfrow = c(1, 2))
plot(boost.hit, i = "CAtBat")
plot(boost.hit, i = "CRuns")

#LOOCV to estimate MSE
boost.errors=sapply(1:n, function(i){
  
  boost.i=gbm(logSal ~ ., data = new_hit[-i,], distribution = "gaussian", 
              n.trees = B, interaction.depth = d,shrinkage = lambda)
  
  yi <- predict(boost.i, newdata = new_hit[i, ], 
                n.trees = B)
  (yi - new_hit$logSal[i])^2
})


error.df["Boosting","MSE"]=mean(boost.errors)
print(mean(boost.errors))

####Question 1.e####

print(error.df)


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


#Number of observations
nobs=nrow(diabetes)

#Dataframe to track errors of each model
class.error=data.frame(Error=rep(0,3))
row.names(class.error)=c("SVC","SVMP","SVMR")

#Hyperparameters for the svm models
costs=c(0.001, 0.01, 0.1, c(1:10),100)
gamma_val=c(0.1,0.5, 1, 2, 3, 4,5)


####Question 2.a####

#Tuning a support vector classifier
set.seed(rdseed)
svc.tune=tune(svm, Outcome ~ ., data = diabetes, kernel = "linear", 
              ranges = list(cost=costs),scale=TRUE)

#Best model cost = 1
bestmod = svc.tune$best.model
summary(bestmod)
print(bestmod$cost)


#Estimated Error Rate
class.error["SVC","Error"]=svc.tune$best.performance


####Question 2.b####

#Tuning support vector machine with polynomial kernel of degree 2
set.seed(rdseed)
svm2.tune=tune(svm, Outcome ~ ., data = diabetes, kernel = "polynomial", degree=2, 
              ranges = list(cost=costs),scale=TRUE)

#Best model cost = 5
bestmod2 = svm2.tune$best.model
summary(bestmod2)
print(bestmod2$cost)

#Estimated Error Rate
class.error["SVMP","Error"]=svm2.tune$best.performance


####Question 2.c####

#Tuning support vector machine with radial kernel
set.seed(rdseed)
svmr.tune=tune(svm, Outcome ~ ., data = diabetes, kernel = "radial", 
               ranges = list(cost=costs, gamma=gamma_val),scale=TRUE)

#Best model cost = 6, gamma = 1
bestmodr = svmr.tune$best.model
summary(bestmodr)
print(bestmodr$cost)
print(bestmodr$gamma)


#Estimated Error Rate
class.error["SVMR","Error"]=svmr.tune$best.performance


####Question 2.d####

print(class.error)