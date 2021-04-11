#Packages
library(ggplot2) #Used for graphics and visual representations
library(ggbiplot) #Used to generate biplot ggplot object 

#library(ggfortify)

library(caret) #Used to handle LOOCV training

library(leaps) #Used for best-subset and forward and backward stepwise selection
#with linear models 
library(bestglm) #Used for best-subset and forward and backward stepwise selection
#with generalized linear models 
library(glmnet) #Use for Ridge Regression and Lasso
library(broom) #To create tidy objects for ggplot visualization

library(ISLR)

library(pls)

rdseed=8466 #Seed to replicate results


#Experiment 1
print("Experiment 1")

#Importing data and removing NA values
Hitters=na.omit(Hitters)

#Number of observations
n=nrow(Hitters)

####Question 1.a####

#Exploring the data
summary(Hitters) #Different scales observed, standardizing the data is recommended


####Question 1.b####
#Separating features and creating dummy variables
y = Hitters$Salary
x = model.matrix(Salary ~ ., Hitters)[, -1]

#Standardizing the predictors
x=scale(x)


#pca_x=prcomp(x)

pca_x=prcomp(x)

#Scores
scores=pca_x$x

#Sample covariance matrix
var_scores=cov(scores)



#Percent of Variance explained
diag(var_scores)/sum(diag(var_scores))
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
#With 5 principal components we explained close to 84% of the variation


plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = 'b')


####Question 1.c####

#Loading vectors of first 2 PCs
loadvec12=pca_x$rotation[,1:2]

#Correlations of quantitative variables and the first two principal components
pca_x$rotation[,1]*var_scores[1,1]
pca_x$rotation[,2]*var_scores[2,2]

df_corr=data.frame(PC1=pca_x$rotation[,1]*var_scores[1,1],PC2=pca_x$rotation[,2]*var_scores[2,2])

#Scores
head(pca_x$x[,1:2])


#covariance of the standardized quantitative variables with the two components.
cov(pca_x$x[,1],x)
cov(pca_x$x[,2],x)
cov(pca_x$x[,1:2],x)

var_scores[1,1]*pca_x$rotation[,1]
var_scores[2,2]*pca_x$rotation[,2]
c(var_scores[1,1],var_scores[2,2])*pca_x$rotation[,1:2]



#correlation of the standardized quantitative variables with the two components.
cor(pca_x$x[,1],x)
cor(pca_x$x[,2],x)
cor(pca_x$x[,1:2],x)

var_scores[1,1]*pca_x$rotation[,1]/(sd(pca_x$x[,1])*sd(x))



#biplot
ggbiplot(pca_x)

pca_x$rotation
biplot(pca_x, scale=0)


#Experiment 2
print("Experiment 2")

####Question 2.a####
# Standardizing data is recommended because otherwise the range of values in each feature will act as a weight when determining how to cluster data, which is typically undesired.

####Question 2.b####

# If Euclidean distance is used, then
# shoppers who have bought very few items overall (i.e. infrequent users of
#                                                  the online shopping site) will be clustered together. This may not be desirable.
# On the other hand, if correlation-based distance is used, then shoppers
# with similar preferences (e.g. shoppers who have bought items A and B but never items C or D) will be clustered together, even if some shoppers with
# these preferences are higher-volume shoppers than others. Therefore, for
# this application, correlation-based distance may be a better choice

####Question 2.c####

hc.x = hclust(dist(x), method = "complete")

plot(hc.x, main = "Complete Linkage", xlab = "", sub = "", 
     cex = 0.7,labels = FALSE)

hc2=cutree(hc.x, 2)

#Indeces of cluster 1
c1.hc=which(hc2==1)

apply(x[c1.hc,],2,mean)
apply(x[-c1.hc,],2,mean)

#Mean of salaries by cluster
mean(y[c1.hc])
mean(y[-c1.hc])



####Question 2.d####


set.seed(rdseed)
km2 = kmeans(x, 2, nstart = 20)

#Cluster means
km2$centers

#Indeces of cluster 1
c1.km=which(km2$cluster==1)

apply(x[km2$cluster==1,],2,mean)
apply(x[km2$cluster==2,],2,mean)


#Mean of salaries by cluster
mean(y[c1.km])
mean(y[-c1.km])


km2$withinss
km2$tot.withinss
km2$totss
km2$betweenss
sum(diag(var(x))*(nrow(x)-1))


km2$betweenss/km2$totss


#Experiment 3
print("Experiment 3")

#Dataframe to track errors of each model
error.df=data.frame(Full=0,PCR=0,PLS=0,RidgeReg=0)

####Question 3.a####

reg_hitters=Hitters
reg_hitters$logSalary=log(Hitters$Salary)
reg_hitters$Salary=NULL

# nm=names(reg_hitters)
# 
# reg_hitters=cbind(model.matrix(logSalary ~ ., reg_hitters)[, -1],reg_hitters$logSalary)
# 
# reg_hitters[,-c(14,15,19)]=scale(reg_hitters[,-c(14,15,19)])
# 
# reg_hitters=as.data.frame(reg_hitters)
# 
# names(reg_hitters)=nm

#m.full=lm(logSalary~.,data=reg_hitters)

control=trainControl(method = "LOOCV")

#LOOCV on full regression model
m_fullloocv = train(logSalary~.,
                    data = reg_hitters,
                    preProcess=c("center","scale"),
                    method = "lm",
                    trControl = control)

#Estimated MSE
error.df$Full=m_fullloocv$results$RMSE^2

####Question 3.b####
pcr.fit=pcr(logSalary ~ ., data = reg_hitters, center=TRUE,scale = TRUE, validation = "LOO")

error.df$PCR=mean(pcr.fit$residuals^2)


####Question 3.c####
pls.fit=plsr(logSalary ~ ., data = reg_hitters, center=TRUE,scale = TRUE, validation = "LOO")

error.df$PLS=mean(pls.fit$residuals^2)


####Question 3.d####
lambdas = 10^seq(10, -3, length = 100)

#Fitting ridge regression models for the different lambdas
m.ridge = glmnet(x, log(y), alpha = 0, lambda = lambdas)

#Applying LOOCV to find best lambda
cv.ridge = cv.glmnet(x, log(y), alpha = 0, nfolds = n, 
                     lambda = lambdas, grouped = FALSE, type.measure = "mse")

cv.ridge$lambda.min

#Tidy data frames to graph
tidy_cv <- tidy(cv.ridge)
glance_cv <- glance(cv.ridge)

#Plot of MSE as a function of lambda
g.ridge = ggplot(tidy_cv, aes(lambda, estimate))+
  geom_point(color="red") + scale_x_log10()+
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = .25)+
  geom_vline(xintercept = glance_cv$lambda.min)+
  labs(x="Lambda",y="MSE")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("MSE vs Lambda, Method Ridge Regression")

print(g.ridge)

#Estimated MSE
error.df$RidgeReg=min(cv.ridge$cvm)