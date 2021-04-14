#Packages
library(ggplot2) #Used for graphics and visual representations
library(ggdendro)
library(factoextra)
library(plotly)
library(ggfortify)  #Used to generate biplot ggplot object 

library(caret) #Used to handle LOOCV training

library(glmnet) #Use for Ridge Regression and Lasso
library(broom) #To create tidy objects for ggplot visualization

library(ISLR)

library(pls) #Used for PCR and PLS.

rdseed=8466 #Seed to replicate results


#Experiment 1
print("Experiment 1")

#Importing data and removing NA values
Hitters=na.omit(Hitters)

#Number of observations
n=nrow(Hitters)

####Question 1.a####

#Exploring the data
summary(Hitters) 
apply(Hitters[,-c(14,15,20)], 2, mean)
apply(Hitters[,-c(14,15,20)], 2, sd)
#Different scales observed, standardizing the data is recommended


####Question 1.b####
#Separating features and creating dummy variables
y = Hitters$Salary
x = model.matrix(Salary ~ ., Hitters)[, -1]

#Standardizing
x.std=scale(x)


#pca_x=prcomp(x)

#Carrying out PCA
pca_x=prcomp(x.std,center = FALSE, scale = FALSE)

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

print(g_pve) #5 principal components seems to be appropriate


g_cum=ggplot(data.frame(PC=1:ncol(x),CumPVE=cumsum(pve)),aes(x=PC,y=CumPVE))+geom_line(size=1)+
  geom_point(shape=1,size=2)+
  xlab("Principal Component")+  ylab("Cumulative Proportion of Variance Explained")

print(g_cum)
#With 5 principal components we explained close to 84% of the variation


####Question 1.c####

#Loading vectors of first 2 PCs
loadvec12=pca_x$rotation[,1:2]



#Correlations of variables and the first two principal components
df_corr=data.frame(PC1=pca_x$rotation[,1]*pca_x$sdev[1],PC2=pca_x$rotation[,2]*pca_x$sdev[2])

#Scores
head(pca_x$x[,1:2])

#correlation of the standardized quantitative variables with the two components.
t(cor(pca_x$x[,1:2],x.std))



autoplot(pca_x,loadings=TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 3,loadings.label.repel=T)

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

x4clust=x.std

row.names(x4clust)=1:n

hc.x = hclust(dist(x4clust), method = "complete")

plot(hc.x, main = "Complete Linkage", xlab = "", sub = "", 
     cex = 0.35)

set.seed(rdseed)
toPlot <- sample(rownames(x4clust), size=floor(n/3))

## use rownames as labels
labels <- rownames(x4clust)
## clear labels not present in toPlot
labels[ !(labels %in% toPlot) ] <- ""





plot(hc.x, cex = 0.5,labels=labels)

plot(hc.x, hang=-1,cex=0.3)

dhc <- as.dendrogram(hc.x)
# Rectangular lines
ddata <- dendro_data(dhc, type = "rectangle")
p <- ggplot(segment(ddata)) + 
  geom_segment()
  #coord_flip()+ scale_y_reverse(expand = c(0.2, 0))

ggdendrogram(hc.x, rotate = FALSE, size = 2,)


plot_dendro(dhc,height = 1600, width = 800) %>% 
  hide_legend() %>% 
  highlight(persistent = TRUE, dynamic = TRUE)



ptest <- ggplot(dhc, horiz = FALSE, theme = NULL)
plotly_build(ptest)


fviz_dend(hc.x,repel=FALSE,cex = 0.5, k = 2, color_labels_by_k = TRUE)


hc2=cutree(hc.x, 2)

lab=ifelse(hc2==1,"1","2")

ggdf1=data.frame(CRuns=x.std[,11],CRBI=x.std[,12],Cluster=lab)

ggplot(data=ggdf1,aes(x=CRuns,y=CRBI,color=Cluster))+geom_point()

ggdf2=data.frame(CAtBat=x.std[,8],CHits=x.std[,9],Cluster=lab)

ggplot(data=ggdf2,aes(x=CAtBat,y=CHits,color=Cluster))+geom_point()

#Indeces of cluster 1
c1.hc=which(hc2==1)

apply(x[c1.hc,],2,mean)
apply(x[-c1.hc,],2,mean)

#Mean of salaries by cluster
mean(y[c1.hc])
mean(y[-c1.hc])



####Question 2.d####


set.seed(rdseed)
km2 = kmeans(x.std, 2, nstart = 20)

#Cluster means
km2$centers

#Indexes of cluster 1
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
labkm=ifelse(km2$cluster==1,"1","2")

ggdf1=data.frame(CRuns=x.std[,11],CRBI=x.std[,12],Cluster=labkm)

ggplot(data=ggdf1,aes(x=CRuns,y=CRBI,color=Cluster))+geom_point()

ggdf2=data.frame(CAtBat=x.std[,8],CHits=x.std[,9],Cluster=labkm)

ggplot(data=ggdf2,aes(x=CAtBat,y=CHits,color=Cluster))+geom_point()




#Experiment 3
print("Experiment 3")

#Dataframe to track errors of each model
error.df=data.frame(Full=0,PCR=0,PLS=0,RidgeReg=0)

####Question 3.a####

# reg_hitters=Hitters
# reg_hitters$logSalary=log(Hitters$Salary)
# reg_hitters$Salary=NULL

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
m_fullloocv = train(log(Salary)~.,
                    data = Hitters,
                    preProcess=c("scale"),
                    method = "lm",
                    trControl = control)

#Estimated MSE
error.df$Full=m_fullloocv$results$RMSE^2

####Question 3.b####
pcr.fit=pcr(log(Salary) ~ ., data = Hitters,scale = TRUE, validation = "LOO")

summary(pcr.fit)

#Validation Plot
ggplot(data.frame(Components=0:19,MSEP=MSEP(pcr.fit)$val[1, 1,]),
       aes(x=Components,y=MSEP))+geom_line()+geom_point()

M_pcr=which.min(MSEP(pcr.fit)$val[1, 1,])
error.df$PCR=MSEP(pcr.fit)$val[1, 1,M_pcr]

pcr.fitBest=pcr(log(Salary) ~ ., data = Hitters,scale = TRUE, validation = "LOO",ncomp=M_pcr-1)
summary(pcr.fitBest)


####Question 3.c####
pls.fit=plsr(log(Salary) ~ ., data = Hitters,scale = TRUE, validation = "LOO")

#Validation Plot
ggplot(data.frame(Components=0:19,MSEP=MSEP(pls.fit)$val[1, 1,]),
       aes(x=Components,y=MSEP))+geom_line()+geom_point()

M_pls=which.min(MSEP(pls.fit)$val[1, 1,])
error.df$PLS=MSEP(pls.fit)$val[1, 1,M_pls]

pls.fitBest=plsr(log(Salary) ~ ., data = Hitters,scale = TRUE, validation = "LOO",ncomp=M_pls-1)
summary(pls.fitBest)


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