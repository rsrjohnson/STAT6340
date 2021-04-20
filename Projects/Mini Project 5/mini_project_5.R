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

####Question 1.a####

#Exploring the data
summary(Hitters[,-c(14,15,19,20)])

df.stats=data.frame(Mean=apply(Hitters[,-c(14,15,19,20)], 2, mean),
                    SD=apply(Hitters[,-c(14,15,19,20)], 2, sd))

#Different scales observed, standardizing the data is recommended
print(df.stats)

####Question 1.b####
#Separating features and creating dummy variables
y = Hitters$Salary
x = model.matrix(Salary ~ ., Hitters)[, -1]

#Carrying out PCA
pca_x=prcomp(x,center = TRUE, scale = TRUE)

#Variance and Percent of Variance explained
pc_var=pca_x$sdev^2
pve=pc_var/sum(pc_var)

print(data.frame(Variance=pc_var, PVE=pve,CumPVE=cumsum(pve)))


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

#Correlations of quantitative variables and the first two principal components
df_corr=data.frame(PC1=pca_x$rotation[-c(14,15,19),1]*pca_x$sdev[1],
                   PC2=pca_x$rotation[-c(14,15,19),2]*pca_x$sdev[2])
print(df_corr)

#Scores
head(pca_x$x[,1:2])

#Biplot
autoplot(pca_x,loadings=TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 3,loadings.label.repel=TRUE)



#Experiment 2
print("Experiment 2")

####Question 2.a####
#Included on report

####Question 2.b####
#Included on report

####Question 2.c####

#Standardizing the data
x.std=scale(x)

x4clust=x.std

#Changing the names of players for an id for graphing pruposes
row.names(x4clust)=1:n

#Hierarchical clustering
hc.x = hclust(dist(x4clust), method = "complete")

#Subset of labels to graph
labs=hc.x$order[seq(1,n,3)]

#Use ids as labels
ids=rownames(x4clust)
#Eliminating labels not present in our selection
ids[ !(ids %in% labs) ] = ""

#Dendrogram
plot(hc.x,main = "Complete Linkage", cex = 0.5,labels=ids,xlab="Index of Player",sub="",hang = -1)


#Cutting at a height for two clusters
hc2=cutree(hc.x, 2)

#Indexes of cluster 1
c1.hc=which(hc2==1)

#Mean of standardized variables by cluster
df.means.std=data.frame(C1.std=apply(x.std[c1.hc,],2,mean),C2.std=apply(x.std[-c1.hc,],2,mean))

#Mean of variables by cluster
df.means=data.frame(C1=apply(x[c1.hc,],2,mean),C2=apply(x[-c1.hc,],2,mean))

#Mean of Salary by cluster
df.Sal.std=data.frame(C1.std=mean(scale(y)[c1.hc]),C2.std=mean(scale(y)[-c1.hc]))
row.names(df.Sal.std)="Salary"

#Mean of Salary by cluster
df.Sal=data.frame(C1=mean(y[c1.hc]),C2=mean(y[-c1.hc]))
row.names(df.Sal)="Salary"

#Displaying quantitative variables only
print(cbind(rbind(df.means,df.Sal),rbind(df.means.std,df.Sal.std))[-c(14,15,19),])

#Some Clusters Visualization
lab=ifelse(hc2==1,"1","2")

#Salary vs CRuns
ggdf1=data.frame(CRuns=Hitters$CRuns,Salary=Hitters$Salary,Cluster=lab)
print(ggplot(data=ggdf1,aes(x=CRuns,y=Salary,color=Cluster))+geom_point()+
        geom_point(data=data.frame(CRuns=c(df.means["CRuns","C1"],
                                           df.means["CRuns","C2"]),
                                   Salary=c(df.Sal$C1,df.Sal$C2),
                                   Cluster=c("1","2")),
                   colour=c("red","blue"),size=3,shape=8))

#Salary vs CRBI
ggdf2=data.frame(CRBI=Hitters$CRBI,Salary=Hitters$Salary,Cluster=lab)
print(ggplot(data=ggdf2,aes(x=CRBI,y=Salary,color=Cluster))+geom_point()+
        geom_point(data=data.frame(CRBI=c(df.means["CRBI","C1"],
                                           df.means["CRBI","C2"]),
                                   Salary=c(df.Sal$C1,df.Sal$C2),
                                   Cluster=c("1","2")),
                   colour=c("red","blue"),size=3,shape=8))


####Question 2.d####

#Fixing seed for kmeans
set.seed(rdseed)
km2 = kmeans(x.std, centers=2, nstart = 20)

#Cluster means (standardized)
km2$centers

#Indexes of cluster 1
c1.km=which(km2$cluster==1)

#Mean of standardized variables by cluster
df.means.stdK=data.frame(C1.std=km2$centers[1,],C2.std=km2$centers[2,])

#Mean of variables by cluster
df.meansK=data.frame(C1=apply(x[c1.km,],2,mean),C2=apply(x[-c1.km,],2,mean))

#Mean of Salary by cluster
df.Sal.stdK=data.frame(C1.std=mean(scale(y)[c1.km]),C2.std=mean(scale(y)[-c1.km]))
row.names(df.Sal.std)="Salary"

#Mean of salaries by cluster
df.SalK=data.frame(C1=mean(y[c1.km]),C2=mean(y[-c1.km]))
row.names(df.SalK)="Salary"

#Displaying quantitative variables only
print(cbind(rbind(df.meansK,df.SalK),rbind(df.means.stdK,df.Sal.stdK))[-c(14,15,19),])


#Some Clusters Visualization
lab=ifelse(hc2==1,"1","2")
labkm=ifelse(km2$cluster==1,"1","2")



#Salary vs CRuns
ggdf3=data.frame(CRuns=Hitters$CRuns,Salary=Hitters$Salary,Cluster=labkm)
print(ggplot(data=ggdf3,aes(x=CRuns,y=Salary,color=Cluster))+geom_point()+
        geom_point(data=data.frame(CRuns=c(df.meansK["CRuns","C1"],
                                          df.meansK["CRuns","C2"]),
                                   Salary=c(df.SalK$C1,df.SalK$C2),
                                   Cluster=c("1","2")),
                   colour=c("red","blue"),size=3,shape=8))
        

dfK=
#Salary vs CRBI
ggdf4=data.frame(CRBI=Hitters$CRBI,Salary=Hitters$Salary,Cluster=labkm)
print(ggplot(data=ggdf4,aes(x=CRBI,y=Salary,color=Cluster))+geom_point()+
        geom_point(data=data.frame(CRBI=c(df.meansK["CRBI","C1"],
                                          df.meansK["CRBI","C2"]),
                                   Salary=c(df.SalK$C1,df.SalK$C2),
                                   Cluster=c("1","2")),
                   colour=c("red","blue"),size=3,shape=8))


#Experiment 3
print("Experiment 3")

#Dataframe to track errors of each model
error.df=data.frame(MSE=rep(0,4))
row.names(error.df)=c("Full","PCR","PLS","RidgeReg")

####Question 3.a####

control=trainControl(method = "LOOCV")

#LOOCV on full regression model
m_fullloocv = train(log(Salary)~.,
                    data = Hitters,
                    preProcess=c("scale"),
                    method = "lm",
                    trControl = control)

#Estimated MSE
error.df["Full","MSE"]=m_fullloocv$results$RMSE^2

####Question 3.b####
pcr.fit=pcr(log(Salary) ~ ., data = Hitters,scale = TRUE, center=TRUE,
            validation = "LOO")

#Dataframe to create validation plot
df.msep1=data.frame(Components=0:19,MSEP=MSEP(pcr.fit)$val[1, 1,])

#Validation Plot
val1=ggplot(df.msep1, aes(x=Components,y=MSEP))+geom_line()+geom_point()

#Optimal M = 16
M_pcr=which.min(MSEP(pcr.fit)$val[1, 1,])
print(val1+geom_point(data=df.msep1[M_pcr,],colour="red"))

#Estimated MSE
error.df["PCR","MSE"]=MSEP(pcr.fit)$val[1, 1,M_pcr]

#Best Model 16 Components
pcr.fitBest=pcr(log(Salary) ~ ., data = Hitters,scale = TRUE, center=TRUE,
                validation = "LOO",ncomp=M_pcr-1)
summary(pcr.fitBest)

####Question 3.c####
pls.fit=plsr(log(Salary) ~ ., data = Hitters,scale = TRUE, center=TRUE,
             validation = "LOO")

#Dataframe to create validation plot
df.msep2=data.frame(Components=0:19,MSEP=MSEP(pls.fit)$val[1, 1,])

#Validation Plot
val2=ggplot(df.msep2,aes(x=Components,y=MSEP))+geom_line()+geom_point()

#Optimal M = 12
M_pls=which.min(MSEP(pls.fit)$val[1, 1,])
print(val2+geom_point(data=df.msep2[M_pls,],colour="red"))

#Estimated MSE
error.df["PLS","MSE"]=MSEP(pls.fit)$val[1, 1,M_pls]

#Best Model 12 Components
pls.fitBest=plsr(log(Salary) ~ ., data = Hitters,scale = TRUE, center=TRUE, 
                 validation = "LOO",ncomp=M_pls-1)
summary(pls.fitBest)


####Question 3.d####
lambdas = 10^seq(10, -3, length = 100)

#Applying LOOCV to find best lambda
cv.ridge = cv.glmnet(x, log(y), alpha = 0, nfolds = n, 
                     lambda = lambdas, grouped = FALSE, type.measure = "mse")

#Best lambda
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
error.df["RidgeReg","MSE"]=min(cv.ridge$cvm)

####Question 3.e####

print(error.df)