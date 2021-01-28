library(rstudioapi)    
setwd(dirname(getActiveDocumentContext()$path))
#Packages, data, functions and Global Variables

library(ggplot2) #Used for graphics an visual representations
library(class) #Used for KNN models

set.seed(8467) #Fixing a seed to replicate results in case of a tie on KNN

classification_error_rate=function(ypred,ytrue)
{
  a=mean(ypred!=ytrue)
 
}


#Top value of k for our experiment
topK=200

#Reading training and testing data set
trn=read.csv("1-training_data.csv")
tst=read.csv("1-test_data.csv")

#Number of training and testing observations
n_trn=nrow(trn)
n_tst=nrow(tst)

#Converting the classes to factors
trn_y=trn$y
tst_y=tst$y

#Dropping the classes to use the training and data sets on the knn function.
trn$y=NULL
tst$y=NULL

#Dataframe to track the errors
Error_df=data.frame(k=seq(1,topK),k_rate=1/seq(1,topK),trn_Error=rep(0,topK),tst_Error=rep(0,topK))

##########################



#Question 1.a
for(i in 1:topK)
{
  trn_pred=knn(trn,trn,cl=trn_y,k=i)
  Error_df$trn_Error[i]=classification_error_rate(trn_pred,trn_y)
  
  
  tst_pred=knn(trn,tst,cl=trn_y,k=i)
  Error_df$tst_Error[i]=classification_error_rate(tst_pred,tst_y)
  
}

#Question 1.b
g1=ggplot(data=Error_df, aes(x=k_rate,y=trn_Error))
g1 +   geom_line(aes(y = trn_Error), color = "blue")+
   geom_line(aes(y = tst_Error), color="orange")

g1

g2=ggplot(data=Error_df, aes(x=k,y=trn_Error))
g2 +   geom_line(aes(y = trn_Error), color = "blue")+
  geom_line(aes(y = tst_Error), color="orange")

g2

#Question 1.c

ind_optimalK=which.min(Error_df$tst_Error)
optimalK=Error_df$k[ind_optimalK]
optK_trnError=Error_df$trn_Error[ind_optimalK]
optK_tstError=Error_df$tst_Error[ind_optimalK]


#Question 1.d

bestK=knn(trn,trn,cl=trn_y,k=optimalK,prob = T )
prob <- attr(bestK, "prob")
prob = ifelse(prob=="1", prob, 1-prob)
x1 <- 1:10
x2 <- 1:100
prob_matrix = matrix(prob, length(x1), length(x2))
contour(x1,x2,prob_matrix,levels=0.5, labels="", xlab="", ylab="", main=
          "Some Picture", lwd=2, axes=FALSE)
gd <- expand.grid(x=x1, y=x2)
points(gd, pch="o", cex=1.2, col=ifelse(prob==1, "blue", "orange"))
box()
