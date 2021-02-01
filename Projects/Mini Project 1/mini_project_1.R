library(rstudioapi)    
setwd(dirname(getActiveDocumentContext()$path))
#Packages, data, functions and Global Variables

library(tidyverse)
library(ggplot2) #Used for graphics an visual representations
library(class) #Used for KNN models
library(caret) #Used for confusion matrix

set.seed(2021) #Fixing a seed to replicate results in case of a tie on KNN

classification_error_rate=function(ypred,ytrue)
{
  mean(ypred!=ytrue)
}

graph_colors=c("blue","orange")
graph_legend=c("Training Set","Testing Set")


##########################


#Experiment 1


#Value of k for experiment 1
topK=200
kvals=seq(1,topK,5)


#Reading training and testing data set
trn=read.csv("1-training_data.csv", stringsAsFactors = TRUE)
tst=read.csv("1-test_data.csv", stringsAsFactors = TRUE)

#Saving
trn_y=trn$y
tst_y=tst$y

#Dropping the classes to use the training and data sets on the knn function.
trn$y=NULL
tst$y=NULL

#Dataframe to track the errors
Error_df=data.frame(k=kvals,k_rate=1/kvals,trn_Error=kvals,tst_Error=kvals)


#Question 1.a
for(i in 1:length(kvals))
{
  trn_pred=knn(trn,trn,cl=trn_y,k=kvals[i])
  Error_df$trn_Error[i]=classification_error_rate(trn_pred,trn_y)
  
  
  tst_pred=knn(trn,tst,cl=trn_y,k=kvals[i])
  Error_df$tst_Error[i]=classification_error_rate(tst_pred,tst_y)
  
}


#Question 1.b

g=ggplot(data=Error_df, aes(x=k,y=trn_Error))

g+geom_line(aes(y=trn_Error,color=graph_legend[1]),size=1.1)+geom_point(color="blue",shape=19)+
  geom_line(aes(y=tst_Error,color=graph_legend[2]),size=1.1)+geom_point(x=Error_df$k,y=Error_df$tst_Error,color="orange",shape=15)+
  scale_color_manual("Legend",values = c("Training Set"="blue","Testing Set"="orange"))+
  labs(title="Classification Error Rate",x="K",y="Error")+
  theme(plot.title=element_text(hjust=0.5))


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
x2 <- 1:50
prob_matrix = matrix(prob, length(x1), length(x2))
contour(x1,x2,prob_matrix,levels=0.5, labels="", xlab="", ylab="", main=
          "Some Picture", lwd=2, axes=FALSE)
gd <- expand.grid(x=x1, y=x2)
points(gd, pch="o", cex=1.2, col=ifelse(prob==1, "blue", "orange"))
box()




#Experiment 2

library(keras)
cifar <- dataset_cifar10()
str(cifar)
x.train <- cifar$train$x
y.train <- cifar$train$y
x.test <- cifar$test$x
y.test <- cifar$test$y
# reshape the images as vectors (column-wise)
# (aka flatten or convert into wide format)
# (for row-wise reshaping, see ?array_reshape)
dim(x.train) <- c(nrow(x.train), 32*32*3) # 50000 x 3072
dim(x.test) <- c(nrow(x.test), 32*32*3) # 50000 x 3072
# rescale the x to lie between 0 and 1
x.train <- x.train/255
x.test <- x.test/255
# categorize the response
y.train <- as.factor(y.train)
y.test <- as.factor(y.test)
# randomly sample 1/100 of test data to reduce computing time
set.seed(2021)
id.test <- sample(1:10000, 100)
x.test <- x.test[id.test,]
y.test <- y.test[id.test]



kvals2=c(50, 100, 200, 300, 400)

Cifar_Error=data.frame(k=kvals2,k_rate=1/kvals2,tst_Error=kvals2)

#Question 2.a

for(i in 1:length(kvals2))
{
  tst_pred=knn(x.train,x.test,cl=y.train,k=kvals2[i])
  Cifar_Error$tst_Error[i]=classification_error_rate(tst_pred,y.test)
  
}

ind_optimalK=which.min(Cifar_Error$tst_Error)
cifar_pred=knn(x.train,x.test,cl=y.train,k=ind_optimalK)

confusionMatrix(table(cifar_pred ,y.test))




