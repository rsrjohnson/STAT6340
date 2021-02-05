library(ggplot2) #Used for graphics an visual representations
library(class) #Used for KNN models
library(data.table)
library(tidyverse)


rdseed=8467 #Seed to replicate results in case of a tie on KNN

#Helper function to calculate the classification error rate
classification_error_rate=function(ypred,ytrue)
{
  mean(ypred!=ytrue)
}


#Setting graphic options.
graph_colors=c("blue","orange")
graph_legend=c("Training Set","Testing Set")


##########################


#Experiment 1

#Values of k for experiment 1
kvals=c(seq(1,200,5),seq(200,400,50))


#Reading training and testing data set
trn=read.csv("1-training_data.csv", stringsAsFactors = TRUE)
tst=read.csv("1-test_data.csv", stringsAsFactors = TRUE)

#Exploring the data set
str(trn)
summary(trn)

#Saving training and testing labels
trn_y=trn$y
tst_y=tst$y

#Dropping the classes to use only the predictors on the knn function.
trn$y=NULL
tst$y=NULL

start_time <- Sys.time()

#Dataframe to track the error rates
Error_df.loop=data.frame(k=kvals,trn_Error=kvals,tst_Error=kvals)

#Question 1.a
for(i in 1:length(kvals))
{
  #Fitting KNN for training data
  set.seed(rdseed)
  trn_pred=knn(trn,trn,cl=trn_y,k=kvals[i])
  Error_df.loop$trn_Error[i]=classification_error_rate(trn_pred,trn_y)

  #Fitting KNN for testing data
  set.seed(rdseed)
  tst_pred=knn(trn,tst,cl=trn_y,k=kvals[i])
  Error_df.loop$tst_Error[i]=classification_error_rate(tst_pred,tst_y)

}

end_time <- Sys.time()
print("for loop")
print(end_time - start_time)



fitKNN_Error=function(testing_set,true_labels,i)
{
  set.seed(rdseed)
  classification_error_rate(knn(trn,testing_set,cl=trn_y,k=i),true_labels)
}

start_time <- Sys.time()

Error.dt=data.table(kval=kvals)
set.seed(rdseed)
Error.dt[,trn_Error:=fitKNN_Error(trn,trn_y,kval),by=.(kval)]

set.seed(rdseed)
Error.dt[,tst_Error:=fitKNN_Error(tst,tst_y,kval),by=.(kval)]


end_time <- Sys.time()
print("data.table")
print(end_time - start_time)


start_time <- Sys.time()

Error_df=data.frame(k=kvals)

Error_df$trn_Error = sapply(kvals, function(i){
  set.seed(rdseed)
  trn_pred = knn(trn,trn,cl=trn_y,k=i)

  (classification_error_rate(trn_pred,trn_y))

})

Error_df$tst_Error = sapply(kvals, function(i){
  set.seed(rdseed)
  tst_pred = knn(trn,tst,cl=trn_y,k=i)

  (classification_error_rate(tst_pred,tst_y))

})

end_time <- Sys.time()
print("sapply")
print(end_time - start_time)

start_time <- Sys.time()

Error.tidy=data.frame(kval=kvals)

Error.tidy=Error.tidy%>%group_by(kval)%>%mutate(trn_Error=fitKNN_Error(trn,trn_y,kval))

Error.tidy=Error.tidy%>%group_by(kval)%>%mutate(tst_Error=fitKNN_Error(tst,tst_y,kval))

end_time <- Sys.time()
print("tidy")
print(end_time - start_time)

#Question 1.b

#Generating plot of training and testing error rates against k.

g=ggplot(data=Error_df, aes(x=k, y=trn_Error))+
  geom_line(aes(y=trn_Error, color=graph_legend[1]), size=1.1)+
  geom_point(color=graph_colors[1], shape=19)+
  geom_line(aes(y=tst_Error, color=graph_legend[2]), size=1.1)+
  geom_point(x=Error_df$k, y=Error_df$tst_Error, color=graph_colors[2], shape=15)+
  scale_color_manual("Legend",values=c("Training Set"=graph_colors[1], "Testing Set"=graph_colors[2]))+
  labs(title="Classification Error Rate", x="K", y="Error")+
  theme(plot.title=element_text(hjust=0.5))

print(g)


#Question 1.c

#Finidng the index of the optimal K, this is the index of the least test error rate
ind_optimalK=which.min(Error_df$tst_Error)

Error_df[ind_optimalK,] #The row of the optimal k constains the associated errors for training and testing.

optimalK=Error_df$k[ind_optimalK]


#Question 1.d

#Creating grid 
x1=seq(min(trn[,1]),max(trn[,1]),length.out=100)
x2=seq(min(trn[,1]),max(trn[,1]),length.out=100)
grid <- expand.grid(x=x1, y=x2)

#Classifying the grid
bestK=knn(trn,grid,cl=trn_y,k=optimalK,prob = TRUE )
prob <- attr(bestK, "prob")
prob = ifelse(bestK=="yes", prob, 1-prob)
prob_matrix = matrix(prob, length(x1), length(x2))

#Plotting the training set.
plot(trn, col=ifelse(trn_y=="yes", "blue", "orange"),
     main=paste("Desicion boundary for Training data K=",optimalK))

#Plotting the decision boundary.
contour(x1,x2,prob_matrix,levels=0.5, labels="", xlab="", ylab="", lwd=2, add = TRUE)


##########################

#Experiment 2

##Preprocessing
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


##Experiment 2 questions

#Question 2.a

kvals2=c(50, 100, 200, 300, 400)

#Dataframe to track the test error rates
#Cifar_Error=data.frame(k=kvals2,tst_Error=kvals2)

# for(i in 1:length(kvals2))
# {
#   #Fitting KNN for testing data
#   set.seed(rdseed)
#   tst_pred=knn(x.train,x.test,cl=y.train,k=kvals2[i])
#   Cifar_Error$tst_Error[i]=classification_error_rate(tst_pred,y.test)
#   
# }

Cifar_Error=data.frame(k=kvals2)

Cifar_Error$tst_Error = sapply(kvals2, function(i){
  set.seed(rdseed)
  tst_pred = knn(x.train,x.test,cl=y.train,k=i)

  (classification_error_rate(tst_pred,y.test))

})

#Question 2.b

#Fitting KNN with the optimal K
set.seed(rdseed)
ind_optimalK=which.min(Cifar_Error$tst_Error)
cifar_pred=knn(x.train,x.test,cl=y.train,k=ind_optimalK,prob=TRUE)

#Displaying confusion matrix
table(cifar_pred ,y.test,dnn=c("predicted","actual"))





