library(class) #Used for KNN models
library(data.table)
library(dplyr)


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

fitKNN_Error=function(testing_set,true_labels,i)
{
  set.seed(rdseed)
  classification_error_rate(knn(trn,testing_set,cl=trn_y,k=i),true_labels)
}



library(microbenchmark)
mbm3 <- microbenchmark("for loop"={Error_df.loop=data.frame(k=kvals,trn_Error=kvals,tst_Error=kvals)

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
  
}},
"data.table"={Error.dt=data.table(kval=kvals)

Error.dt[,trn_Error:=fitKNN_Error(trn,trn_y,kval),by=.(kval)]


Error.dt[,tst_Error:=fitKNN_Error(tst,tst_y,kval),by=.(kval)]},
"sapply"={Error_df.sappy=data.frame(k=kvals)

Error_df.sappy$trn_Error = sapply(kvals, function(i){
  set.seed(rdseed)
  trn_pred = knn(trn,trn,cl=trn_y,k=i)
  
  (classification_error_rate(trn_pred,trn_y))
  
})

Error_df.sappy$tst_Error = sapply(kvals, function(i){
  set.seed(rdseed)
  tst_pred = knn(trn,tst,cl=trn_y,k=i)
  
  (classification_error_rate(tst_pred,tst_y))
  
})},
"dplyr"={Error.tidy=data.frame(kval=kvals)

Error.tidy=Error.tidy%>%group_by(kval)%>%mutate(trn_Error=fitKNN_Error(trn,trn_y,kval))

Error.tidy=Error.tidy%>%group_by(kval)%>%mutate(tst_Error=fitKNN_Error(tst,tst_y,kval))},times=5)


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



fitKNN_Error_cifar=function(i)
{
  set.seed(rdseed)
  classification_error_rate(knn(x.train,x.test,cl=y.train,k=i),y.test)
}



cifarmbm=microbenchmark("for loop"={
  Cifar_Error=data.frame(k=kvals2,tst_Error=kvals2)
  
  for(i in 1:length(kvals2))
  {
    #Fitting KNN for testing data
    set.seed(rdseed)
    tst_pred=knn(x.train,x.test,cl=y.train,k=kvals2[i])
    Cifar_Error$tst_Error[i]=classification_error_rate(tst_pred,y.test)
    
  }
},
"data.table"={Cifar_Error.dt=data.table(kval=kvals2)

Cifar_Error.dt[,tst_Error:=fitKNN_Error_cifar(kval),by=.(kval)]},

"sapply"={cifarError_df.sapply=data.frame(k=kvals2)

cifarError_df.sapply$trn_Error = sapply(kvals2, function(i){
  set.seed(rdseed)
  tst_pred = knn(x.train,x.test,cl=y.train,k=i)
  
  (classification_error_rate(tst_pred,y.test))
  
})

},

"dplyr"={Error_cifar.tidy=data.frame(kval=kvals2)

Error_cifar.tidy=Error_cifar.tidy%>%group_by(kval)%>%mutate(tst_Error=fitKNN_Error_cifar(kval))
},times=3)
