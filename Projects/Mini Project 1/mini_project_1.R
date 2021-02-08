#Packages
library(ggplot2) #Used for graphics an visual representations
library(class) #Used for KNN models

rdseed=8467 #Seed to replicate results in case of a tie on KNN

#Helper function to calculate the classification error rate
classification_error_rate=function(ypred,ytrue)
{
  mean(ypred!=ytrue)
}


#Setting graphic options.
error_colors=c("blue","orange")
graph_legend=c("Training Set","Testing Set")


##########################


#Experiment 1

#Values of K for experiment 1
kvals=c(seq(1,200,5),seq(200,400,25))


#Reading training and testing data sets
trn=read.csv("1-training_data.csv", stringsAsFactors = TRUE)
tst=read.csv("1-test_data.csv", stringsAsFactors = TRUE)

#Exploring the data set
str(trn)
summary(trn)

str(tst)
summary(tst)
#We can appreciate that the class labels are equally distributed
#in both training and testing sets

#Saving training and testing labels
trn_y=trn$y
tst_y=tst$y

#Dropping the classes to use only the predictors on the knn function.
trn$y=NULL
tst$y=NULL

#Dataframe to track the error rates
Error_df=data.frame(kval=kvals)

#Question 1.a

#Fitting KNN for different values of K (training data)
Error_df$trn_Error = sapply(kvals, function(i){
  set.seed(rdseed)
  trn_pred = knn(trn,trn,cl=trn_y,k=i)
  
  (classification_error_rate(trn_pred,trn_y))
})

#Fitting KNN for different values of K (testing data)
Error_df$tst_Error = sapply(kvals, function(i){
  set.seed(rdseed)
  tst_pred = knn(trn,tst,cl=trn_y,k=i)
  
  (classification_error_rate(tst_pred,tst_y))
})


#Question 1.b

#Generating plot of training and testing error rates against k.

g=ggplot(data=Error_df, aes(x=kval, y=trn_Error))+
  geom_line(aes(y=trn_Error, color=graph_legend[1]), size=1)+
  geom_point(color=error_colors[1], shape=19)+
  geom_line(aes(y=tst_Error, color=graph_legend[2]), size=1,alpha=.6)+
  geom_point(x=Error_df$kval, y=Error_df$tst_Error, color=error_colors[2], shape=15,alpha=.6)+
  scale_color_manual("Legend",values=c("Training Set"=error_colors[1], "Testing Set"=error_colors[2]))+
  labs(title="Classification Error Rate", x="K", y="Error Rate")+
  theme(plot.title=element_text(hjust=0.5))

print(g)

#Question 1.c

#Finding the index of the optimal K, this is the index of the least test error rate
ind_optimalK=which.min(Error_df$tst_Error)

Error_df[ind_optimalK,] #The row of the optimal k contains the associated errors for training and testing.

optimalK=Error_df$kval[ind_optimalK]


#Question 1.d

#Creating grid 
x1=seq(min(trn[,1]),max(trn[,1]),length.out=100)
x2=seq(min(trn[,2]),max(trn[,2]),length.out=100)
grid <- expand.grid(x=x1, y=x2)

#Classifying the grid
bestK=knn(trn,grid,cl=trn_y,k=optimalK,prob = TRUE ) #Fitting with optimal K
prob = attr(bestK, "prob")
prob = ifelse(bestK=="yes", prob, 1-prob)

#Data Frame to generate the surface for the contour
df_contour=data.frame(x=grid[,1],y=grid[,2],z=prob)
  
#Data Frame to plot the training set
df_trn=data.frame(x1=trn[,1],x2=trn[,2],classes=trn_y)

#Plotting the training set and decision boundary
g2=ggplot()+geom_point(aes(x=x1,y=x2,color=classes),data=df_trn)+
  geom_contour(aes(x=x,y=y,z=z),
               data=df_contour,size=2,colour="black",breaks = 0.5)
print(g2)

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

#Values of K for experiment 2
kvals2=c(50, 100, 200, 300, 400)

#Dataframe to track the test error rates
cifar_Error=data.frame(kval=kvals2)

##Experiment 2 questions

#Question 2.a

#Fitting KNN for different values of K
cifar_Error$Error = sapply(kvals2, function(i){
  set.seed(rdseed)
  tst_pred = knn(x.train,x.test,cl=y.train,k=i)
  
  (classification_error_rate(tst_pred,y.test))
  
})


#Question 2.b

ind_optimalK=which.min(cifar_Error$Error) #Finding index of optimal K
optimalK2=cifar_Error$kval[ind_optimalK]


#Fitting KNN with the optimal K
set.seed(rdseed)
cifar_pred=knn(x.train,x.test,cl=y.train,k=optimalK2)

#Displaying confusion matrix
table(cifar_pred ,y.test,dnn=c("predicted","actual"))


