#Packages, data, functions and Global Variables

library(ggplot2) #Used for graphics an visual representations
library(class) #Used for knn models

training_error_rate=function(ypred,ytrue,n)
{
  sum(ypred==ytrue)/n
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
  Error_df$trn_Error[i]=training_error_rate(trn_pred,trn_y,n_trn)
  
  
  tst_pred=knn(trn,tst,cl=trn_y,k=i)
  Error_df$tst_Error[i]=training_error_rate(tst_pred,tst_y,n_tst)
  
}

#Question 1.b

g1=ggplot(Error_df, aes(x=k_rate)) + 
  geom_line(aes(y = trn_Error), color = "darkred") + 
  geom_line(aes(y = tst_Error), color="steelblue", linetype="twodash") 