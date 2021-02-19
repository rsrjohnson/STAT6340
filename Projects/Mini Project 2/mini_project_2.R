#Packages
library(ggplot2) #Used for graphics and visual representations
library(GGally) #Visualization of pairs
library(ggpubr) #Graphics handling
library(ggcorrplot) #Visualization of correlations

library(fastDummies) #One hot encoding categorical data
library(MASS) #Used for ld and qd analysis
library(pROC) #Used to obtain ROC curve and find cutoff
library(caret) #Confusion matrix and classification metrics


rdseed=8467 #Seed to replicate results in case of a tie on LDA or QDA

#Experiment 1

#Reading the data
raw_data=read.delim("wine.txt")

wine=raw_data

#Converting Region to factor
wine$Region=as.factor(wine$Region)

#Question 1.a

#Exploring correlations

#We want now to visualize correlations. 
#Since regions is a factor variable, to be able to establish correlations with it
#we need to one hot encode the data. We proceed to leave one class out hot encode Region
#we manipulate a copy of wine for this purpose

wine_std=wine
#wine_std[,1:6]=scale(wine_std[,1:6])

wine_dummy=wine
wine_dummy=dummy_cols(wine_dummy, select_columns = "Region",remove_first_dummy = TRUE)
wine_dummy$Region=NULL #dropping the Region factors

#Placing Quality as last variable
temp=wine_dummy$Quality
wine_dummy$Quality=NULL
wine_dummy$Quality=temp

#Finding correlation matrix
corr1 = round(cor(wine_dummy), 2)


ggcorrplot(corr1,lab=TRUE,type = "full")
#Visualizing the correlation matrix we can see that predictors Oakiness and Clarity have a very low correlation with Quality
#Therefore it is likely that this variables will not provide much information to predict Quality

#we infer that the variables Oakiness and Clarity are not much relevant to the model
#and are the most likely to be removed. 
#Also variable Body presents a positive correlation not as high as other predictors, 
#therefore we required further analysis to determine the relevance of this predictor for the response variable

#Another point to notice it that predictors Aroma, Body, Flavor and Region are highly correlated. This can cause overfitting issues, 
#so with further analysis one or some of this predictors might be dropped since they can be explained by the others.

ggpairs(wine[,1:6], upper=list(continuous="cor"),axisLabels="internal")
#As expected from the correlation matrix we can notice that the Predictors Aroma, Body and Flavor exhibit a positive linear relation with Quality. 

g_region=ggplot(wine,aes(Region,Quality,fill=Region))+geom_boxplot()


#Question 1.b

#m_Flavor= lm(Quality~Flavor,data = wine)
# Residual plot

# QQ plot

# Time series plot of residuals


# wine$logQuality=sqrt(wine$Quality)
# 
# m_transform=lm(logQuality~Flavor,data=wine)
# 
# # Residual plot
# plot(fitted(m_Flavor), resid(m_Flavor))
# abline(h = 0)
# plot(fitted(m_transform), resid(m_transform))
# abline(h = 0)
# # QQ plot
# qqnorm(resid(m_Flavor))
# qqnorm(resid(m_transform))
# # Time series plot of residuals
# plot(resid(m_Flavor), type="l")
# abline(h=0)
# plot(resid(m_transform), type="l")
# abline(h=0)
# 
# g_Flavor=ggplot(wine,aes(x=Flavor,y=Quality))+geom_point()+
#   geom_smooth(method = "lm",se=FALSE,color="green")
# 
# g_Flavor_log=ggplot(wine,aes(x=Flavor,y=logQuality))+geom_point()+
#   geom_smooth(method = "lm",se=FALSE,color="green")

#Standardization and log transform
# wine_dummy_std=wine_std
# wine_dummy_std=dummy_cols(wine_dummy_std, select_columns = "Region",remove_first_dummy = TRUE)
# wine_dummy_std$Region=NULL
# 
# temp=wine_dummy_std$Quality
# wine_dummy_std$Quality=NULL
# wine_dummy_std$Quality=temp
# corr_std = round(cor(wine_dummy_std), 2)
# ggcorrplot(corr_std,lab=TRUE,type = "full")
# 
# 
# wine_log=log(wine[,1:6])
# corr_log=round(cor(wine_log), 2)
# ggcorrplot(corr_log,lab=TRUE,type = "full")


#Question 1.c

m_Clarity=lm(Quality~Clarity,data=wine)

m_Aroma=lm(Quality~Aroma,data=wine)

m_Body=lm(Quality~Body,data=wine)

m_Flavor=lm(Quality~Flavor,data=wine)

m_Oakiness=lm(Quality~Oakiness,data=wine)

m_Region=lm(Quality~Region,data=wine)

summary(m_Clarity)
summary(m_Aroma)
summary(m_Body)
summary(m_Flavor)
summary(m_Oakiness)
summary(m_Region)

#There is a statistically significant association between the predictor
#and the response on all models except for the predictors Clarity and Oakiness. 
#This is consistent with our previous exploration.


#g_Region=ggplot(wine,aes(x=Region,y=Quality))+geom_point()

g_Aroma=ggplot(wine,aes(x=Aroma,y=Quality))+geom_point()+
  geom_smooth(method = "lm",se=FALSE,color="red")

g_Body=ggplot(wine,aes(x=Body,y=Quality))+geom_point()+
  geom_smooth(method = "lm",se=FALSE,color="blue")
  
g_Flavor=ggplot(wine,aes(x=Flavor,y=Quality))+geom_point()+
  geom_smooth(method = "lm",se=FALSE,color="green")
 

ggarrange(g_Aroma, g_Body, g_Flavor,
                    labels = c("Quality vs Aroma", "Quality vs Body", "Quality vs Flavor"),
                    ncol = 3, nrow = 1)

#For the predictor Region, we will explore additional graphics when we study interactions effects.


#Question 1.d
m_Full=lm(Quality~.,data=wine)
summary(m_Full)

#We can observe that we can reject the null hypothesis for the Flavor and Region predictors.
#We will proceed to drop one variable at a time and recheck the p-values.

#Question 1.e

m1=lm(Quality~.-Clarity,data=wine) #dropping Clarity
summary(m1)

m2=lm(Quality~.-Clarity-Body,data=wine) #dropping Clarity and Body
summary(m2)

m3=lm(Quality~.-Clarity-Body-Aroma,data=wine) #dropping Clarity, Body and Aroma
summary(m3)

m4=lm(Quality~.-Clarity-Body-Aroma-Oakiness,data=wine) #dropping Clarity, Body, Aroma and Oakiness
summary(m4)

#Finally on m4 all predictors all are statistically significant for the response variable

anova(m4,m_Full)

#The anova test confirms our findings. With a p-value of 0.6528, we fail to reject the null hypothesis, this is all additional predictors are 0.


#Exploring interactions

m_final=lm(Quality~Flavor + Region,data=wine)
summary(m_final)
m_final2=lm(Quality~Flavor + Region + Flavor:Region,data=wine)
summary(m_final2)

anova(m_final,m_final2)
#With a p-value of 0.3378, we fail to reject the null hypothesis, this is all additional predictors are 0,
#meaning the interactions between Flavor and Region are not meaningful for our model
#Also notice that the gain on the adjusted R-squared is very small, another evidence that these extra predictors add overfit to our model.

#Question 1.f

#model Quality = Intercept +  Flavor + Region2 + Region3
#      Quality = 7.0943 + (1.1155)Flavor + -1.5335(Region2) + 1.2234(Region3)



#Question 1.g


k=which(wine$Region=="1")
predict1=m_final$fitted.values[k]


#ggplot(wine,aes(x=Flavor,y=Quality,color=Region))+geom_point()+geom_line(aes(x=wine$Flavor[k],y=predict1),data=wine[k,])

ggplot(wine,aes(x=Flavor,y=Quality,color=Region))+geom_point()+geom_abline(slope=m_final$coefficients[2],intercept = m_final$coefficients[1] )


#95% prediction intervals
predict (m_final ,wine[k,], se.fit = TRUE,
         interval ="prediction")

#95% confidence intervals
predict (m_final ,wine[k,], se.fit = TRUE,
         interval ="confidence")

#95% prediction intervals
predict (m_final ,data.frame(Flavor=mean(wine$Flavor),Region=as.factor(1)),
         interval ="prediction")

#95% confidence intervals
predict (m_final ,wine[k,], se.fit = TRUE,
         interval ="confidence")


predict (m_final ,data.frame(Flavor=mean(wine$Flavor),Region=as.factor(1)),
         interval ="prediction")

predict (m_final ,data.frame(Flavor=mean(wine$Flavor),Region=as.factor(1)),
         interval ="confidence")



#########################

#Experiment 2

admission = read.csv("admission.csv")
admission$Group=as.factor(admission$Group)

k1=which(admission$Group==1)
k2=which(admission$Group==2)
k3=which(admission$Group==3)



adm.test=rbind(admission[k1,][1:5,],admission[k2,][1:5,],admission[k3,][1:5,])

adm.train=rbind(admission[k1,][-(1:5),],admission[k2,][-(1:5),],admission[k3,][-(1:5),])


#Question 2.a
g=ggplot(adm.train,aes(x=GPA,y=GMAT,color=Group))+geom_point()
g_GPA=ggplot(adm.train,aes(Group,GPA,fill=Group))+geom_boxplot()
g_GMAT=ggplot(adm.train,aes(Group,GMAT,fill=Group))+geom_boxplot()
ggarrange(g+theme(legend.position = "none"),g_GPA, g_GMAT,
          labels = c("GMAT vs GPA","Boxplot GPA", "Boxplot GMAT"),
          ncol = 3, nrow = 1)



#Question 2.b

mlda = lda(Group ~ GPA + GMAT, data = adm.train)

#Creating grid 
x1=seq(min(adm.train[,1]),max(adm.train[,1]),length.out=100)
x2=seq(min(adm.train[,2]),max(adm.train[,2]),length.out=100)
grid = expand.grid(x=x1, y=x2)

#Classifying the grid
names(grid)=names(adm.train)[1:2]
grid.pred = predict(mlda, grid)

prob = grid.pred$posterior
prob12=pmax(prob[,1],prob[,2]) #we just need the probabilities of max of 1 and 2 
                               #since 3 can be obtain from the others

#Data Frame to generate the surface for the contour
df_contour.lda=data.frame(x=grid[,1],y=grid[,2],z=prob12)


#Plotting the training set and decision boundary
g_lda=g + geom_contour(aes(x=x,y=y,z=z), data=df_contour.lda,size=1,colour="purple",breaks = 0.5)
print(g_lda)

pred_train_lda=predict(mlda, adm.train)$class

M1=table(pred_train_lda,adm.train$Group,dnn=c("predicted","actual"))
acc_train_lda=sum(diag(M1))/sum(M1)


pred_test_lda=predict(mlda, adm.test)$class

M2=table(pred_test_lda,adm.test$Group,dnn=c("predicted","actual"))
acc_test_lda=sum(diag(M2))/sum(M2)


#Question 2.c

mqda = qda(Group ~ GPA + GMAT, data = adm.train)

grid.pred.qda = predict(mqda, grid)

prob = grid.pred.qda$posterior
prob12=pmax(prob[,1],prob[,2]) #we just need the probabilities of max of 1 and 2 
#since 3 can be obtain from the others

#Data Frame to generate the surface for the contour
df_contour.qda=data.frame(x=grid[,1],y=grid[,2],z=prob12)


#Plotting the training set and decision boundary
g_qda=g + geom_contour(aes(x=x,y=y,z=z), data=df_contour.qda,size=1,colour="gold1",breaks = 0.5)
print(g_qda)

pred_train_qda=predict(mqda, adm.train)$class

M3=table(pred_train_qda,adm.train$Group,dnn=c("predicted","actual"))
acc_train_qda=sum(diag(M3))/sum(M3)


pred_test_qda=predict(mqda, adm.test)$class

M4=table(pred_test_qda,adm.test$Group,dnn=c("predicted","actual"))
acc_test_qda=sum(diag(M4))/sum(M4)


#Question 2.d
g_train=g+theme(legend.position = "none")+geom_contour(aes(x=x,y=y,z=z), data=df_contour.lda,size=1,colour="purple",breaks = 0.5)+
  geom_contour(aes(x=x,y=y,z=z), data=df_contour.qda,size=1,colour="gold1",breaks = 0.5)


#Creating grid 
x1=seq(min(adm.test[,1]),max(adm.test[,1]),length.out=100)
x2=seq(min(adm.test[,2]),max(adm.test[,2]),length.out=100)
grid = expand.grid(x=x1, y=x2)

#Classifying the grid
names(grid)=names(adm.train)[1:2]
grid.pred = predict(mlda, grid)

prob = grid.pred$posterior
prob12=pmax(prob[,1],prob[,2]) #we just need the probabilities of max of 1 and 2 
#since 3 can be obtain from the others

#Data Frame to generate the surface for the contour
df_contour.lda_test=data.frame(x=grid[,1],y=grid[,2],z=prob12)

grid.pred.qda = predict(mqda, grid)

prob = grid.pred.qda$posterior
prob12=pmax(prob[,1],prob[,2]) #we just need the probabilities of max of 1 and 2 
#since 3 can be obtain from the others

#Data Frame to generate the surface for the contour
df_contour.qda_test=data.frame(x=grid[,1],y=grid[,2],z=prob12)

g2=ggplot(adm.test,aes(x=GPA,y=GMAT,color=Group))+geom_point()

g_test=g2+geom_contour(aes(x=x,y=y,z=z), data=df_contour.lda_test,size=1,colour="purple",breaks = 0.5)+
  geom_contour(aes(x=x,y=y,z=z), data=df_contour.qda_test,size=1,colour="gold1",breaks = 0.5)+
  theme(legend.position = "none",axis.title.y = element_blank())


ggarrange(g_train, g_test,
          labels = c("Training Set", "Testing Set"),
          ncol = 2, nrow = 1)


########################


#Experiment 3


diabetes = read.csv("diabetes.csv")

names(diabetes)=c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin",   "BMI",  "DiabetesPedigreeFunction", "Age","Outcome" )

diabetes$Outcome=as.factor(diabetes$Outcome)



#Question 3.a

table(diabetes$Outcome)

g1=ggplot(diabetes,aes(Outcome,Pregnancies,fill=Outcome))+geom_boxplot()
g2=ggplot(diabetes,aes(Outcome,Glucose,fill=Outcome))+geom_boxplot()
g3=ggplot(diabetes,aes(Outcome,BloodPressure,fill=Outcome))+geom_boxplot()
g4=ggplot(diabetes,aes(Outcome,SkinThickness,fill=Outcome))+geom_boxplot()
ggarrange(g1, g2, g3, g4, ncol = 2, nrow = 2)

g5=ggplot(diabetes,aes(Outcome,Insulin,fill=Outcome))+geom_boxplot()
g6=ggplot(diabetes,aes(Outcome,BMI,fill=Outcome))+geom_boxplot()
g7=ggplot(diabetes,aes(Outcome,DiabetesPedigreeFunction,fill=Outcome))+geom_boxplot()
g8=ggplot(diabetes,aes(Outcome,Age,fill=Outcome))+geom_boxplot()
ggarrange(g5, g6, g7, g8, ncol = 2, nrow = 2)

ggplot(diabetes,aes(x=Glucose,y=Age,color=Outcome))+geom_point()



actual=diabetes$Outcome

#Question 3.b

m_diabetes_lda=lda(Outcome~., data=diabetes)


predicted_lda=predict(m_diabetes_lda,diabetes)

CM1=confusionMatrix(predicted_lda$class, actual, positive="1")
CM1$table
1-CM1$overall[["Accuracy"]]
CM1$byClass[["Sensitivity"]]
CM1$byClass[["Specificity"]]

roc.lda = roc(diabetes$Outcome, predicted_lda$posterior[, "1"], levels = c("0", "1"),direction="<")

ggroc(roc.lda,color = "#F8766D"  ,legacy.axes = TRUE)+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="grey", linetype="dashed")+
  ggtitle("Linear Discriminant Analysis")

#Question 3.c

m_diabetes_qda=qda(Outcome~., data=diabetes)

predicted_qda=predict(m_diabetes_qda,diabetes)

CM2=confusionMatrix(predicted_qda$class, actual, positive="1")
CM2$table
1-CM2$overall[["Accuracy"]]
CM2$byClass[["Sensitivity"]]
CM2$byClass[["Specificity"]]

roc.qda = roc(diabetes$Outcome, predicted_qda$posterior[, "1"], levels = c("0", "1"),direction="<")

ggroc(roc.qda, color="#00BFC4",legacy.axes = TRUE)+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="grey", linetype="dashed")+
  ggtitle("Quadratic Discriminant Analysis")

#Question 3.d


ggroc(list(lda=roc.lda,qda=roc.qda),legacy.axes = TRUE)+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="grey", linetype="dashed")+
  annotate("text", x = 0.5, y = 0.83, vjust = 0, label = paste("AUC =",sprintf("%.4f",roc.lda$auc)),color="#F8766D")+
  annotate("text", x = 0.5, y = 0.78, vjust = 0, label = paste("AUC =",sprintf("%.4f",roc.qda$auc)),color="#00BFC4")



roc.lda$auc
roc.qda$auc

cut.lda=coords(roc.lda,"best")
cut.qda=coords(roc.qda,"best")


pred_newcutoff=ifelse(predicted_lda$posterior[,2]>=cut.lda[1,1],"1","0")
pred_newcutoff=as.factor(pred_newcutoff)
CM3=confusionMatrix(pred_newcutoff, actual, positive="1")
CM3$table