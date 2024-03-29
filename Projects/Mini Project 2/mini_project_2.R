#Packages
library(ggplot2) #Used for graphics and visual representations
library(GGally) #Used for visualization of pairs
library(ggpubr) #Used for graphics handling
library(ggcorrplot) #Used for visualization of correlations


library(fastDummies) #Used for One hot encoding categorical data
library(MASS) #Used for LD and QD analysis
library(pROC) #Used to obtain ROC curve and find cutoff
library(caret) #Used to obtain Confusion matrix and classification metrics


rdseed=8467 #Seed to replicate results in case of a tie on LDA or QDA

#Experiment 1
print("Experiment 1")

#Reading the data
raw_data=read.delim("wine.txt")

wine=raw_data

#Converting Region to factor
wine$Region=as.factor(wine$Region)

####Question 1.a####

#Exploring correlations

#We want now to visualize correlations. 
#Since regions is a factor variable, to be able to find the correlations with it
#we need to one hot encode the data. 
#We proceed to one hot encode Region leaving one class out. 
#We manipulate a copy of wine for this purpose.

wine_dummy=wine
#One hot encode Region leaving one class out
wine_dummy=dummy_cols(wine_dummy, select_columns = "Region",
                      remove_first_dummy = TRUE)
wine_dummy$Region=NULL #dropping the Region factors

#Placing Quality as last variable
temp=wine_dummy$Quality
wine_dummy$Quality=NULL
wine_dummy$Quality=temp

#Finding correlation matrix
corr1 = round(cor(wine_dummy), 2)

print(ggcorrplot(corr1,lab=TRUE,type = "full"))

#Exploring scatterplots
print(ggpairs(wine[,1:6], upper=list(continuous="cor"),axisLabels="internal"))
#As expected from the correlation matrix we can notice that the predictors 
#Aroma, Body and Flavor exhibit a positive linear relation with Quality. 

#Exploring distribution by region
g_region=ggplot(wine,aes(Region,Quality,fill=Region))+geom_boxplot()
print(g_region)

####Question 1.b####

#Exploring boxplot of Quality to identify possible outliers
print(ggplot(wine ,aes(y=Quality))+
        geom_boxplot(fill="gray64", color="black",
                     outlier.colour = "red", outlier.shape = 1))
#No visible outliers detected

#Exploring Histogram 
print(ggplot(wine, aes(x=Quality)) + 
        geom_histogram(col="black",fill="salmon",breaks=7:17, 
          position="identity", alpha=0.5))
#The quality distribution looks approximately normal.

#At this moment we consider that transformation is not required and 
#we will check model assumptions once we find an appropriate model


####Question 1.c####

#Fitting one simple regression model for each predictor
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

#Visualizing some models
g_Aroma=ggplot(m_Aroma,aes(x=Aroma,y=Quality))+geom_point()+
  geom_line(aes(y = .fitted),size=1,color="green")

g_Body=ggplot(m_Body,aes(x=Body,y=Quality))+geom_point()+
  theme(axis.title.y = element_blank())+
  geom_line(aes(y = .fitted),size=1,color="blue")
  
g_Flavor=ggplot(m_Flavor,aes(x=Flavor,y=Quality))+geom_point()+
  theme(axis.title.y = element_blank())+
  geom_line(aes(y = .fitted),size=1,color="red")
 
#Visualizing significant continuous predictors
print(ggarrange(g_Aroma, g_Body, g_Flavor,
      labels = c("Quality vs Aroma", "Quality vs Body", "Quality vs Flavor"),
      ncol = 3, nrow = 1))


####Question 1.d####

#Fitting the full model
m_Full=lm(Quality~.,data=wine)
summary(m_Full)


####Question 1.e####

#Dropping one predictor at a time
m1=lm(Quality~.-Clarity,data=wine) #dropping Clarity
summary(m1)

m2=lm(Quality~.-Clarity-Body,data=wine) #dropping Clarity and Body
summary(m2)

m3=lm(Quality~.-Clarity-Body-Aroma,data=wine) #dropping Clarity, Body and Aroma
summary(m3)

m4=lm(Quality~.-Clarity-Body-Aroma-Oakiness,data=wine) #dropping Clarity,
                                                       #Body, Aroma and Oakiness
summary(m4)

#Finally on m4 all predictors are statistically significant 
#for the response variable

print(anova(m4,m_Full))


#Exploring interactions
m_final=lm(Quality~Flavor + Region,data=wine)
summary(m_final)
m_final2=lm(Quality~Flavor + Region + Flavor:Region,data=wine)
summary(m_final2)

print(anova(m_final,m_final2))
#With a p-value of 0.3378, we fail to reject the null hypothesis, 
#this is all additional predictors are 0, meaning that 
#the interactions between Flavor and Region are not meaningful for our model


#We can appreciate the lines of our final model for each region.
print(ggplot(m_final, aes(Flavor, Quality,color=Region)) +
  geom_point()+geom_line(aes(y = .fitted),size=1, show.legend =F)+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Final Model: Quality ~ Flavor + Region"))
  

#Verifying model assumptions:

#Residual plot
gR=ggplot(m_final,aes(m_final$fitted.values, m_final$residuals)) +
  geom_point() +
  geom_hline(yintercept=0,color="gray")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Residuals vs Fitted")+
  labs(y="Residuals", x = "Fitted Values")

#QQ plot
gQ=ggplot(m_final, aes(sample = .stdresid))+ stat_qq() + stat_qq_line()+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Normal Q-Q Plot")+
  labs(y="Sample Quantiles", x = "Theoretical Quantiles")


#Time series plot of residuals
#Data Frame to plot Residuals Time Series
df.resid=data.frame(Index=1:nrow(wine),Residuals=m_final$residuals)

gT=ggplot(df.resid,aes(Index,Residuals))+geom_line()+
  geom_hline(yintercept=0,color="gray",linetype="dashed")+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("Residuals Time Series")
  
print(ggarrange(gR, gQ, gT,
                ncol = 3, nrow = 1))

####Question 1.f####

print(m_final)



####Question 1.g####

#Predicted for mean Flavor and Region 1
predicted=predict(m_final,data.frame(Flavor=mean(wine$Flavor),
                                     Region=as.factor(1)))

#95% prediction intervals
print(predict (m_final ,data.frame(Flavor=mean(wine$Flavor),
                                   Region=as.factor(1)),
               interval ="prediction"))

#95% confidence intervals
print(predict (m_final ,data.frame(Flavor=mean(wine$Flavor)
                                   ,Region=as.factor(1)),
               interval ="confidence"))


#########################

#Experiment 2
print("Experiment 2")

#Reading the data
admission = read.csv("admission.csv")
admission$Group=as.factor(admission$Group)

#Building training and testing sets
k1=which(admission$Group==1)
k2=which(admission$Group==2)
k3=which(admission$Group==3)

adm.test=rbind(admission[k1,][1:5,],
               admission[k2,][1:5,],
               admission[k3,][1:5,])

adm.train=rbind(admission[k1,][-(1:5),],
                admission[k2,][-(1:5),],
                admission[k3,][-(1:5),])


####Question 2.a####

#Exploring importance of predictors for response

#Scatterplot
g=ggplot(adm.train,aes(x=GPA,y=GMAT,color=Group))+geom_point()

#Boxplot of GPA by groups
g_GPA=ggplot(adm.train,aes(Group,GPA,fill=Group))+geom_boxplot()+
  theme(legend.position = "none")

#Boxplot of GMAT by groups
g_GMAT=ggplot(adm.train,aes(Group,GMAT,fill=Group))+geom_boxplot()

print(ggarrange(g+theme(legend.position = "none"),g_GPA, g_GMAT,
          labels = c("GMAT vs GPA","GPA by Groups", "GMAT by Groups"),
          ncol = 3, nrow = 1))



####Question 2.b####

#Fitting LDA
mlda = lda(Group ~ GPA + GMAT, data = adm.train)

#Representing the decision boundary over the Training Set
#Creating grid 
x1=seq(min(adm.train[,1]),max(adm.train[,1]),length.out=100)
x2=seq(min(adm.train[,2]),max(adm.train[,2]),length.out=100)
grid = expand.grid(x=x1, y=x2)

#Classifying the grid
names(grid)=names(adm.train)[1:2]
grid.pred.lda = predict(mlda, grid)

prob = grid.pred.lda$posterior
prob12=pmax(prob[,1],prob[,2]) #using maximum probabilities over Group 1 and 2 

#Data Frame to generate the surface for the contour
df_contour.lda=data.frame(x=grid[,1],y=grid[,2],z=prob12)

#Adding the classes to the predicted classes to the grid
grid$Group=grid.pred.lda$class

#Plotting the training set and decision boundary
glda=ggplot(adm.train,aes(x=GPA,y=GMAT,color=Group))+
  geom_point(aes(x=GPA, y=GMAT, color=Group), size=3, shape=1)+
  geom_contour(aes(x=x,y=y,z=z), data=df_contour.lda,size=1,
               color="purple",breaks = 0.5)+
  geom_point(aes(x=GPA, y=GMAT, color=Group),data=grid,alpha=0.5,size=0.5)+
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("LDA Decision Boundary")
print(glda)

#Predicted classes for Training Set
pred_train_lda=predict(mlda, adm.train)$class

#LDA Confusion Matrix for Training Set
M1=table(pred_train_lda,adm.train$Group,dnn=c("predicted","actual"))
print(M1)
acc_train_lda=sum(diag(M1))/sum(M1)
print(paste("Misclassifcation Rate Training LDA:",1-acc_train_lda))

#Predicted classes for Testing Set
pred_test_lda=predict(mlda, adm.test)$class

#LDA Confusion Matrix for Testing Set
M2=table(pred_test_lda,adm.test$Group,dnn=c("predicted","actual"))
print(M2)
acc_test_lda=sum(diag(M2))/sum(M2)
print(paste("Misclassifcation Rate Test LDA:",1-acc_test_lda))


####Question 2.c####

#Fitting QDA
mqda = qda(Group ~ GPA + GMAT, data = adm.train)

#Representing the decision boundary over the Training Set
#Creating grid 
grid.pred.qda = predict(mqda, grid)

prob = grid.pred.qda$posterior
prob12=pmax(prob[,1],prob[,2]) #using maximum probabilities over Group 1 and 2 
                               
#Data Frame to generate the surface for the contour
df_contour.qda=data.frame(x=grid[,1],y=grid[,2],z=prob12)

#Adding the classes to the predicted classes to the grid
grid$Group=grid.pred.qda$class

#Plotting the training set and decision boundary
gqda=ggplot(adm.train,aes(x=GPA,y=GMAT,color=Group))+
  geom_point(aes(x=GPA, y=GMAT, color=Group), size=3, shape=1)+
  geom_contour(aes(x=x,y=y,z=z), data=df_contour.qda,size=1,
               color="gold1",breaks = 0.5)+
  geom_point(aes(x=GPA, y=GMAT, color=Group),data=grid,alpha=0.5,size=0.5)+ 
  theme(plot.title=element_text(hjust=0.5))+
  ggtitle("QDA Decision Boundary")
print(gqda)

#Predicted classes for Training Set
pred_train_qda=predict(mqda, adm.train)$class

#QDA Confusion Matrix for Training Set
M3=table(pred_train_qda,adm.train$Group,dnn=c("predicted","actual"))
print(M3)
acc_train_qda=sum(diag(M3))/sum(M3)
print(paste("Misclassifcation Rate Training QDA:",1-acc_train_qda))

#Predicted classes for Testing Set
pred_test_qda=predict(mqda, adm.test)$class

#QDA Confusion Matrix for Testing Set
M4=table(pred_test_qda,adm.test$Group,dnn=c("predicted","actual"))
print(M4)
acc_test_qda=sum(diag(M4))/sum(M4)
print(paste("Misclassifcation Rate Testing LDA:",1-acc_test_qda))


####Question 2.d####
#Included in report


########################


#Experiment 3

print("Experiment 3")

#Reading the data
diabetes = read.csv("diabetes.csv")

#Renaming predictors
names(diabetes)=c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin",   "BMI",  
                  "DiabetesPedigreeFunction", "Age","Outcome" )

#Converting Outcome to factor
diabetes$Outcome=as.factor(diabetes$Outcome)


####Question 3.a####

#Exploring data
print(table(diabetes$Outcome) )
#We can notice how the data is unbalanced for our classes

#Visualizing Boxplots of several predictors
g1=ggplot(diabetes,aes(Outcome,Pregnancies,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g2=ggplot(diabetes,aes(Outcome,Glucose,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g3=ggplot(diabetes,aes(Outcome,BloodPressure,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g4=ggplot(diabetes,aes(Outcome,SkinThickness,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
print(ggarrange(g1, g2, g3, g4, ncol = 2, nrow = 2))

g5=ggplot(diabetes,aes(Outcome,Insulin,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g6=ggplot(diabetes,aes(Outcome,BMI,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g7=ggplot(diabetes,aes(Outcome,DiabetesPedigreeFunction,fill=Outcome))+
  geom_boxplot()+  theme(legend.position = "none")
g8=ggplot(diabetes,aes(Outcome,Age,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
print(ggarrange(g5, g6, g7, g8, ncol = 2, nrow = 2))

#From the previous boxplots we consider Glucose and Age are relevant predictors
print(ggplot(diabetes,aes(x=Glucose,y=Age,color=Outcome))+geom_point()+
        theme(legend.position = "none"))


#Storing true classes
actual=diabetes$Outcome

####Question 3.b####

#Fitting LDA
m_diabetes_lda=lda(Outcome~., data=diabetes)

#Predicted Classes
predicted_lda=predict(m_diabetes_lda,diabetes)

#Confusion Matrix for LDA
CM1=confusionMatrix(predicted_lda$class, actual, positive="1")
print(CM1$table)


#ROC Curve for LDA
roc.lda = roc(diabetes$Outcome, predicted_lda$posterior[, "1"], 
              levels = c("0", "1"),direction="<")


print(ggroc(roc.lda,color = "salmon"  ,legacy.axes = TRUE)+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black", 
               linetype="dashed")+ theme(plot.title=element_text(hjust=0.5))+
  ggtitle(paste("Linear Discriminant Analysis AUC:",round(roc.lda$auc,4))))

####Question 3.c####

#Fitting QDA
m_diabetes_qda=qda(Outcome~., data=diabetes)

#Predicted Classes
predicted_qda=predict(m_diabetes_qda,diabetes)

#Confusion Matrix for QDA
CM2=confusionMatrix(predicted_qda$class, actual, positive="1")
print(CM2$table)


#ROC Curve for QDA
roc.qda = roc(diabetes$Outcome, predicted_qda$posterior[, "1"], 
              levels = c("0", "1"),direction="<")


print(ggroc(roc.qda, color="turquoise3",legacy.axes = TRUE)+
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="black",
               linetype="dashed")+ theme(plot.title=element_text(hjust=0.5))+
    ggtitle(paste("Quadratic Discriminant Analysis AUC:",round(roc.qda$auc,4))))


#Visualizing Error, Sensitivity and Specificity of both models.
df.tab=data.frame("Error Rate" =c(1-CM1$overall[["Accuracy"]],
                                  1-CM2$overall[["Accuracy"]]), 
                  "Sensitivity"=c(CM1$byClass[["Sensitivity"]],
                                  CM2$byClass[["Sensitivity"]]),
                  "Specificity"=c(CM1$byClass[["Specificity"]],
                                  CM2$byClass[["Specificity"]]))
row.names(df.tab)=c("LDA","QDA")

print(df.tab)

####Question 3.d####

#Finding new cutoff
cut.lda=coords(roc.lda,"best") #best cutoff determine by Youden's Index

#Classifying using the new Cutoff
pred_newcutoff=ifelse(predicted_lda$posterior[,2]>=cut.lda[1,1],"1","0")
pred_newcutoff=as.factor(pred_newcutoff)

#New Confusion Matrix
CM3=confusionMatrix(pred_newcutoff, actual, positive="1")
print(CM3$table)

df.tab.new=data.frame("Error Rate" =1-CM3$overall[["Accuracy"]], 
                  "Sensitivity"=CM3$byClass[["Sensitivity"]],
                  "Specificity"=CM3$byClass[["Specificity"]])
row.names(df.tab.new)="New Cutoff"

print(df.tab.new)
