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
diabetes = read.csv("diabetes.csv")

#Renaming predictors
names(diabetes)=c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin",   "BMI",  
                  "DiabetesPedigreeFunction", "Age","Outcome" )

#Converting Outcome to factor
diabetes$Outcome=as.factor(diabetes$Outcome)


####Question 1.a####

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

#Fitting the full model
m_Full=glm(Outcome~.,data=diabetes,family = binomial)
summary(m_Full)
m0=glm(Outcome~1,data=diabetes,family = binomial)

####Question 1.b####

#Dropping one predictor at a time
m1=glm(Outcome~.-SkinThickness,data=diabetes,family=binomial) #dropping SkinThickness
summary(m1)


anova(m1, m_Full, test = "Chisq")

anova(m0, m1, test = "Chisq")

m_aic=step(m_Full)