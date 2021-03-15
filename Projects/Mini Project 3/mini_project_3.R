#Packages
library(ggplot2) #Used for graphics and visual representations
library(GGally) #Used for visualization of pairs
library(ggpubr) #Used for graphics handling
library(ggcorrplot) #Used for visualization of correlations


library(fastDummies) #Used for One hot encoding categorical data
library(MASS) #Used for LD and QD analysis
library(pROC) #Used to obtain ROC curve and find cutoff
library(caret) #Used to obtain Confusion matrix, classification metrics, train controls for CV and LOOCV
library(boot)
library(e1071)

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

#contrasts(diabetes$Outcome)

####Question 1.a####

#Exploring data
print(table(diabetes$Outcome) )
#We can notice how the data is unbalanced for our classes

#Visualizing Boxplots of several predictors
g_1=ggplot(diabetes,aes(Outcome,Pregnancies,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g_2=ggplot(diabetes,aes(Outcome,Glucose,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g_3=ggplot(diabetes,aes(Outcome,BloodPressure,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g_4=ggplot(diabetes,aes(Outcome,SkinThickness,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
print(ggarrange(g_1, g_2, g_3, g_4, ncol = 2, nrow = 2))

g_5=ggplot(diabetes,aes(Outcome,Insulin,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g_6=ggplot(diabetes,aes(Outcome,BMI,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
g_7=ggplot(diabetes,aes(Outcome,DiabetesPedigreeFunction,fill=Outcome))+
  geom_boxplot()+  theme(legend.position = "none")
g_8=ggplot(diabetes,aes(Outcome,Age,fill=Outcome))+geom_boxplot()+
  theme(legend.position = "none")
print(ggarrange(g_5, g_6, g_7, g_8, ncol = 2, nrow = 2))

#From the previous boxplots we consider Glucose, Pregnancies and BMI are good candidates
#predictors for our model
print(ggplot(diabetes,aes(x=Glucose,y=Pregnancies,color=Outcome))+geom_point()+
        theme(legend.position = "none"))

print(ggplot(diabetes,aes(x=Glucose,y=BMI,color=Outcome))+geom_point()+
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


#m_Final=glm(Outcome~Pregnancies + Glucose + BloodPressure + Insulin + BMI + DiabetesPedigreeFunction + Age,data=diabetes,family = binomial)
#m_aic=step(m_Full)

####Question 1.c####

#Equation logit(p(Y|X=x))=x^t*beta
#x^t*beta=-8.0273146 + 0.1263707*Pregnancies + 0.0336810*Glucose -0.0095806*BloodPressure
#         -0.0012123*Insulin + 0.0778743*BMI + 0.8894946*DiabetesPedigreeFunction + 0.0128944*Age

summary(m1)
confint(m1)

exp(m1$coefficients)


#Experiment 2
print("Experiment 2")

####Question 2.a####

m_Full=glm(Outcome~.,data=diabetes,family = binomial)
lr.prob = m_Full$fitted.values
lr.class = ifelse(lr.prob >= 0.5, 1, 0)

# CM=confusionMatrix(lr.class, actual, positive="1")
# print(CM$table)

CM=table(lr.class, actual,dnn=c("predicted","actual"))

acc_train=sum(diag(CM))/sum(CM)
print(paste("Misclassifcation Error Rate:",1-acc_train))

sens=CM[2,2]/sum(CM[,2])
  
spec=CM[1,1]/sum(CM[,1])

####Question 2.b####

n=nrow(diabetes)

error_LOOCV=1:n

for(i in 1:n){
  m_i=glm(Outcome~.,data=diabetes[-i,],family = binomial)
  
  lr_i_class=as.factor(ifelse(predict(m_i,diabetes[i,],type = "response") >= 0.5, 1, 0))
  
  error_LOOCV[i]= lr_i_class != diabetes$Outcome[i]
}

mean_error_LOOCV=mean(error_LOOCV)

####Question 2.c####

control=trainControl(method = "LOOCV", number = 1)

mloocv = train(
  form = Outcome ~ .,
  data = diabetes,
  trControl = control,
  method = "glm",
  family = "binomial"
)

mloocv_error=(1-mloocv$results[1,2])

costfunc <- function(r, pi = 0) mean(abs(r-pi) >= 0.5)
cv.err <- cv.glm(diabetes,m_Full,cost=costfunc)$delta[1]


####Question 2.d####

error.df=data.frame(Error=rep(0,4))

row.names(error.df)=c("LR","LDA","QDA","KNN")


m1_loocv = train(
  form = Outcome ~ .-SkinThickness,
  data = diabetes,
  trControl = control,
  method = "glm",
  family = "binomial"
)

m1error_loocv=(1-m1_loocv$results[1,2])

error.df["LR","Error"]=m1error_loocv

####Question 2.e####

lda_loocv = train(
  form = Outcome ~ .-SkinThickness,
  data = diabetes,
  trControl = control,
  method = "lda"
)

lda_error=(1-lda_loocv$results[1,2])

error.df["LDA","Error"]=lda_error

####Question 2.f####

qda_loocv = train(
  form = Outcome ~ .-SkinThickness,
  data = diabetes,
  trControl = control,
  method = "qda"
)

qda_error=(1-qda_loocv$results[1,2])

error.df["QDA","Error"]=qda_error

####Question 2.g####
set.seed(rdseed)
#knn_loocv = train(Outcome ~ .-SkinThickness, data=diabetes, method = "knn",trControl = control,tuneLength=1:10)
#knn_loocv = train(Outcome ~ .-SkinThickness, data=diabetes, method = "knn",trControl = control,tuneGrid=data.frame(k=1:10))
knn_loocv = train(Outcome ~ .-SkinThickness, data=diabetes, method = "knn",trControl = control,tuneGrid=data.frame(k=1))

knn_loocv_error=(1-knn_loocv$results[1,2])
error.df["KNN","Error"]=knn_loocv_error

#knntunned=tune.knn(diabetes[,-c(4,9)], diabetes$Outcome,k = 1:40, tunecontrol = tune.control(cross=n))

####Question 2.h####



#Experiment 3
print("Experiment 3")

