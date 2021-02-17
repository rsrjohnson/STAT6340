#Packages
library(ggplot2) #Used for graphics an visual representations
library(GGally) #Visualization of correlations and pairs
library(ggpubr)
library(ggcorrplot) #Visualization of correlations
library(fastDummies)

rdseed=8467 #Seed to replicate results in case of a tie on KNN

#Experiment 1

#Reading the data
raw_data=read.delim("wine.txt")

wine=raw_data

#Converting Region to factor
wine$Region=as.factor(wine$Region)

#Question 1.a

#Exploring correlations

#we want now to visualize correlations. 
#Since regions is a factor variable, to be able to establish correlations
#we need to one hot encode the data. We proceed to leave one out hot encode
#a copy of wine for this purpose

wine_std=wine
wine_std[,1:6]=scale(wine_std[,1:6])

wine_dummy=wine
wine_dummy=dummy_cols(wine_dummy, select_columns = "Region",remove_first_dummy = TRUE)
wine_dummy$Region=NULL

temp=wine_dummy$Quality
wine_dummy$Quality=NULL
wine_dummy$Quality=temp

corr1 = round(cor(wine_dummy), 2)

#ggcorr(wine_dummy,label = TRUE,label_round = 2)

ggcorrplot(corr1,lab=TRUE,type = "full")
#Visualizing the correlation matrix we can see that predictors Oakiness and Clarity have a very low correlation with Quality
#Therefore it is likely that this variables will not provide much information to predict Quality

#we infer that the variables Oakiness and Clarity are not much relevant to the model
#and are the most likely to be removed. 
#Also variable Body presents a positive correlation not as high as other predictors, 
#therefore we required further analysis to determine the relevance of this predictor for the response variable

#Another point to notice that predictors Aroma and Body are are highly with Flavor, this can cause overfitting issues, 
#so with further analysis one or some of this predictors might be dropped since they can be explained by the other predictor.

ggpairs(wine[,1:6], upper=list(continuous="points"),axisLabels="internal")
#As expected from the correlation matrix we can notice that the Predictors Aroma, Body and Flavor exhibit a positive linear relation with Quality. 



#Question 1.b

m_Flavor= lm(Quality~Flavor,data = wine)
# Residual plot

# QQ plot

# Time series plot of residuals


wine$logQuality=sqrt(wine$Quality)

m_transform=lm(logQuality~Flavor,data=wine)

# Residual plot
plot(fitted(m_Flavor), resid(m_Flavor))
abline(h = 0)
plot(fitted(m_transform), resid(m_transform))
abline(h = 0)
# QQ plot
qqnorm(resid(m_Flavor))
qqnorm(resid(m_transform))
# Time series plot of residuals
plot(resid(m_Flavor), type="l")
abline(h=0)
plot(resid(m_transform), type="l")
abline(h=0)

g_Flavor=ggplot(wine,aes(x=Flavor,y=Quality))+geom_point()+
  geom_smooth(method = "lm",se=FALSE,color="green")

g_Flavor_log=ggplot(wine,aes(x=Flavor,y=logQuality))+geom_point()+
  geom_smooth(method = "lm",se=FALSE,color="green")

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

#We can observe that the Flavor and Region predictors We can reject the null hypothesis for predictor Flavor. Even 
