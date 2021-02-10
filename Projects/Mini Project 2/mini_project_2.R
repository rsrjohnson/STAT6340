#Packages
library(ggplot2) #Used for graphics an visual representations

rdseed=8467 #Seed to replicate results in case of a tie on KNN

#Experiment 1

raw_data=read.delim("wine.txt")

wine=raw_data

wine$Region=as.factor(wine$Region)

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

g_Region=ggplot(wine,aes(x=Region,y=Quality))+geom_point()
g_Flavor=ggplot(wine,aes(x=Flavor,y=Quality))+geom_point()+geom_smooth(method = "lm")


#Flavor, body, aroma, region, are important


#Question 1.d
m_Full=lm(Quality~.,data=wine)
summary(m_Full)

