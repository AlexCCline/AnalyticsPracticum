##############################################################################
#_____________________________Logistic Regression_____________________________
#load data sets and view
Fund <- read.csv("~/CSDA 5330 Alex Cline/Fundraising.csv")
View(Fund)
FutFund <- read.csv("~/CSDA 5330 Alex Cline/FutureFundraising.csv")
View(FutFund)
#____________________________Data Preparation_________________________________

#50-50 sampling
table(Fund$TARGET_B)

#mean=13
#Do NOT drop TARGET_D for this step
tapply(Fund$TARGET_D, Fund$TARGET_B, mean)

#0.63
netppdonors <- mean(Fund$TARGET_D[Fund$TARGET_B==1]-0.68)*0.051
#-0.65
netppnondonors <- mean(Fund$TARGET_D[Fund$TARGET_B==0]-0.68)*.949


#drop rowID columns and TARGET_D
drop <- c("ï..RowId","RowId.","TARGET_D") 
Fund = Fund[,!(names(Fund) %in% drop)]

#correlation matrix and plot
library(corrplot)
corr1 <- cor(Fund)
corrplot(corr1)

#____________________________Data Partitioning_________________________________
#60-40 training and test partitioning
library(tidyverse)
library(caret)
set.seed(12345)
samF<-createDataPartition(Fund$TARGET_B, p=0.6, list = FALSE)
trainF<-Fund[samF,]
testF<-Fund[-samF,]

#_____________________________Model Selection____________________________________
#model for all variables
logit.regF <- glm(trainF$TARGET_B~., data = trainF, family = "binomial")
options(scipen=999)
summary(logit.regF)
logit.regF

#ANOVA test for ChiSq
anova(logit.regF, test = "Chisq")

logit.regF2 <- glm(trainF$TARGET_B~LASTGIFT+totalmonths+NUMPROM, data = trainF, family = "binomial")
options(scipen=999)
summary(logit.regF2)
logit.regF2

#_______________________________Prediction________________________________________

install.packages("pscl")
#McFadden R2 best fit
library("pscl")
pR2(logit.regF)
pR2(logit.regF2)

predictF<-predict(logit.regF2, testF)
pF1<-data.frame(predictF, testF)
pF1


#Write new file for predicted Logistic Regression
PredictedF <-data.frame(predictF, testF)
write.csv(PredictedF, file = "FutureFundraising1.csv")




cm_lgF <- table(predictF, testF$TARGET_B)
row.names(cm_lgF) <- c("Actual: 0", "Actual: 1")
colnames(cm_lgF) <- c("Predicted: 0", "Predicted: 1")
cm_lgF <- addmargins(A= cm_lgF, FUN = list(Total = sum), quiet = TRUE)
cm_lgF

#______________________________Validation__________________________________

#####Test for Accuracy
(cm_lgF[1,1]+cm_lgF[2,2])/(cm_lgF[1,1]+cm_lgF[1,2]+cm_lgF[2,1]+cm_lgF[2,2])
#####Error Rate
1-((cm_lgF[1,1]+cm_lgF[2,2])/(cm_lgF[1,1]+cm_lgF[1,2]+cm_lgF[2,1]+cm_lgF[2,2]))
#####Sensitivity
cm_lgF[1,1]/(cm_lgF[1,1]+cm_lgF[2,1])
#####Specificity
cm_lgF[2,2]/(cm_lgF[2,2]+cm_lgF[1,2])
####Calculate Precision
cm_lgF[1, 1]/(cm_lgF[1, 1] + cm_lgF[1, 2])


#____________________Visualization & Model Performance________________________

library(ROCR)
p <- predict(logit.regF, newdata=testF, type="response")
pr <- prediction(p, testF$TARGET_B)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

p2 <- predict(logit.regF2, newdata=testF, type="response")
pr2 <- prediction(p2, testF$TARGET_B)
prf2 <- performance(pr2, measure = "tpr", x.measure = "fpr")
plot(prf2)

auc2 <- performance(pr2, measure = "auc")
auc2 <- auc2@y.values[[1]]
auc2


library(gains)
gain <- gains(testF$TARGET_B, predictF, groups = length(predictF))

plot(c(0, gain$cume.pct.of.total*sum(testF$TARGET_B))
     ~c(0,gain$cume.obs),xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0, sum(testF$TARGET_B))~c(0, dim(testF)[1]), lty=2)

heights<- gain$mean.resp/mean(testF$TARGET_B)
midpoints<- barplot(heights, names.arg = gain$depth, ylim = c(0,9),
                    xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise lift chart")
text(midpoints, heights+0.5, labels=round(heights,1), cex = 0.8)





########################################################################################

#___________________Neural Network Using Logistic Regression______________________
#load data sets and view
Fund <- read.csv("~/CSDA 5330 Alex Cline/Fundraising.csv")
View(Fund)
FutFund <- read.csv("~/CSDA 5330 Alex Cline/FutureFundraising.csv")
View(FutFund)
#____________________________Data Preparation_________________________________
#drop rowID columns and TARGET_D
drop <- c("ï..RowId","RowId.","TARGET_D") 
Fund = Fund[,!(names(Fund) %in% drop)]

library(tidyverse)
library(neuralnet)
glimpse(Fund)

#normalize data for NN
library(caret)
preProcess(Fund)

#___________________________Data Partitioning___________________________________
library(caret)
set.seed(12345)
samFNN<-createDataPartition(Fund$TARGET_B, p=0.6, list = FALSE)
trainFNN<-Fund[samFNN,]
testFNN<-Fund[-samFNN,]


#________________________Model Selection________________________________________
set.seed(12345)
NN1 <- neuralnet(TARGET_B ~ ., data = trainFNN, hidden = 1, act.fct = "logistic", rep = 10, err.fct = "sse")

plot(NN1, rep = 'best')
NN1$result.matrix


set.seed(12345)
NN2 <- neuralnet(TARGET_B ~ totalmonths+INCOME+Icavg, data = trainFNN, hidden = 2, act.fct = "logistic", rep = 10, err.fct = "sse")

plot(NN2, rep = "best")
NN2$result.matrix

#___________________________Prediction_________________________________________

#Test the resulting output
temp_test <- subset(testFNN, select = c("totalmonths","INCOME","Icavg"))
head(temp_test)
nn.results <- compute(NN2, temp_test)
results <- data.frame(actual = testFNN$TARGET_B, prediction = nn.results$net.result)

roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

results <- data.frame(actual = testFNN$TARGET_B, prediction = nn.results$net.result)
results


predictFNN<-predict(NN2, testFNN)
pFNN1<-data.frame(predictFNN, testFNN)
pFNN1

#Write new file for predicted NN
PredictedFNN <-data.frame(predictFNN, testFNN)
write.csv(PredictedF, file = "FutureFundraising2.csv")


cm_lgFNN <- table(predictFNN, testFNN$TARGET_B)
row.names(cm_lgFNN) <- c("Actual: 0", "Actual: 1")
colnames(cm_lgFNN) <- c("Predicted: 0", "Predicted: 1")
cm_lgFNN <- addmargins(A= cm_lgFNN, FUN = list(Total = sum), quiet = TRUE)
cm_lgFNN

#______________________________Validation_______________________________________


#####Test for Accuracy
(cm_lgFNN[1,1]+cm_lgFNN[2,2])/(cm_lgFNN[1,1]+cm_lgFNN[1,2]+cm_lgFNN[2,1]+cm_lgFNN[2,2])
#####Error Rate
1-((cm_lgFNN[1,1]+cm_lgFNN[2,2])/(cm_lgFNN[1,1]+cm_lgFNN[1,2]+cm_lgFNN[2,1]+cm_lgFNN[2,2]))
#####Sensitivity
cm_lgFNN[1,1]/(cm_lgFNN[1,1]+cm_lgFNN[2,1])
#####Specificity
cm_lgFNN[2,2]/(cm_lgFNN[2,2]+cm_lgFNN[1,2])
####Calculate Precision
cm_lgFNN[1, 1]/(cm_lgFNN[1, 1] + cm_lgFNN[1, 2])

#_______________________Visualization & Model Performance_____________________

#may need to detach neuralnet package
#detach(package:neuralnet)


library(ROCR)
pNN <- predict(NN1, newdata=testFNN, type="response")
prNN <- prediction(pNN, testFNN$TARGET_B)
prfNN <- performance(prNN, measure = "tpr", x.measure = "fpr")
plot(prfNN)

aucNN <- performance(prNN, measure = "auc")
aucNN <- aucNN@y.values[[1]]
aucNN

pNN2 <- predict(NN2, newdata=testFNN, type="response")
prNN2 <- prediction(pNN2, testFNN$TARGET_B)
prfNN2 <- performance(prNN2, measure = "tpr", x.measure = "fpr")
plot(prfNN2)

aucNN2 <- performance(prNN2, measure = "auc")
aucNN2 <- aucNN2@y.values[[1]]
aucNN2


library(gains)
gain <- gains(testFNN$TARGET_B, predictFNN, groups = length(predictFNN))

plot(c(0, gain$cume.pct.of.total*sum(testFNN$TARGET_B))
     ~c(0,gain$cume.obs),xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0, sum(testFNN$TARGET_B))~c(0, dim(testFNN)[1]), lty=2)

heights<- gain$mean.resp/mean(testFNN$TARGET_B)
midpoints<- barplot(heights, names.arg = gain$depth, ylim = c(0,9),
                    xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise lift chart")
text(midpoints, heights+0.5, labels=round(heights,1), cex = 0.8)

#_______________________________________________________________________________




#net profit gain chart for both models

net.profit <- ifelse(testFNN$TARGET_B == 1, (13-0.68)/9.8, -0.68/0.53)
prob.rank <- order(predictF, decreasing = T)
plot(cumsum(net.profit[prob.rank]), xlab = "Probability Rank", ylab = "profit", type = "l", col = "red")
prob.rank <- order(predictFNN, decreasing = T)
lines(cumsum(net.profit[prob.rank]), xlab = "Probability Rank", ylab = "profit", col = "blue")
















