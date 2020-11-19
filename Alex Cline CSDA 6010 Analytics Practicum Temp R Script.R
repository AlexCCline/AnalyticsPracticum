######################Clustering Neural Network###########################

######load dataset and glimpse data types
tx<-read.csv("Taxi-cancellation-case.csv")
View(tx)
###libraries
library(caret)
library(tidyverse)
library(dplyr)
library(magrittr)
glimpse(tx)
#####check for null values
install.packages(VIM)
library(VIM)
aggr(tx)

#################DATA PREPARATION##################################

#change NA's in to_lat and to_long to mean; change other column Na's = 0
#reinstall ggplot2 if install error for imputeTS
install.packages("imputeTS")
library(imputeTS)
tx$to_lat <- na.mean(tx$to_lat, option = "mean")
tx$to_long <- na.mean(tx$to_long, option ="mean")
tx$package_id <- na.replace(tx$package_id, fill = 0)
tx$to_area_id <- na.replace(tx$to_area_id, fill = 0)
tx$from_city_id <- na.replace(tx$from_city_id, fill = 0)
tx$to_city_id <- na.replace(tx$to_city_id, fill = 0)

##_________________ compute trip length from GPS data ________________________#
#Create the distance calculation function. This function gets: the latitude
#and longitudes of two points on each and returns the distance in kilometers
dist <- function (long1, lat1, long2, lat2){
        rad <- pi/180 #columns are
        a1 <- lat1 * rad
        a2 <- long1 * rad
        b1 <- lat2 * rad
        b2 <- long2 * rad
        dlon <- b2 - a2
        dlat <- b1 - a1
        a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
        c <- 2 * atan2(sqrt(a), sqrt(1 - a))
        R <- 6378.145
        d <- R * c
        return(d)}
dist(tx$from_long, tx$from_lat, tx$to_long, tx$to_lat)

#library to use dates for analysis
install.packages(anytime)
library(anytime)
tx$booking_created <- anytime::anydate(as.numeric(tx$booking_created))
tx$from_date <- anytime::anydate(as.numeric(tx$from_date))
tx$to_date <- anytime::anydate(as.numeric(tx$to_date))

# Separate or mutate the Date/Time columns
library(lubridate)
tx$booking_Year <- as.numeric(year(tx$booking_created))
tx$booking_Month <- as.numeric(month(tx$booking_created))
tx$booking_Day <- as.numeric(day(tx$booking_created))
tx$booking_Hour <- as.numeric(hour(tx$booking_created))
tx$booking_Minute <- as.numeric(minute(tx$booking_created))

tx$from_date_Year <- as.numeric(year(tx$from_date))
tx$from_date_Month <- as.numeric(month(tx$from_date))
tx$from_date_Day <- as.numeric(day(tx$from_date))
tx$from_date_Hour <- as.numeric(hour(tx$from_date))
tx$from_date_Minute <- as.numeric(minute(tx$from_date))

tx$to_date_Year <- as.numeric(year(tx$to_date))
tx$to_date_Month <- as.numeric(month(tx$to_date))
tx$to_date_Day <- as.numeric(day(tx$to_date))
tx$to_date_Hour <- as.numeric(hour(tx$to_date))
tx$to_date_Minute <- as.numeric(minute(tx$to_date))
#change any empty values to 0
tx$booking_Hour <- na.replace(tx$booking_Hour, fill = 0)
tx$booking_Minute <- na.replace(tx$booking_Minute, fill = 0)
tx$from_date_Hour <- na.replace(tx$from_date_Hour, fill = 0)
tx$from_date_Minute <- na.replace(tx$from_date_Minute, fill = 0)
tx$to_date_Hour <- na.replace(tx$to_date_Hour, fill = 0)
tx$to_date_Minute <- na.replace(tx$to_date_Minute, fill = 0)

####drop columns with many null values, row, and user id
drop <- c("user_id","row.") 
dfTX = tx[,!(names(tx) %in% drop)]
dfTX <- na.omit(dfTX)
glimpse(dfTX)
View(dfTX)

#normalize data
dfTX1.norm<-sapply(dfTX, scale)
row.names(dfTX1.norm)<-row.names(dfTX)

#find correlation between variables
CorDF <- cor(dfTX1.norm, method = c("pearson","kendall","spearman"))
library(corrplot)
corrplot(CorDF)


#######################DATA PARTITIONING###########################
#libraries
library(ggplot2)
library(nnet)
library(caret)
library(neuralnet)

set.seed(2020)
#Split data into index subset for training (60%) and testing (40%) instances
inTrain <- runif(nrow(dfTX)) < 0.60

#######________Predictor Model Selection______##############
#Train neural network with n nodes in the hidden layer via nnet(formula, data=d, size=n ...)
#Output variable: Car Cancellation (binary ' 1-y/0-n')
nn <- nnet(Car_Cancellation ~ ., data=dfTX[inTrain,],
           size=15, maxit=100, rang=0.1, decay=5e-4)
library(gamlss.add)
plot(nn)
#change formula for most correlated variables to Car_Cancellation and adjust node size
#to prevent overfitting of model
set.seed(2020)
nn2 <- nnet(Car_Cancellation~ to_long+to_lat, data = dfTX[inTrain,],
            size=10, maxit=100, rang=0.1, decay=5e-4)
plot(nn2)


##########_________Validation__________##############
#Predict car cancellation clusters via predict(nn, test, type="raw")
pred <- predict(nn2, dfTX[-inTrain,],
                type="raw")
#Extract confusion matrix via 
#table(pred=pred_classes, true=true_classes)
cm_nn <- table(pred, dfTX$Car_Cancellation[-inTrain])
row.names(cm_nn) <- c("Actual: 0", "Actual: 1")
colnames(cm_nn) <- c("Predicted: 0", "Predicted: 1")
cm_nn <- addmargins(A= cm_nn, FUN = list(Total = sum), quiet = TRUE)
cm_nn
############____________Performance________________################

#####Test for Accuracy
(cm_nn[1,1]+cm_nn[2,2])/(cm_nn[1,1]+cm_nn[1,2]+cm_nn[2,1]+cm_nn[2,2])
#####Error Rate
1-((cm_nn[1,1]+cm_nn[2,2])/(cm_nn[1,1]+cm_nn[1,2]+cm_nn[2,1]+cm_nn[2,2]))
#####Sensitivity
cm_nn[1,1]/(cm_nn[1,1]+cm_nn[2,1])
#####Specificity
cm_nn[2,2]/(cm_nn[2,2]+cm_nn[1,2])
####Calculate Precision
cm_nn[1, 1]/(cm_nn[1, 1] + cm_nn[1, 2])









#################################################################
#################################################################
#################################################################
############# Logistic Regression Analysis ######################


######load dataset and glimpse data types
tx<-read.csv("Taxi-cancellation-case.csv")
View(tx)
###libraries
library(tidyverse)
library(dplyr)
library(magrittr)
glimpse(tx)
#####check for null values
install.packages(VIM)
library(VIM)
aggr(tx)

#################DATA PREPARATION##################################

#change NA's in to_lat and to_long to mean; other NA's = 0
#reinstall ggplot2 if install error for imputeTS
install.packages("imputeTS")
library(imputeTS)
tx$to_lat <- na.mean(tx$to_lat, option = "mean")
tx$to_long <- na.mean(tx$to_long, option ="mean")
tx$package_id <- na.replace(tx$package_id, fill = 0)
tx$to_area_id <- na.replace(tx$to_area_id, fill = 0)
tx$from_city_id <- na.replace(tx$from_city_id, fill = 0)
tx$to_city_id <- na.replace(tx$to_city_id, fill = 0)


##_________________ compute trip length from GPS data ________________________#
#Create the distance calculation function. This function gets: the latitude
#and longitudes of two points on each and returns the distance in kilometers
dist <- function (long1, lat1, long2, lat2){
        rad <- pi/180 #columns are
        a1 <- lat1 * rad
        a2 <- long1 * rad
        b1 <- lat2 * rad
        b2 <- long2 * rad
        dlon <- b2 - a2
        dlat <- b1 - a1
        a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
        c <- 2 * atan2(sqrt(a), sqrt(1 - a))
        R <- 6378.145
        d <- R * c
        return(d)}
dist(tx$from_long, tx$from_lat, tx$to_long, tx$to_lat)

#separate or mutate Date/Time columns
install.packages(anytime)
library(anytime)
tx$booking_created <- anytime::anydate(as.numeric(tx$booking_created))
tx$from_date <- anytime::anydate(as.numeric(tx$from_date))
tx$to_date <- anytime::anydate(as.numeric(tx$to_date))

# Separate or mutate the Date/Time columns
library(lubridate)
tx$booking_Year <- as.numeric(year(tx$booking_created))
tx$booking_Month <- as.numeric(month(tx$booking_created))
tx$booking_Day <- as.numeric(day(tx$booking_created))
tx$booking_Hour <- as.numeric(hour(tx$booking_created))
tx$booking_Minute <- as.numeric(minute(tx$booking_created))

tx$from_date_Year <- as.numeric(year(tx$from_date))
tx$from_date_Month <- as.numeric(month(tx$from_date))
tx$from_date_Day <- as.numeric(day(tx$from_date))
tx$from_date_Hour <- as.numeric(hour(tx$from_date))
tx$from_date_Minute <- as.numeric(minute(tx$from_date))

tx$to_date_Year <- as.numeric(year(tx$to_date))
tx$to_date_Month <- as.numeric(month(tx$to_date))
tx$to_date_Day <- as.numeric(day(tx$to_date))
tx$to_date_Hour <- as.numeric(hour(tx$to_date))
tx$to_date_Minute <- as.numeric(minute(tx$to_date))
#change any empty values to 0
tx$booking_Hour <- na.replace(tx$booking_Hour, fill = 0)
tx$booking_Minute <- na.replace(tx$booking_Minute, fill = 0)
tx$from_date_Hour <- na.replace(tx$from_date_Hour, fill = 0)
tx$from_date_Minute <- na.replace(tx$from_date_Minute, fill = 0)
tx$to_date_Hour <- na.replace(tx$to_date_Hour, fill = 0)
tx$to_date_Minute <- na.replace(tx$to_date_Minute, fill = 0)

####drop columns with many null values, row, and user id
drop <- c("user_id","row.") 
dfTX = tx[,!(names(tx) %in% drop)]
dfTX <- na.omit(dfTX)
glimpse(dfTX)
View(dfTX)

#normalize data
dfTX1.norm<-sapply(dfTX, scale)
row.names(dfTX1.norm)<-row.names(dfTX)


######################Data Partitioning############################
library(caret)

# Split the data into training (60%) and test set (40%) 
set.seed(2020)
train.index <- sample(c(1:dim(dfTX1.norm)[1]), dim(dfTX1.norm)[1]*0.6)
train.df <- dfTX[train.index,]
valid.df <- dfTX[-train.index,]

#model for all variables
logit.reg <- glm(train.df$Car_Cancellation~., data = train.df, family = "binomial")
options(scipen=999)
summary(logit.reg)
logit.reg


#model3 with less variables to prevent overfitting
logit.reg3 <- glm(valid.df$Car_Cancellation~ valid.df$vehicle_model_id+valid.df$travel_type_id, data = train.df, family = "binomial")
options(scipen=999)
summary(logit.reg3)
logit.reg3


#Prediction for model3
blr1.predict<-predict(logit.reg3, valid.df)
blr1<-data.frame(valid.df, blr1.predict)
blr1



#Validation
RMSE<-sqrt(mean((valid.df$Car_Cancellation-pred.test)^2))
RMSE


cm_lg <- table(valid.df$Car_Cancellation, blr1.predict)
row.names(cm_lg) <- c("Actual: 0", "Actual: 1")
colnames(cm_lg) <- c("Predicted: 0", "Predicted: 1")
cm_lg <- addmargins(A= cm_lg, FUN = list(Total = sum), quiet = TRUE)
cm_lg


#####Test for Accuracy
(cm_lg[1,1]+cm_lg[2,2])/(cm_lg[1,1]+cm_lg[1,2]+cm_lg[2,1]+cm_lg[2,2])
#####Error Rate
1-((cm_lg[1,1]+cm_lg[2,2])/(cm_lg[1,1]+cm_lg[1,2]+cm_lg[2,1]+cm_lg[2,2]))
#####Sensitivity
cm_lg[1,1]/(cm_lg[1,1]+cm_lg[2,1])
#####Specificity
cm_lg[2,2]/(cm_lg[2,2]+cm_lg[1,2])
####Calculate Precision
cm_lg[1, 1]/(cm_lg[1, 1] + cm_lg[1, 2])


#################Gain Chart#######################################
library(gains)
gain <- gains(valid.df$Car_Cancellation, blr1.predict, groups = length(blr1.predict))

plot(c(0, gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))
     ~c(0,gain$cume.obs),xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0, sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

heights<- gain$mean.resp/mean(valid.df$Car_Cancellation)
midpoints<- barplot(heights, names.arg = gain$depth, ylim = c(0,9),
                    xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise lift chart")
text(midpoints, heights+0.5, labels=round(heights,1), cex = 0.8)



##################################################################
##################################################################
##################################################################
####################Decision Tree Classification##################

######load dataset and glimpse data types
tx<-read.csv("Taxi-cancellation-case.csv")
View(tx)
###libraries
library(tidyverse)
library(dplyr)
library(magrittr)
glimpse(tx)
#####check for null values
install.packages(VIM)
library(VIM)
aggr(tx)

#################DATA PREPARATION##################################

#change NA's in to_lat and to_long to mean; other NA's = 0
#reinstall ggplot2 if install error for imputeTS
install.packages("imputeTS")
library(imputeTS)
tx$to_lat <- na.mean(tx$to_lat, option = "mean")
tx$to_long <- na.mean(tx$to_long, option ="mean")
tx$package_id <- na.replace(tx$package_id, fill = 0)
tx$to_area_id <- na.replace(tx$to_area_id, fill = 0)
tx$from_city_id <- na.replace(tx$from_city_id, fill = 0)
tx$to_city_id <- na.replace(tx$to_city_id, fill = 0)

##_________________ compute trip length from GPS data ________________________#
#Create the distance calculation function. This function gets: the latitude
#and longitudes of two points on each and returns the distance in kilometers
dist <- function (long1, lat1, long2, lat2){
        rad <- pi/180 #columns are
        a1 <- lat1 * rad
        a2 <- long1 * rad
        b1 <- lat2 * rad
        b2 <- long2 * rad
        dlon <- b2 - a2
        dlat <- b1 - a1
        a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
        c <- 2 * atan2(sqrt(a), sqrt(1 - a))
        R <- 6378.145
        d <- R * c
        return(d)}
dist(tx$from_long, tx$from_lat, tx$to_long, tx$to_lat)

#separate or mutate the Date/Time columns
install.packages(anytime)
library(anytime)
tx$booking_created <- anytime::anydate(as.numeric(tx$booking_created))
tx$from_date <- anytime::anydate(as.numeric(tx$from_date))
tx$to_date <- anytime::anydate(as.numeric(tx$to_date))

# Separate or mutate the Date/Time columns
library(lubridate)
tx$booking_Year <- as.numeric(year(tx$booking_created))
tx$booking_Month <- as.numeric(month(tx$booking_created))
tx$booking_Day <- as.numeric(day(tx$booking_created))
tx$booking_Hour <- as.numeric(hour(tx$booking_created))
tx$booking_Minute <- as.numeric(minute(tx$booking_created))

tx$from_date_Year <- as.numeric(year(tx$from_date))
tx$from_date_Month <- as.numeric(month(tx$from_date))
tx$from_date_Day <- as.numeric(day(tx$from_date))
tx$from_date_Hour <- as.numeric(hour(tx$from_date))
tx$from_date_Minute <- as.numeric(minute(tx$from_date))

tx$to_date_Year <- as.numeric(year(tx$to_date))
tx$to_date_Month <- as.numeric(month(tx$to_date))
tx$to_date_Day <- as.numeric(day(tx$to_date))
tx$to_date_Hour <- as.numeric(hour(tx$to_date))
tx$to_date_Minute <- as.numeric(minute(tx$to_date))
#change any empty values to 0
tx$booking_Hour <- na.replace(tx$booking_Hour, fill = 0)
tx$booking_Minute <- na.replace(tx$booking_Minute, fill = 0)
tx$from_date_Hour <- na.replace(tx$from_date_Hour, fill = 0)
tx$from_date_Minute <- na.replace(tx$from_date_Minute, fill = 0)
tx$to_date_Hour <- na.replace(tx$to_date_Hour, fill = 0)
tx$to_date_Minute <- na.replace(tx$to_date_Minute, fill = 0)

####drop columns with many null values, row, and user id
drop <- c("user_id","row.") 
dfTX = tx[,!(names(tx) %in% drop)]
dfTX <- na.omit(dfTX)
glimpse(dfTX)
View(dfTX)

########################Data Partitioning##########################
library(caret)
set.seed(2020)
samv<-createDataPartition(dfTX$Car_Cancellation, p=0.6, list = FALSE)
trainz<-dfTX[samv,]
testz<-dfTX[-samv,]


###############Decision Tree Algorithm#############################

###########______Predictor Model Selection_____#############
#libraries
library(rpart)
library(e1071)
library(rpart.plot)
#tree model
dfTX1.tree2<-rpart(Car_Cancellation ~mobile_site_booking+online_booking+package_id, data = trainz, control = rpart.control(minibucket = 10, cp=0))


#Visualizing Tree
library(rpart.plot)
prp(dfTX1.tree2, type = 2, nn = TRUE, fallen.leaves = TRUE, faclen = 4, varlen = 8, shadow.col = "green")


#Prune Tree 
printcp(dfTX1.tree2)
plotcp(dfTX1.tree2)
dfTX1.pruned<-prune(dfTX1.tree2, 0.0005)

#Plot Pruned Tree Graph
prp(dfTX1.pruned, type = 2, nn = TRUE, fallen.leaves = TRUE, faclen = 4, varlen = 8, shadow.col = "gray")

#Rebuild Tree
dfTX1.tree4<-rpart(Car_Cancellation ~mobile_site_booking, data = testz, control = rpart.control(minisplit = 10, cp=0))
#Plot Tree
prp(dfTX1.tree4, type = 2, nn = TRUE, fallen.leaves = TRUE, faclen = 4, varlen = 8, shadow.col = "green")


##############________Performance & Validation______________###############
#Predicting Training and Test Dataset and Create CM
pred.trainz<-predict(dfTX1.tree4, trainz, type="vector")
cmz1<-table(trainz$Car_Cancellation,pred.trainz, dnn=c("Actual", "Predicted"))
cmz1
pred.testz<-predict(dfTX1.tree4, testz, type="vector")
cmz2<-table(testz$Car_Cancellation, pred.testz, dnn=c("Actual", "Predicted"))
cmz2


#Extract confusion matrix
cm4 <- table(testz$Car_Cancellation,pred.testz)
row.names(cm4) <- c("Actual: 0", "Actual: 1")
colnames(cm4) <- c("Predicted: 0", "Predicted: 1")
cm4 <- addmargins(A= cm4, FUN = list(Total = sum), quiet = TRUE)
cm4


#####Test for Accuracy
(cm4[1,1]+cm4[2,2])/(cm4[1,1]+cm4[1,2]+cm4[2,1]+cm4[2,2])
####Calculate Precision
cm4[1, 1]/(cm4[1, 1] + cm4[1, 2])
#####Error Rate
1-((cm4[1,1]+cm4[2,2])/(cm4[1,1]+cm4[1,2]+cm4[2,1]+cm4[2,2]))
#####Sensitivity
cm4[1,1]/(cm4[1,1]+cm4[2,1])
#####Specificity
cm4[2,2]/(cm4[2,2]+cm4[1,2])



