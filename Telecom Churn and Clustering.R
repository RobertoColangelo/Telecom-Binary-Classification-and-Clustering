library(reshape2) #Required for data transformation
library(memisc) #Required for finding missing values
library(caret) #Required for train-test partition and pre-processing
library(xgboost) #Required for Extreme Gradient Boosting model
library(randomForest) #Required for Random Forest model
library(tidyverse) #Required for dplyr and ggplot2
library(dplyr) #Required for data manipulation
library(ggplot2) #Required for data visualization 
library(MASS) 
library(pROC) #Required for Roc curve
library(class) #Required for knn
library(e1071) # For Naive Bayes
library(factoextra) # for Clustering visualization
library(gridExtra) # For clustering plots grid visualization

#Importing csv file
df<- read.csv(file = '/Users/robertocolangelo/Desktop/Progetto Brazilian Houses/Data Analysis final project/TelecomChurn.csv')
df

dim(df) #3333 instances and 20 features

str(df) #Let's investigate on the typologies of the variables (4 character variables to handle)

head(df,10)

#Computing some basic statistics
summary(df)

#Null values 
is.null(df) #return FALSE ->not even a single missing value

#Removing duplicates and creating a new dataframe (df) with no duplicates
df<-unique(df)
dim(df) 
#Now 3333 instances.There were no duplicates


#Looking for null values 
colSums(is.na(df))
which(colSums(is.na(df))>0)
names(which(colSums(is.na(df))>0))
#No null values the dataset is perfect

#Data Exploration

#Let's check our target variable Churn
table(df$Churn) #2850 customers are not churning and 483 are churning 
prop.table(table(df$Churn)) #85.6% of customer are not churning, 14.4 are churning

#Histogram Churn
ggplot(df)+
  geom_bar(aes(x=Churn,fill=Churn))+
  labs(y='Number of Customers')

#Barplot of Churn by state
ggplot(df)+
  geom_bar(aes(x=State,fill=Churn))+
  theme(axis.text.x = element_text(vjust=0.5, angle = 0,size=10),axis.text.y=element_text(size=5))+
  coord_flip()
#NJ,TX and MD are the states with the highest number of churning customers

#Total day minutes vs total day charge
ggplot(df)+
  geom_point(aes(x=Total.day.minutes,y=Total.day.charge))
#Perfectly linear relationship --> this will be seen also in the correlation matrix

#Violin plot of Total daily calls vs Customer service calls
dfte<-df
dfte$dailycalls<-dfte$Total.day.calls+dfte$Total.eve.calls+dfte$Total.night.calls
ggplot(dfte)+
  geom_violin(aes(x=as.character(Customer.service.calls),y=dailycalls,fill=as.character(Customer.service.calls)),show.legend=F)+
  labs(x='Customer Service Calls',y='Daily Calls')
#On average customer who make more phone calls are also the ones who call the most the customer service

#Boxplot of Number of voice mail messages vs Churn
ggplot(df)+
  geom_boxplot(aes(x=Churn ,y=Number.vmail.messages,fill=Churn))
#We can Notice for Voice-Mail Feature when there are more than 20 voice-mail messages then certainly there is a churn indicating improving the voice-mail feature or setting a limit and check whether a customer is retained

#Boxplot of day charges vs Churn
ggplot(df)+
  geom_boxplot(aes(x=Churn ,y=Total.day.charge,fill=Churn))
#The higher the day  charge on the customer the more the customers will churn , the company should set a limit of charge or promoting some offer to avoid these huge charges

#Let's handle the categorical variables
#State can be removed since it has 51 values and would lead to the creation of too many useless variables
df$State<-NULL
unique(df$Area.code) #Might be useful to see the variability in the data stemming from the geographic location
df$International.plan<-ifelse(df$International.plan=='Yes',1,0)
df$Voice.mail.plan<-ifelse(df$Voice.mail.plan=='Yes',1,0)

df$Churn<-ifelse(df$Churn=='True',1,0)
df[,]%>%
  cor()%>%
  melt()%>%
  ggplot(aes(Var1,Var2, fill=value))+
  scale_fill_distiller(palette = "YlOrBr")+
  geom_tile(color='white')+
  geom_text(aes(label=paste(round(value,2))),size=2, color='black')+
  theme(axis.text.x = element_text(vjust=0.5, angle = 90))+
  labs(title='Correlation between variables',
       x='',y='')
#Most correlated variables with Churn--> Area Code, Total.day.charge,Total.day.minutes
#Perfect collinearity between minutes and charges
#Very high collinearity between Vmail plan and Number vmail messages

#Modeling
#Task 1 Binary Classification --> Predicting Churn

# Train - test split
set.seed(21) # In order to avoid any inconsistencies when we rerun the code
training <- df$Churn %>%
  createDataPartition(p = 0.70, list = FALSE) #Using 70% because we don't have many instances to evaluate our models on
trainset  <- df[training, ]
xtrain<-trainset[,-19]
ytrain<-trainset[,19]
testset<- df[-training, ]
xtest<-testset[,-19]
ytest<-testset[,19]
#Scaling independent variables
xtrain <- xtrain %>% scale()
xtest <- xtest %>% scale(center=attr(xtrain, "scaled:center"), 
                         scale=attr(xtrain, "scaled:scale"))
dim(xtrain) #2334 instances to train our models
dim(xtest) #999 instances to evaluate our models

#Baseline Model -> Simple logistic regression
baseline<-glm(data=trainset,Churn~Area.code,family='binomial')
summary(baseline) 
#Estimate of the coefficient of Area.code is 0.0004 this means that it has a low predictive impact and is not significant
predictions<-predict(baseline,testset[,-19],type='response')
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])#Predicts all as 0 ('FALSE)
baselineaccuracy<-847 /(847+152)
baselineaccuracy #0.848
ROC1 <- roc(testset[,19], predictions, plot = TRUE,
                 legacy.axes=TRUE, col="midnightblue", lwd=3,
                 auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)

#Complete logistic Regression
lr<-glm(data=trainset,Churn~.,family='binomial')
summary(lr) #Many variables don't seem to be statistically significant ,later we'll try with stepwise selection
predictions<-predict(lr,testset[,-19],type='response')
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])
confmat<-confusionMatrix(table(testset[,19],predictions))
lraccuracy<-confmat$overall[1]
#Accuracy 0.8649 improves a bit
ROC2 <- roc(testset[,19], predictions, plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#Even the AUC improves a bit

#Stepwise Models
#Forward selection
AICfor<-step(lr,direction='forward')
summary(AICfor)
predictions<-predict(AICfor,testset[,-19],type='response')
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])
confmat<-confusionMatrix(table(testset[,19],predictions))
AICforaccuracy<-confmat$overall[1]
#Accuracy 0.8649 ,same as before
ROC3 <- roc(testset[,19], predictions, plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC same as before

#Backword selection
AICback<-step(lr,direction='backward')
summary(AICback)
predictions<-predict(AICback,testset[,-19],type='response')
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])
confmat<-confusionMatrix(table(testset[,19],predictions))
AICbackaccuracy<-confmat$overall[1]
#Accuracy 0.8659 ,improves a little bit
ROC4 <- roc(testset[,19], predictions, plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC same as before

#Mixed selection
AICmixed<-step(lr,direction='both')
summary(AICmixed)
predictions<-predict(AICmixed,testset[,-19],type='response')
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])
confmat<-confusionMatrix(table(testset[,19],predictions))
AICmixedaccuracy<-confmat$overall[1]
#Accuracy 0.8659 same as backward
ROC5<- roc(testset[,19], predictions, plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC improves a very little bit

#Interaction term 
Interaction<-glm(Churn ~ Account.length + Area.code + International.plan + Voice.mail.plan + 
                   Number.vmail.messages +I(Total.day.minutes*Total.day.charge)+ Total.day.minutes + Total.day.calls + 
                   Total.day.charge + Total.eve.minutes + Total.eve.calls + 
                   Total.eve.charge + Total.night.minutes + Total.night.calls + 
                   Total.night.charge + Total.intl.minutes + Total.intl.calls + 
                   Total.intl.charge + Customer.service.calls,data=trainset,family='binomial')
summary(Interaction)
predictions<-predict(Interaction,testset[,-19],type='response')
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])
confmat<-confusionMatrix(table(testset[,19],predictions))
Interactionaccuracy<-confmat$overall[1]
#Accuracy 0.8798 improves  a lot by 0.02
ROC6 <- roc(testset[,19], predictions, plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC improves a by 0.05

#Naive Bayes Classifier
NBclassifier <- naiveBayes(Churn ~ ., data = trainset, usekernel = T) 
predictions<-predict(NBclassifier,testset[,-19])
predictions #The values of the predictions are the probabilities for each customer to churn
table(predictions, testset[,19])
confmat<-confusionMatrix(table(testset[,19],predictions))
NBaccuracy<-confmat$overall[1]
#Accuracy 0.8728 
ROC7 <- roc(testset[,19], as.numeric(predictions), plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC improves a lot--> 0.747 !

#KNN Classifier
knn <- train(as.factor(Churn)~.,data=trainset,method='knn',
                        tuneGrid=expand.grid(k=1:20),metric="Accuracy",
                        trControl = trainControl(method = 'repeatedcv',
                                                 number=10,repeats = 15))
plot(knn) #Plot of the accuracy given number of neighbours best is 12 neighbours
knn <- knn(trainset,testset,trainset$Churn,k=12,prob=TRUE)
knn #Predictions
 #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(knn>0.5,1,0)
#convert the predicted probability of winning class to the positive class
confmat<-confusionMatrix(knn, as.factor(testset[,19]), positive ='1')
knnaccuracy<-confmat$overall[1]
#Accuracy 0.8729 
ROC8 <- roc(testset[,19],predictor=as.numeric(knn), plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC is a bit lower--> 0.625

#Support Vector Machine Classifier

#Linear Kernel
linearSVM<- svm(Churn~.,data=trainset,cost=10,kernel='linear') #We could have tried to tune the costs but R is giving problems with the number of iterations
summary(linearSVM)#check the summary,values of gamma and epsilon
predictions<-predict(linearSVM,testset[,-19])
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])#Predicts all as 0 ('FALSE)
linearSVMaccuracy<-847 /(847+152)
linearSVMaccuracy #0.848
#Accuracy 0.8728 
ROC9 <- roc(testset[,19], as.numeric(predictions), plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC  0.5
#Worst model with the baseline so far

#Radial Basis Function Kernel
RBFSVM<- svm(Churn~.,data=trainset,cost=10,kernel='radial')
summary(RBFSVM)#check the summary,values of gamma and epsilon
predictions<-predict(RBFSVM,testset[,-19])
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])#Predicts all as 0 ('FALSE)
confmat<-confusionMatrix(table(predictions,testset[,19]))
RBFSVMaccuracy<-confmat$overall[1]
RBFSVMaccuracy #0.9259 Great!
ROC10 <- roc(testset[,19], as.numeric(predictions), plot = TRUE,
            legacy.axes=TRUE, col="midnightblue", lwd=3,
            auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC  0.797 
#Changing kernel improves the performances of the model

#Random Forest Classifier
RfC <- randomForest(formula= Churn~ ., data=trainset, ntree=500,importance=F)
predictions<-predict(RfC,testset[,-19],type = 'response')
predictions #The values of the predictions are the probabilities for each customer to churn
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])#Predicts all as 0 ('FALSE)
confmat<-confusionMatrix(table(predictions, testset[,19]))
RfCaccuracy<-confmat$overall[1]
#Accuracy --> 0.955 almost perfect!
ROC11 <- roc(testset[,19], predictions, plot = TRUE,
             legacy.axes=TRUE, col="midnightblue", lwd=3,
             auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC 0.874 the best one so far

#XGBoost Classifier
XGBoost = xgboost(data = as.matrix(trainset[,-19]), label = as.matrix(trainset[,19]), nrounds = 150, 
                  objective = "reg:squarederror", eval_metric = "error")

err_xg_tr = xgboost$evaluation_log$train_error
predictions<-predict(XGBoost,as.matrix(testset[,-19]))
predictions<-ifelse(predictions>0.5,1,0)
table(predictions, testset[,19])
confmat<-confusionMatrix(table(predictions, testset[,19]))
XGBoostaccuracy<-confmat$overall[1]
#Accuracy --> 0.950 almost as good as Random Forest
ROC12 <- roc(testset[,19], predictions, plot = TRUE,
             legacy.axes=TRUE, col="midnightblue", lwd=3,
             auc.polygon=T, auc.polygon.col="lightblue", print.auc=T)
#AUC 0.871 the second best one so far

#Results Comparison
models<-c('baseline Logistic','Logistic Regression','Step_forwardAIC', "Step_backwardAIC", "MixedAIC",'Interaction', "Naive Bayes",'KNN',"Linear SVM","Radial SVM ", "Random Forest", 'XGBoost')
r=c(baselineaccuracy,lraccuracy,AICforaccuracy,AICbackaccuracy, AICmixedaccuracy, Interactionaccuracy,NBaccuracy, knnaccuracy, linearSVMaccuracy,RBFSVMaccuracy, RfCaccuracy,XGBoostaccuracy)
Comparazione <- data.frame(Model = models, Accuracy = r)
comp<-Comparazione[rev(order(Comparazione$Accuracy)),]#Prints 5 most performing models
comp[1:5,]
#Best model is random forest followed by XGBoost and Radial kernel SVM
Performanceplot<- ggplot(data = Comparazione) + 
  geom_point(mapping = aes(x = Model, y=Accuracy,colour='Red'),show.legend=FALSE) +
  labs(x = 'Models', y = 'Accuracy')
Performanceplot+theme(axis.text.x=element_text(angle=90))


#TASK 2 CLUSTERING
set.seed(34)
dfclust<-df[,c(1,8)] #Dataframe with only two variables, we want a more interpretable clustering

#Hierarchical clustering 
d1 <- dist(df, method = "euclidean")
hc1 <- hclust(d1, method = "complete" ) #Complete linkage
fviz_nbclust(df, FUN = hcut, method = "wss") #Takes a while to run and doens't clearly shows an elbow, so let's use 4 clusters
sub_grp1 <- cutree(hc1, k = 4)
fviz_cluster(list(data = df, cluster = sub_grp1),geom = "point")
# Clusters are not so clear so let's try with only a couple of variables (account length and total day charge )

#Hierarchical clustering with only 2 variables
d2 <- dist(dfclust, method = "euclidean")
hc2 <- hclust(d2, method = "complete" ) #Complete linkage
fviz_nbclust(dfclust, FUN = hcut, method = "wss") #Takes a while to run and shows the best number of clusters is 4 according to the elbow method
sub_grp2 <- cutree(hc2, k = 4)
fviz_cluster(list(data = dfclust, cluster = sub_grp2),geom = "point")

# Kmeans Clustering 
l<-list()
for (k in 2:5) {
  km <- kmeans(df,centers=k,nstart=10)
  l[[k]]<-km$tot.withinss
  e[[k]]<-km
}
l[[1]]<-NULL
l<-matrix(l)
Nclust<-c(2,3,4,5)
Clustercomparisonwss<-data.frame(WCSS=l,K=Nclust)
Clustercomparisonwss
#Not really a great variation of Within Cluster Sum of Squares

# Let's plot the clusters graphically
p1 <- fviz_cluster(e[[2]], geom = "point", data = df) + ggtitle("k = 2")
p2 <- fviz_cluster(e[[3]], geom = "point",  data = df) + ggtitle("k = 3")
p3 <- fviz_cluster(e[[4]], geom = "point",  data = df) + ggtitle("k = 4")
p4 <- fviz_cluster(e[[5]], geom = "point",  data = df) + ggtitle("k = 5")
grid.arrange(p1, p2, p3, p4, nrow = 2)
#Really messy clusters,can't really be interpreted

#Let's try with just two variables (account length and total day charge)
l<-list()
e<-list()
for (k in 2:5) {
  km <- kmeans(dfclust,centers=k,nstart=10)
  l[[k]]<-km$tot.withinss
  e[[k]]<-km
}
l[[1]]<-NULL
l<-matrix(l)
Nclust<-c(2,3,4,5)
Clustercomparisonwss<-data.frame(WCSS=l,K=Nclust)
Clustercomparisonwss
#Obviously the more clusters,the less Within cluster sum of squares, WATCH OUT the first two value seem shorter but just because they don't have decimal, THEY ARE BIGGER

# Let's see the representation to compare
pm1 <- fviz_cluster(e[[2]], geom = "point", data = dfclust) + ggtitle("k = 2")
pm2 <- fviz_cluster(e[[3]], geom = "point",  data = dfclust) + ggtitle("k = 3")
pm3 <- fviz_cluster(e[[4]], geom = "point",  data = dfclust) + ggtitle("k = 4")
pm4 <- fviz_cluster(e[[5]],geom = "point",  data = dfclust) + ggtitle("k = 5")
grid.arrange(pm1, pm2, pm3, pm4, nrow = 2)
#We can see that the clusters are mostly affected by the account length and as we add more clusters the algorithm just creates new groups based on the account length ranges
