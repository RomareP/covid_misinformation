library(tm)
library(proxy)
library(dplyr)
library(data.table)
library(qdap)
library(quanteda)
library(SnowballC)
library(RColorBrewer)
library(superml)
library(textreg)
library(caret)
library(kernlab)
library(mlbench)
library(stats)
library(factoextra)
library(pROC)


eval <- function(cm){
  tp <- cm$table[1]
  tn <- cm$table[4]
  fp <- cm$table[3]
  fn <- cm$table[2]
  tpr <- tp/(tp+fn)
  tnr <- tn/(tn+fp)
  precision <- tp/(tp+fp)
  npv <- tn/(tn+fn)
  return(list(tpr, tnr, precision, npv))
  
}




standardize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm=TRUE))
}




data.features <- read.csv('/Users/pieroromare/Desktop/Data_Science/II/Statistical\ Learning\ Mod.\ B/project/content-features.csv')
summary(data.features)
data.features$X <- NULL
data.features$fea_body_word_unique_percent <- NULL
#View(data)

names(data.features)




#FEATURES STANDARTIZATION
std.features <- as.data.frame(apply(data.features[1:30], 2, standardize))

sapply(std.features, class)





#SPLIT DATASET
features <- std.features
features$reliability <- NULL
labels <- as.factor(data.features$reliability)
df <- data.table(features, labels)

rows <- sample(nrow(df))
shuffle_df <- df[rows, ]


sample <- sample.int(n = nrow(shuffle_df), size = floor(.8*nrow(shuffle_df)), replace = F)
train <- shuffle_df[sample, ] #.8
test  <- shuffle_df[-sample, ] #.2
sum(train$labels == 0)
sum(train$labels == 1)
sum(test$labels == 0)
sum(test$labels == 1)



#UNDERSAMPLING
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3, 
                     verboseIter = FALSE,
                     sampling = "down")

#OVERSAMPLING
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3, 
                     verboseIter = FALSE,
                     sampling = "up")
#rose #smote #down #up




#K-NEAREST NEIGHBOARD
set.seed(100)
knn <- train(labels ~ ., data = train, method = "knn", trControl = ctrl)
importance.knn <- varImp(knn, scale=FALSE)
plot(importance.knn)


knn.predict <- predict(knn, newdata = test)

#knn
cm.knn <- confusionMatrix(knn.predict, as.factor(test$labels))
cm.knn$table
e.knn <- eval(cm.knn)
tpr.knn <- e.knn[[1]]
tnr.knn <- e.knn[[2]]
tpr.knn
tnr.knn

roc.knn <- roc(as.numeric(knn.predict), as.numeric(test$labels),
               smoothed = TRUE,
               # arguments for ci
               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE)

coords(roc.knn, "best", transpose = T)



#LDA
set.seed(100)
lda  <- train(labels ~ . , data = train, method = "lda") 

# predict on test data
lda.predict <- predict(lda,newdata = test)
#lda
cm.lda <- confusionMatrix(lda.predict , as.factor(test$labels))
cm.lda$table

e.lda <- eval(cm.lda)
tpr.lda <- e.lda[[1]]
tnr.lda <- e.lda[[2]]
tpr.lda
tnr.lda

roc.lda <- roc(as.numeric(lda.predict), as.numeric(test$labels),
               smoothed = TRUE,
               # arguments for ci
               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE)

coords(roc.lda, "best", transpose = T)





#LOGISTIC REGRESSION
set.seed(100)
glm  <- train(labels ~ . , data = train, method = "glm", family = "binomial")

# predict on test data
glm.predict <- predict(glm,newdata = test)
importance.glm <- varImp(glm, scale=FALSE)
plot(importance.glm)


cm.glm <- confusionMatrix(glm.predict , as.factor(test$labels))
cm.glm$table

e.glm <- eval(cm.glm)
tpr.glm <- e.glm[[1]]
tnr.glm <- e.glm[[2]]


tpr.glm
tnr.glm
#glm

roc.glm <- roc(as.numeric(glm.predict), as.numeric(test$labels),
               smoothed = TRUE,
               # arguments for ci
               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE)

coords(roc.glm, "best", transpose = T)




#DECISION TREE
set.seed(100)
tree  <- train(labels ~ . , data = train, method = "rpart", trControl = ctrl )
importance.tree <- varImp(tree, scale=FALSE)
plot(importance.tree)

tree.predict <- predict(tree,newdata = test)

#plot(tree)
cm.tree <- confusionMatrix(tree.predict , as.factor(test$labels))
cm.tree$table
e.tree <- eval(cm.tree)
tpr.tree <- e.tree[[1]]
tnr.tree <- e.tree[[2]]


tpr.tree
tnr.tree
#tree

roc.tree <- roc(as.numeric(tree.predict), as.numeric(test$labels),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)

coords(roc.tree, "best")