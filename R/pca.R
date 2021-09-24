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


#PCA
data.features.pr <- prcomp(data.features[c(1:30)], center = TRUE, scale = TRUE)
summary(data.features.pr)

#plot(data.features.pr$x[,1],data.features.pr$x[,2], xlab="PC1 (46.3%)", ylab = "PC2 (11.5%)", main = "PC1 / PC2 - plot")

fviz_pca_ind(data.features.pr, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = as.factor(data.features$reliability), 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Reliability") +
  ggtitle("2D PCA-plot from 30 features dataset") +
  theme(plot.title = element_text(hjust = 0.5))





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



#PCA
pca = preProcess(x = train, method = 'pca', pcaComp = 8)
train <-  predict(pca, train)
test = predict(pca, test)


ctrl <- trainControl(method="LOOCV")


#K-NEAREST NEIGHBOARD
set.seed(100)
knn <- train(labels ~ ., data = train, method = "knn", trControl = ctrl)
importance.knn <- varImp(knn, scale=FALSE)
plot(importance.knn)


knn.predict <- predict(knn, newdata = test)

#knn
#plot(knn)
cm.knn <- confusionMatrix(knn.predict, as.factor(test$labels))
cm.knn$table
e.knn <- eval(cm.knn)
tpr.knn <- e.knn[[1]]
tnr.knn <- e.knn[[2]]

tpr.knn
tnr.knn


roc.knn <- roc(as.numeric(test$labels), as.numeric(knn.predict),
               smoothed = TRUE,
               # arguments for ci
               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE)

coords(roc.knn, "best")
#sens.ci <- ci.se(roc.knn)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")



#SUPPORT VECTOR MACHINE
# kernel: linear 
# tuning parameters: C 
#set.seed(100)
#svm.linear  <- train(labels ~ . , data=train, trControl = ctrl, method = "svmLinear")

#plot(svm.linear)
#importance.svm <- varImp(svm.linear, scale=FALSE)
#plot(importance.svm)


# predict on test data
#svm.linear.predict <- predict(svm.linear,newdata = test)
#svm.linear
#cm.svm <- confusionMatrix(svm.linear.predict, as.factor(test$labels))
#cm.svm$table
#cm.svm
#e.svm <- eval(cm.svm)
#tpr.svm <- e.svm[[1]]
#tnr.svm <- e.svm[[2]]
#precision.svm <- e.svm[[3]]
#npv.svm <- e.svm[[4]]
#tpr.svm
#tnr.svm
#precision.svm
#npv.svm

#roc.svm <- roc(as.numeric(svm.linear.predict), as.numeric(test$labels),
#               smoothed = TRUE,
               # arguments for ci
#               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
 #              plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
 #              print.auc=TRUE, show.thres=TRUE)


#sens.ci <- ci.se(roc.svm)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")

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


roc.lda <- roc( as.numeric(test$labels), as.numeric(lda.predict),
               smoothed = TRUE,
               # arguments for ci
               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE)

coords(roc.lda, "best")
#sens.ci <- ci.se(roc.lda)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")






#LOGISTIC REGRESSION
set.seed(100)
glm  <- train(labels ~ . , data = train, method = "glm", family = "binomial")

# predict on test data
glm.predict <- predict(glm,newdata = test)
#glm
importance.glm <- varImp(glm, scale=FALSE)
plot(importance.glm)


cm.glm <- confusionMatrix(glm.predict , as.factor(test$labels))
cm.glm$table

e.glm <- eval(cm.glm)
tpr.glm <- e.glm[[1]]
tnr.glm <- e.glm[[2]]

tpr.glm
tnr.glm


roc.glm <- roc(as.numeric(test$labels), as.numeric(glm.predict), 
               smoothed = TRUE,
               # arguments for ci
               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE)

coords(roc.glm, "best")
#sens.ci <- ci.se(roc.glm)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")



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


roc.tree <- roc(as.numeric(test$labels), as.numeric(tree.predict),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)

coords(roc.tree, "best")
#sens.ci <- ci.se(roc.tree)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")




roc.list <- list(KNN = roc.knn, LDA = roc.lda, GLM = roc.glm, TREE = roc.tree)

ggroc(roc.list, aes = c("line", "color"), legacy.axes = T) +
  geom_abline() +
  theme_classic() +
  ggtitle("PCA ROC") +
  labs(x = "1 - Specificity",
       y = "Sensitivity",
       linetype = "Legend")

