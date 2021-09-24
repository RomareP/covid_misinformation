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





stopwords <- read.csv('/Users/pieroromare/Desktop/Data_Science/II/Statistical\ Learning\ Mod.\ B/project/stopwords.csv')
sw <-  c(stopwords$X0)

data <- read.csv('/Users/pieroromare/Desktop/Data_Science/II/Statistical\ Learning\ Mod.\ B/project/news-dataset.csv')
summary(data)
#View(data)
data <- data[!(is.na(data$body_text) | data$body_text==""), ]
sents <- data$body_text

#convert vector to corpus
docs <- Corpus(VectorSource(sents))
#feature preprocessing
docs <- tm_map(docs, content_transformer(tolower)) #text to lowercase
docs <- tm_map(docs, removePunctuation) #remove punct
docs <- tm_map(docs, removeNumbers) #remove numbers
docs <- tm_map(docs, removeWords, sw) #remove stopwords
docs <- tm_map(docs, stemDocument, language="italian") #stemming
docs <- tm_map(docs, stripWhitespace) # Eliminate extra white spaces

#convert corpus to dataframe, vector
sents <- data.frame(text = sapply(docs, as.character), stringsAsFactors = FALSE)
dim(sents)


#TF-IDF transformation 100
#tfv.100 <- TfIdfVectorizer$new(max_features = 100, max_df=0.4)
#tf_mat.100 <- tfv.100$fit_transform(sents$text)
#tf_mat.100.shape <- dim(tf_mat.100)
#tf_mat.100.shape
#head(tf_mat.100, 5)

#TF-IDF transformation 200
tfv.200 <- TfIdfVectorizer$new(max_features = 200, max_df=0.4)
tf_mat.200 <- tfv.200$fit_transform(sents$text)
tf_mat.200.shape <- dim(tf_mat.200)
tf_mat.200.shape
head(tf_mat.200, 5)



#INFORMATIVE PCA TO TFIDF (NOT USEFUL)
#pcat <- prcomp(tf_mat)
#summary(pcat)
#plot(pcat$x[,1],pcat$x[,2], xlab="PC1 (46.3%)", ylab = "PC2 (11.5%)", main = "PC1 / PC2 - plot")

#fviz_pca_ind(pcat, geom.ind = "point", pointshape = 21, 
#             pointsize = 2, 
#             fill.ind = as.factor(data$reliability), 
#             col.ind = "black", 
#             palette = "jco", 
#             addEllipses = TRUE,
#             label = "var",
#             col.var = "black",
#             repel = TRUE,
#             legend.title = "Reliability") +
#  ggtitle("2D PCA-plot from 30 features dataset") +
#  theme(plot.title = element_text(hjust = 0.5))

#CORRELATION PLOT FOR TF-IDF IS UNFESEABLE
#library(corrplot)
#library(RColorBrewer)
#M <-cor(tf_mat)
#corrplot(M,  order="hclust", tl.col = "black", tl.cex = 0.5, number.cex = 0.5)
#highlyCor.half <- findCorrelation(M, 0.30)
#Apply correlation filter at 050,
#then we remove all the variable correlated with more 0.5.
#M.filtered <- M[,-highlyCor.half]
#corMatMy.half <- cor(M.filtered)
#corrplot(corMatMy.half, order = "hclust", tl.col = "black", tl.cex = 0.75, number.cex = 0.5)



#SPLIT DATASET
features <- tf_mat.200
labels <- as.factor(data$reliability)
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





#K-NEAREST NEIGHBOARD
ctrl <- trainControl(method="LOOCV", number = 10)

set.seed(100)
knn <- train(labels ~ ., data = train, method = "knn", trControl = ctrl)
#getTrainPerf(knn)
importance.knn <- varImp(knn, scale=FALSE)
temp.knn <- importance.knn
temp.knn$importance <- importance.knn$importance[1:20, ]
plot(temp.knn) #20 most impontant features

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
#knn$finalModel




#SUPPORT VECTOR MACHINE
# kernel: linear 
#set.seed(100)
#svm.linear  <- train(labels ~ . , data=train, trControl = ctrl, method = "svmLinear")

#plot(svm.linear)
#importance.svm <- varImp(svm.linear, scale=FALSE)
#temp.svm <- importance.svm
t#emp.svm$importance <- importance.svm$importance[1:20, ]
#plot(temp.svm) #20 most impontant features


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
#               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
#               print.auc=TRUE, show.thres=TRUE)


#sens.ci <- ci.se(roc.svm)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")

#LDA
set.seed(100)
lda  <- train(labels ~ . , data = train, method = "lda")
importance.lda <- varImp(lda, scale=FALSE)
temp.lda <- importance.lda
temp.lda$importance <- importance.lda$importance[1:20, ]
plot(temp.lda) #20 most impontant features

# predict on test data
lda.predict <- predict(lda,newdata = test)



cm.lda <- confusionMatrix(lda.predict , as.factor(test$labels))
cm.lda$table

e.lda <- eval(cm.lda)
tpr.lda <- e.lda[[1]]
tnr.lda <- e.lda[[2]]
#precision.lda <- e.lda[[3]]
#npv.lda <- e.lda[[4]]
tpr.lda
tnr.lda
#precision.lda
#npv.lda

roc.lda <- roc(as.numeric(test$labels), as.numeric(lda.predict), 
               smoothed = TRUE,
               # arguments for ci
               ci=TRUE, ci.alpha=0.9, stratified=FALSE,
               # arguments for plot
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE)

coords(roc.lda, "best")
summary(lda$finalModel$means)
lda
#sens.ci <- ci.se(roc.lda)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")






#LOGISTIC REGRESSION 
set.seed(100)
glm  <- train(labels ~ . , data = train, method = "glm", family = "binomial")
importance.glm <- varImp(glm, scale=FALSE)
temp.glm <- importance.glm
temp.glm$importance <- importance.glm$importance[1:20, ]
plot(temp.glm) #20 most impontant features

# predict on test data
glm.predict <- predict(glm,newdata = test)
#glm


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
summary(glm)
#sens.ci <- ci.se(roc.glm)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")



#DECISION TREE
set.seed(100)
tree  <- train(labels ~ . , data = train, method = "ctree", trControl = ctrl )
importance.tree <- varImp(tree, scale=FALSE)
temp.tree <- importance.tree
temp.tree$importance <- importance.tree$importance[1:20, ]
plot(temp.tree) #20 most impontant features

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
summary(tree$finalModel)
#sens.ci <- ci.se(roc.tree)
#plot(sens.ci, type="shape", col="lightblue")
#plot(sens.ci, type="bars")




roc.list <- list(KNN = roc.knn, LDA = roc.lda, GLM = roc.glm, TREE = roc.tree)

ggroc(roc.list, aes = c("line", "color"), legacy.axes = T) +
  geom_abline() +
  theme_classic() +
  ggtitle("TF-IDF ROC") +
  labs(x = "1 - Specificity",
       y = "Sensitivity",
       linetype = "Legend")

