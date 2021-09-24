library(ggplot2)
library(tm)
library(gridExtra)
library(stringi)
library(knitr)
library(qdap)
library(slam)
library(tokenizers)
library(wordcloud)
library(scales)
library(corrplot)

data <- read.csv('/Users/pieroromare/Desktop/Data_Science/II/Statistical\ Learning\ Mod.\ B/project/news-dataset.csv')
data.features <- read.csv('/Users/pieroromare/Desktop/Data_Science/II/Statistical\ Learning\ Mod.\ B/project/content-features.csv')

summary(data)
summary(data.features)
#names(data)
#names(data.features)

#View(data)
#View(data.features)
dim(data)
dim(data.features)

#drop X columns which is basically an index inside the dataframe and publish data columns (different wrt publish_date)
data$X <- NULL
data$publish_data <- NULL
data.features$X <- NULL
data.features$fea_body_word_unique_percent <- NULL
#length(names(data.features))


#RREMOVE ROW WHERE BODY_TEXT IS NOT RETRIEVED
dim(data)
dim(data[!(is.na(data$body_text) | data$body_text==""), ])
data <- data[!(is.na(data$body_text) | data$body_text==""), ]
dim(data)

sum(data$reliability == 0)
sum(data$reliability == 1)
#TYPE OF DATA IN DATASET
sapply(data, class)

#EXPLORE DATE FEATURE
length(data$publish_date)
data$publish_date <- replace(data$publish_date, data$publish_date=="", NA)
date <- data[!(is.na(data$publish_date)), 3]
length(date)

freqs <- aggregate(date, by=list(date), FUN=length)
freqs$names <- as.Date(freqs$Group.1, format="%Y-%m-%d")

ggplot(freqs, aes(x=names, y=x)) + geom_bar(stat="identity") +
  scale_x_date(breaks="1 month", labels=date_format("%Y-%b"),
               limits=c(as.Date("2020-02-01"),as.Date("2020-03-30"))) +
  ylab("Frequency") + xlab("Year and Month") +
  theme_bw()


#EXPLORE PUBLISHER
lcm <- data[data[, 8]==0, ]
dim(lcm)
lcm$publisher <- gsub("https://", "", lcm$publisher)
lcm$publisher <- gsub("http://", "", lcm$publisher)

freq.lcm <- aggregate(lcm$publisher, by=list(lcm$publisher), FUN=length)
freq.lcm$names <- as.vector(freq.lcm$Group.1)
ggplot(freq.lcm, aes(x=names, y=x)) + geom_bar(stat='identity') + ylab("Count") + 
  xlab('LCM Publisher') + theme(axis.text.x = element_text( angle = 0)) + coord_flip()


hcm <- data[data[, 8]==1, ]
dim(hcm)
hcm$publisher <- gsub("https://", "", hcm$publisher)
hcm$publisher <- gsub("http://", "", hcm$publisher)

freq.hcm <- aggregate(hcm$publisher, by=list(hcm$publisher), FUN=length)
freq.hcm$names <- as.vector(freq.hcm$Group.1)
ggplot(freq.hcm, aes(x=names, y=x)) + geom_bar(stat='identity') + ylab("Count") + 
  xlab('HCM Publisher') + theme(axis.text.x = element_text( angle = 0)) + coord_flip()

#FEATURES DISTRIBUTION
#par(mfrow = c(2,2))
# Checking distribution of  data
#for (i in 1:29){
#  data.features[, i] <- as.numeric(data.features[, i])
#  hist(data.features[, i], probability = TRUE, main = names(data.features[i]), col = "steelblue", xlab = names(data.features[i]))
#  lines(density(data.features[, i]),col=1)
#}




summary(data.features)
hist(data.features$reliability)

##plot relevant features 
features<-colnames(data.features)
features_rel<-features[1:29]


for(i in features_rel ){
  
  p<-ggplot(data.features,aes_string(x=i,fill="reliability"))+geom_histogram(bins=50,alpha=0.8,colour='steelblue')
  print(p)
}




sample_data <- data$body_text
title <- data$title


par(mfrow = c(1,1))
#LENGTH OF TITLE AND FULL TEXT
wordcount.title <- stri_count_words(title)
hist(wordcount.title, prob=TRUE, main="Title Length", xlab="Title length (words)")
lines(density(wordcount.title))
wordcount.text <- stri_count_words(sample_data)
hist(wordcount.text, breaks = 100, xlim=c(0, 5000), prob=TRUE, main="Body length", xlab="Body length (words)")
lines(density(wordcount.text))
hist(log10(wordcount.text), breaks = 10, xlim=c(1, 5), prob=TRUE, main="Body length (log10)", xlab="Body length (words)")
lines(density(log10(wordcount.text)))



mean(data.features$fea_title_word_nums)
sd(data.features$fea_title_word_nums)
mean(data.features$fea_body_word_nums)
sd(data.features$fea_body_word_nums)

par(mfrow = c(1,1))
#correlation features
corrplot(cor(data.features[,1:30]), type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.5)




stopwords <- read.csv('/Users/pieroromare/Desktop/Data_Science/II/Statistical\ Learning\ Mod.\ B/project/stopwords.csv')
sw <-  c(stopwords$X0)

data_corpus = Corpus(VectorSource(sample_data))

cleanCorpus <- function(myCorpus) {
  myCorpus <- tm_map(myCorpus, removeNumbers)
  myCorpus <- tm_map(myCorpus, removeWords, sw)
  myCorpus <- tm_map(myCorpus, removePunctuation)
  myCorpus <- tm_map(myCorpus, stripWhitespace)
  myCorpus <- tm_map(myCorpus, tolower)
  return(myCorpus)
}


#N-GRAMS
unigram <- function(thisCorpus) {
  thisTDM2D <- TermDocumentMatrix(thisCorpus)
  thisTDM1D <- rollup(thisTDM2D, 2, na.rm = TRUE, FUN = sum)
  thisUniGramDF <- data.frame(words = thisTDM1D$dimnames$Terms, freq = thisTDM1D$v)
  thisUniGramDFOrdered <- thisUniGramDF[order(-thisUniGramDF$freq),]
  thisUniGramDFOrdered$words <- reorder(thisUniGramDFOrdered$words, thisUniGramDFOrdered$freq)
  thisUniGramDFOrdered$percentage <- (thisUniGramDFOrdered$freq / sum(thisUniGramDFOrdered$freq))
  thisUniGramDFOrdered$cumsum <- cumsum(thisUniGramDFOrdered$freq)
  thisUniGramDFOrdered$cumpercentage <- cumsum(thisUniGramDFOrdered$percentage)
  return(thisUniGramDFOrdered)
}


plotNGram <- function(thisDF, nTerms, title)
{
  DFforPlot <- thisDF[1:nTerms,]
  DFforPlot$words <- reorder(DFforPlot$words, DFforPlot$freq)
  p <- ggplot(DFforPlot, aes(x = words, y = percentage)) +
    geom_bar(stat = "identity") +
    ggtitle(title) +
    coord_flip() +
    theme(legend.position = "none")
  return(p)
}


generate_nGrams <- function(thisDF, nValue){
  thisDF <- unlist(thisDF)
  nGramsList <- vector(mode = "character")
  for (i in 1:length(thisDF)) {
    this_nGramsList <- tokenize_ngrams(
      thisDF[i], n = nValue, simplify = FALSE)
    nGramsList <- c(nGramsList, this_nGramsList[[1]])
  }
  return(nGramsList)
}


generate_nGramsDF <- function(thisCorpus, nValue){
  thisDF <- data.frame(text = sapply(thisCorpus, as.character), stringsAsFactors = FALSE)
  thisNGrams <- unname(unlist(sapply(thisDF, generate_nGrams, nValue)))
  thisGramsDF <- data.frame(table(thisNGrams))
  thisGramsDF$percentage <- (thisGramsDF$Freq/sum(thisGramsDF$Freq))
  thisGramsDF <- thisGramsDF[order(-thisGramsDF$Freq),]
  colnames(thisGramsDF) <- c("words","freq","percentage")
  return(thisGramsDF)
}


data_corpus <- cleanCorpus(data_corpus)

#1-GRAMS
UniGramDF <- unigram(data_corpus)
p1 <- plotNGram(UniGramDF, 10, "Top10 Unigram")
grid.arrange(p1)

#2-GRAMS
BiGramsDF <- generate_nGramsDF(data_corpus, 2)
p2 <- plotNGram(BiGramsDF, 10, "Top10 Bigram")
grid.arrange(p2, ncol=1)

#3-GRAMS
TriGramsDF <- generate_nGramsDF(data_corpus, 3)
p3 <- plotNGram(TriGramsDF, 10, "Top10 Trigram")
grid.arrange(p3, ncol=1)


#WORDCLOUD
tdm <- TermDocumentMatrix(data_corpus)
tf <- as.matrix(tdm)

v <- sort(rowSums(tf), decreasing = TRUE)
df_freq <- data.frame(word = names(v), freq = v)
#head(df_freq, 5)
wordcloud(words = df_freq$word, 
          freq = df_freq$freq, 
          min.freq = 10,
          max.words = 100, 
          random.order = FALSE,
          rot.per = 0.0, # proportion words with 90 degree rotation
          colors = brewer.pal(4, "Set1"))
