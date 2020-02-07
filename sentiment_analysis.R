# Movie review: Sentiment Analysis











# 1. Install necessary libraries
# install.packages(c('tm', 'SnowballC', 'wordcloud', 'rpart', 'rpart.plot', 'randomForest'), dependencies=T)
# install.packages('caret')
library(tm)
library(SnowballC)
library(wordcloud)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)





# 2. Creating Document-Term Matrix
imdb_train <- read.csv('labeledTrainData.tsv', 
                 sep='\t', 
                 quote='', 
                 stringsAsFactors=FALSE)
imdb_test <- read.csv('testData.tsv',
                      sep='\t',
                      quote='',
                      stringsAsFactors = F)
imdb_test$sentiment <- NA
imdb <- rbind(imdb_train, imdb_test)

names(imdb)
table(imdb$sentiment)
  
# Corpus (from 'tm' package): dump all the data and put them in order
corpus <- Corpus(VectorSource(imdb$review))

# wordcloud
wordcloud(corpus, colors=rainbow(7), max.words=50)

# The top words are: 'and', 'the', which really does not help much understanding
# the true meaning of textual content. Need to do some pre-processing & data cleaning.
corpus <- tm_map(corpus, tolower) # convert all words to lower case

corpus <- tm_map(corpus, removePunctuation) # remove all punctuations in Term Document Matrix

stopwords('english')[1:10] # show the first 10 English Stop Words in 'tm' package

# remove Stop Words and some other popular words
corpus <- tm_map(corpus, removeWords, c('and', 'the', 'film', 'movi', stopwords('english')))

# To Stem Document: compress a words from various tenses to single basic word
corpus <- tm_map(corpus, stemDocument)

# Every word left in corpus becomes a corpus header
frequencies <- DocumentTermMatrix(corpus)

# take a look at the matrix. This is a very sparse matrix
inspect(frequencies[1000:1005, 505:515])

# find words that appear more than 20 times
findFreqTerms(frequencies, lowfreq=20)

# sparse: the percentage of sparsity (0 frequency: sparsity=1; but never actually reach 1)
# only terms with less than 0.92 sparse are retained, remove terms that have sparse > 0.92
# We are left with around only 200 words
sparse <- removeSparseTerms(frequencies, 0.92)

# matrix must be converted to dataframe with variables=terms in Document-Term-Matrix
imdbsparse <- as.data.frame(as.matrix(sparse))

# make all variables names R-friendly
colnames(imdbsparse) <- make.names(colnames(imdbsparse))
length(names(imdbsparse))

# Add dependent variables
imdbsparse$sentiment <- imdb$sentiment














# Build CART model (Using Decision Tree)
imdbCART <- rpart(sentiment~., data=imdbsparse[!is.na(imdbsparse$sentiment),], method='class')
rpart.plot(imdbCART)
prp(imdbCART, extra=2) # plot the RPART tree model

# prediction on training data
imdbCARTpred <- predict(imdbCART, data=imdbsparse[!is.na(imdbsparse$sentiment),])
imdbCARTCM <- table('prediction'=imdbCARTpred[, 2] > 0.5, 'Actual'=imdb_train$sentiment) # confusion matrix
imdbCARTCM

# accuracy
accuracyCART <- (imdbCARTCM[1]+imdbCARTCM[4])/sum(imdbCARTCM)
round(accuracyCART*100, 2)

# precision & recall
recall <- imdbCARTCM[4]/(imdbCARTCM[4]+imdbCARTCM[2])
precision <- imdbCARTCM[4]/(imdbCARTCM[4]+imdbCARTCM[3])
F1Score_CART <- ((2*recall*precision)/(recall+precision))
F1Score_CART









# Submission
imdbtest_CART <- predict(imdbCART, data=imdbsparse[is.na(imdbsparse$sentiment),]) # Need to process like in training set before predicting?
imdbtest_CART$sentiment <- imdbtest_CART[, 2] > 0.5 # probabilities greater than 0.5 is True
imdbtest_CART <- data.frame(id=imdb_test$id, sentiment=imdbtest_CART$sentiment)
imdbtest_CART$sentiment <- as.integer(imdbtest_CART$sentiment) # convert T/F to 1/0
dim(imdbtest_CART)
any(is.na(imdbtest_CART))

write.csv(imdbtest_CART, file="cartmodel_processed.csv", row.names = F, quote=F) # remove quotes from Character variables










# Using Random Forest (RF)
imdbRF <- randomForest(data=imdbsparse[!is.na(imdbsparse$sentiment),],
                       sentiment~.,
                       ntree=50)
varImpPlot(imdbRF)

# Predicting:
imdbRFCM <- table('Actual'=imdb_train$sentiment, 'prediction'=imdbRF$predicted>0.5)
imdbRFCM

# accuracy
accuracyRF <- (imdbRFCM[1]+imdbRFCM[4])/sum(imdbRFCM)
round(accuracyRF*100, 2)

# recall & precision
recall <- imdbRFCM[4]/(imdbRFCM[4]+imdbRFCM[2])
precision <- imdbRFCM[4]/(imdbRFCM[4]+imdbRFCM[3])
F1Score_RF <- ((2*recall*precision)/(recall+precision))
F1Score_RF

# Predict on Test data
imdbtest_RF <- predict(imdbRF, data=imdbsparse[is.na(imdbsparse$sentiment),])
imdbtest_RF$sentiment <- imdbtest_RF>0.5

imdbtest_RF <- data.frame(id=imdb_test$id, sentiment=imdbtest_RF$sentiment)
imdbtest_RF$sentiment <- as.integer(imdbtest_RF$sentiment) # convert T/F to 1/0
dim(imdbtest_RF)
any(is.na(imdbtest_RF$sentiment))

write.csv(imdbtest_RF, file="RF_processed.csv", row.names = F, quote=F) # remove quotes from Character variables

# Predict on unprocessed Test data
imdb_test <- read.csv('testData.tsv',
                      sep='\t',
                      quote='',
                      stringsAsFactors = F)
imdbtest_RF <- predict(imdbRF, data=imdb_test)
imdbtest_RF$sentiment <- imdbtest_RF>0.5

imdbtest_RF <- data.frame(id=imdb_test$id, sentiment=imdbtest_RF$sentiment)
imdbtest_RF$sentiment <- as.integer(imdbtest_RF$sentiment) # convert T/F to 1/0
dim(imdbtest_RF)
any(is.na(imdbtest_RF$sentiment))

write.csv(imdbtest_RF, file="RF_unprocessed.csv", row.names = F, quote=F) # remove quotes from Character variables









# Using GLM - Logistic Regression
imdblr <- glm(data=imdbsparse[!is.na(imdbsparse$sentiment),],
              sentiment~.,
              family='binomial')
imdblrCM <- table('Actual'=imdb_train$sentiment, 'Prediction'=imdblr$fitted.values>0.5)
summary(imdblr)

# p-value and z-value of each coefficient (which is each word in this case)
round((1-(imdblr$deviance/imdblr$null.deviance))*100, 2)
words <- as.data.frame((coef(summary(imdblr))[coef(summary(imdblr))[,4] < 0.05,3:4]))
colnames(words) <- c('pvalue', 'zvalue')
words

# accuracy
accuracylr <- (imdblrCM[1]+imdblrCM[4])/sum(imdblrCM)
round(accuracylr*100, 2)

# recall & precision for F1 score
recall <- imdblrCM[4]/(imdblrCM[4]+imdblrCM[2])
precision <- imdblrCM[4]/(imdblrCM[4]+imdblrCM[3])
F1Score_glm <- ((2*recall*precision)/(recall+precision))
F1Score_glm

# accuracy and F1score of all models
round(c('Accuracy_CART'=accuracyCART, 'Accuracy_RF'=accuracyRF, 'Accuracy_GLM'=accuracylr)*100, 2)
round(c('F1Score_CART'=F1Score_CART, 'F1Score_RF'=F1Score_RF, 'F1Score_GLM'=F1Score_glm)*100, 2)




# Submission
imdbtest_lr <- predict(imdblr, data=imdbsparse[is.na(imdbsparse$sentiment),])
imdbtest_lr$sentiment <- imdbtest_lr>0.5 # use a different variable other than the predicted outputs?

imdbtest_lr <- data.frame(id=imdb_test$id, sentiment=imdbtest_lr$sentiment)
imdbtest_lr$sentiment <- as.integer(imdbtest_lr$sentiment)
dim(imdbtest_lr)
any(is.na(imdbtest_lr$sentiment))

write.csv(imdbtest_lr, file='logistic_regression_processed.csv', row.names=F, quote=F)
