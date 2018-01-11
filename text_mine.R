library(dplyr)
library(tm)
library(naivebayes)
library(qdap)
library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(pROC)
library(readr)

source("data_prep.R")

# hardcoded seed to reproduce results
set.seed(3)
# Get data and shuffle it up to randomize positions.
input_data <- read_data()
input_data <- input_data[sample(nrow(input_data)),] # Shuffle the data.

# Seperate into a train and test set.
seperate <- round(nrow(input_data) * 0.75, 0)
train_data <- input_data[1:seperate,]
test_data <- input_data[(seperate + 1):nrow(input_data),]

# Create the corpuses. Corpi?
train_corpus <- create_corpus(train_data$review)
test_corpus <- create_corpus(test_data$review)
# Clean corpus.
cleaned_train_corpus <- clean_corpus(train_corpus)
cleaned_test_corpus <- clean_corpus(test_corpus)

# Init the train DTM, removing a bunch of sparse terms.
train_DTM <- create_dtm(cleaned_train_corpus, 0.986)
# We have to remove some sparse terms from the test DTM too, else R crashes due to loading too much data into memory
test_DTM <- create_dtm(cleaned_test_corpus, 0.999)

train_df <- create_dataframe(train_data, train_DTM)
test_df <- create_dataframe(test_data, test_DTM)

# Create a model using naiveBayes approach to classification.
# https://en.wikipedia.org/wiki/Naive_Bayes_classifier
bayes_model <- naiveBayes(select(train_df, -sentiment), train_df$sentiment)
pred <- predict(bayes_model, test_df)
confusionMatrix(pred, test_df$sentiment)
plot(roc(test_df$sentiment, as.numeric(pred)))

# Lets try to find the best accuracy through various sparsity values.
sparsities <- seq(0.900, 0.990, by = 0.001)
accuracies <- c()
amount_of_terms <- c()
# Sparsity exploration hurts a lot! So don't always do this.
# Just make sure it happens at least once to figure out the right sparsity for you.
for(i in 1:91) {
  # Init the train DTM, removing a bunch of sparse terms.
  train_DTM_tmp <- create_dtm(cleaned_train_corpus, sparsities[i])
  train_df_tmp <- create_dataframe(train_data, train_DTM_tmp)
   
  bayes_model_tmp <- naiveBayes(train_df_tmp, train_df_tmp$sentiment)
  pred_tmp <- predict(bayes_model_tmp, test_df)
  
  accuracy <- sum(pred_tmp == test_df$sentiment) / nrow(test_df)
  accuracies[i] <- accuracy
  amount_of_terms[i] <- ncol(train_DTM_tmp)
}

# Single rpart tree without cutting it up
tree_model <- rpart(sentiment ~ ., train_df, method="class")
tree_pred <- predict(tree_model, test_df, type="class")
confusionMatrix(tree_pred, test_df$sentiment)
plot(roc(test_df$sentiment, as.numeric(tree_pred)))

# Random forest, basic AF.
# We must remove this from the dataset because break is an internal function and randomForest is retarded.
removed_breaks <- train_df$"break"
train_df$"break" <- NULL
rf_model <- randomForest(formula=sentiment ~ ., ntree=500, data = train_df)
rf_pred <- predict(rf_model, test_df, type="class")
confusionMatrix(rf_pred, test_df$sentiment)
plot(roc(test_df$sentiment, as.numeric(rf_pred)))
# Totes add this back though. 
# Find correct amount of trees in the forest.
tree_counters <- c(50, 100, 250, 500, 1000)
tree_accuracies <- c()
for(i in 1:5) {
  rf_model_tmp <- randomForest(formula=sentiment ~ ., ntree=tree_counters[i], data = train_df)
  rf_pred_tmp <- predict(rf_model_tmp, test_df, type="class")
  accuracy <- sum(rf_pred_tmp == test_df$sentiment) / nrow(test_df)
  tree_accuracies[i] <- accuracy
}
train_df$"break" <- removed_breaks
