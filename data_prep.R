library(readr)
library(dplyr)
library(tm)
library(naivebayes)
library(qdap)
library(caret)
library(e1071)

# Read data if it doesn't exist. Returns the data anyway.
read_data <- function() {
  if(!exists("input_data")) {
    # Read in data from .tsv file. Keep header.
    input_data <- read.table(file = 'labeledTrainData.tsv', sep = '\t', header = TRUE, stringsAsFactors = TRUE) %>%
      select(-id)
    
    stanford_data <- read.csv("stanford_data.csv") %>%
      select(-X)
    
    # Add data from stanford dataset.
    input_data <- rbind(input_data, stanford_data)
    
    input_data$sentiment <- factor(input_data$sentiment, levels=c(0, 1), labels=c("negative", "positive"))
    
  }
  return(input_data)
}

# Function to load in the standford review files into a dataframe.
# Really really slow due to huge amount of files!
# You should probably manually write this to another file to make loading it much much easier.
read_stanford_data <- function() {
  stanford_neg_path <- "./stanford_data/neg/"
  stanford_pos_path <- "./stanford_data/pos/"
  
  full_data_frame <- data.frame(review=c(), sentiment=c())
  for(file in list.files(stanford_neg_path)) {
    read_review <- read_file(paste(stanford_neg_path, file, sep=""))
    file_data_frame <- data.frame(review=c(read_review), sentiment=c(0))
    full_data_frame <- rbind(full_data_frame, file_data_frame)
  }
  for(file in list.files(stanford_pos_path)) {
    read_review <- read_file(paste(stanford_pos_path, file, sep=""))
    file_data_frame <- data.frame(review=c(read_review), sentiment=c(1))
    full_data_frame <- rbind(full_data_frame, file_data_frame)
  }
  write.csv(full_data_frame, "stanford_data.csv")
  return(full_data_frame)
}

# Helper to gsub some things out of the text, replace them with spaces.
toSpace <- content_transformer(function(x, pattern) {return (gsub(pattern, " ", x))})

# Clean a corpus using tm functions.
clean_corpus <- function(corpus) {
  corpus <- tm_map(corpus, toSpace, "-")
  corpus <- tm_map(corpus, toSpace, ":")
  corpus <- tm_map(corpus, toSpace, "'")
  corpus <- tm_map(corpus, toSpace, "'")
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"), stopwords("SMART")))
  corpus <- tm_map(corpus, stripWhitespace)
  
  return(corpus)
}

qdap_clean <- function(x){
  x <- clean(x)
  x <- strip(x)
  return(x)
}

# Convert available 
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# Create a corpus based on a text vector.
create_corpus <- function(text) {
  corpus <- Corpus(VectorSource(qdap_clean(text)))
  return(corpus)
}

# Create a DocumentTermMatrix from a corpus and remove sparse terms with given sparsity.
create_dtm <- function(corpus, sparsity) {
  DTM <- DocumentTermMatrix(corpus) %>%
    removeSparseTerms(., sparsity)
  return(DTM)
} 

# Combines data and it's DTM into a useable dataframe.
create_dataframe <- function(data, DTM) {
  df <- as.data.frame(as.matrix(DTM))
  # Convert variables to proper catagorical variables.
  df <- apply(df, MARGIN = 2, convert_counts)
  # cbind the data together with the dtm data to include sentiment.
  full_df <- cbind(data, df)
  # Unset the review column to ensure the full review isn't used 
  # in creating the model / predicting based on it.
  full_df$review <- NULL
  
  return(full_df)
}
