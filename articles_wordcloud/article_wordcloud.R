
library(tm) #Text Mining Package

library(dplyr) # Data Manipulation

library(textstem) #Text Lemmatization and stemming

library(word2vec)#Toc convert words to vector embeddings

library(Rtsne) #Dimensionality reduction

library(wordcloud2) # Wordcloud package

library(ggplot2) #plotting library

#Data Reference: https://www.kaggle.com/datasets/hsankesara/medium-articles


#Reading the data
article_df <- read.csv('articles.csv',header=TRUE)

#Loading the data as a corpus
medium_articles <- Corpus(VectorSource(article_df$text))

#Applying the transformation to the text data

remove_emojis_text <- function(text) {
  gsub("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]", "", text, perl = TRUE)
}


#Cleaning the data
clean_medium_articles <- function(medium_articles){
  medium_articles %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(stripWhitespace) %>%
  tm_map(content_transformer(remove_emojis_text))%>%
  tm_map(content_transformer(function(x) gsub("[^[:alnum:] ]", " ", x)))
  
}

medium_articles <- clean_medium_articles(medium_articles)

#Text stemming to reduce words to their base form
#medium_articles <- tm_map(medium_articles,stemDocument)

text_matrix <- TermDocumentMatrix(medium_articles)%>%as.matrix()

v <- sort(rowSums(text_matrix), decreasing = TRUE) 

d <- data.frame(word = names(v), freq=v)

head(d,10)

set.seed(1234)

findFreqTerms(dtm, lowfreq = 4)
findAssocs(dtm, terms = "freedom", corlimit = 0.3)
head(d,10)

high_freq10_words <- d[order(-d$freq),][1:10,]


#Lemmatizing the text to convert the text to their intial form
lemmatize_text <- lemmatize_words(d$word)

lemmatized_words <- unlist(lemmatize_text)
new_freq_1 <- tapply(d$freq, lemmatized_words, sum)

lemmatized <- data.frame(word = names(new_freq_1), freq = new_freq_1)

length(names(new_freq_1))

length(new_freq_1)


# Plotting the results

# Generate the word cloud in a specific shape

gradient_colors <- colorRampPalette(c("white", "lightblue"))(100)

worcloud_plot <- wordcloud2(max,data = lemmatized, size = 1, color = gradient_colors,rotateRatio = 0)

worcloud_plot

# the barplot on word frequencies
ggplot(high_freq10_words, aes(x = reorder(word, freq), y = freq)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  geom_text(aes(label = freq), vjust = -0.3, size = 3) +
  coord_flip() +
  labs(title = "Most Frequent Words",
       x = "Words",
       y = "Word Frequencies") +
  theme_minimal() +
  theme(axis.text.y = element_text(angle = 0))








