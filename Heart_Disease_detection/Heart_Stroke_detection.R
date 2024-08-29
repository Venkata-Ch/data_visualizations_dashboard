library("readr") #Library to read embedded nulls

library("dplyr") #Data Manipulation  

library("MASS") #Dimensionality Reduction

library("corrplot") #Feature selection

library("ggplot2") #Data Visualization

library("rpart") 
#Dataset Reference: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

heart_df <- read_csv('heart_2022_with_nans.csv')

heart_df <- data.frame(heart_df)

summary(heart_df)

#Data Cleaning

clean_df <- na.omit(heart_df)

summary(clean_df)

#Dropping irrelevant columns

clean_df <- clean_df[, !names(clean_df) %in% c("LastCheckupTime","RemovedTeeth","TetanusLast10Tdap","RaceEthnicityCategory","ECigaretteUsage")]


#EDA Plots
#heatmap
health_heatmap <- clean_df %>%
  group_by(HadHeartAttack, GeneralHealth) %>%
  summarize(BMI_mean = mean(BMI, na.rm = TRUE), .groups = 'drop')

# Create the heatmap
ggplot(health_heatmap, aes(x = HadHeartAttack, y = GeneralHealth, fill = BMI_mean)) +
  geom_tile() +
  scale_fill_gradient(low = "brown", high = "lightpink") +
  labs(title = "Heatmap of mean BMI with Heart Attack Status by General Health",
       x = "Had Heart Attack",
       y = "General Health") +
  theme_minimal()
#Barplot
ggplot(clean_df, aes(x = GeneralHealth)) +
        geom_bar() +
        labs(title = "Counts by General Health", x = "General Health", y = "Count")

#Encoding the object or character columns for analysis


#Applying label encoding to binary columns
binary_columns <- c("PhysicalActivities", "HadHeartAttack", "HadAngina","HadStroke", "HadAsthma", "HadSkinCancer",
          "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes", "DeafOrHardOfHearing", 
          "BlindOrVisionDifficulty", "DifficultyConcentrating","DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands",
          "ChestScan", "AlcoholDrinkers", "HIVTesting","FluVaxLast12", "PneumoVaxEver", "HighRiskLastYear", "CovidPos","Sex")

clean_df[binary_columns] <- lapply(clean_df[binary_columns], function(x) as.numeric(factor(x)))
   

#Applying ordinal encoding for ordinal columns
#GeneralHealth
health_categories <- unique(clean_df$GeneralHealth)

clean_df$GeneralHealth <- factor(clean_df$GeneralHealth, levels = health_categories, ordered=TRUE)

clean_df$GeneralHealth <- as.numeric(clean_df$GeneralHealth)

#SmokerStatus 
sm_categories <- unique(clean_df$SmokerStatus)

clean_df$SmokerStatus <- factor(clean_df$SmokerStatus, levels = sm_categories, ordered=TRUE)

clean_df$SmokerStatus <- as.numeric(clean_df$SmokerStatus)

#AgeCategory
age_categories <- unique(clean_df$AgeCategory)

clean_df$AgeCategory <- factor(clean_df$AgeCategory, levels = age_categories, ordered=TRUE)

clean_df$AgeCategory <- as.numeric(clean_df$AgeCategory)

#State
states <- unique(clean_df$State)

clean_df$State <- factor(clean_df$State, levels = states, ordered=TRUE)

clean_df$State <- as.numeric(clean_df$State)

#Correlation plot for feature selection

corr_matrix <- cor(clean_df[, sapply(clean_df, is.numeric)])

corrplot(corr_matrix, method = "square", type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 60, 
         number.cex = 0.5,  
         tl.cex = 0.5) 
#Highly correlated features are WeightInKilograms, HadDiabetes, DifficultyConcentrating,MentalHealthDays,PhysicalHealthDays,DifficultyWalking,DifficultyDressingBathing

#Tree model
set.seed(123)
train_index <- sample(nrow(clean_df), 0.7 * nrow(clean_df))

train_data <- clean_df[train_index, ]
test_data <- clean_df[-train_index, ]
tree_model <- rpart(GeneralHealth ~ ., data = train_data, method = "class", minsplit = 2, minbucket =1)
plot(tree_model)
text(tree_model)
predictions <- predict(tree_model, test_data, type = "class")



library(MASS)


# To predict the PhysicalHealth variable
# using the AgeCategory and BMI variables
lda_data <- x[, c("WeightInKilograms", "HadDiabetes", "DifficultyConcentrating","MentalHealthDays","PhysicalHealthDays","DifficultyWalking","DifficultyDressingBathing")]


# Fit the LDA model
lda_model <- lda(lda_data, clean_data$HadHeartAttack)

# Predict the test set
lda_predictions <- predict(lda_model, newdata = test_data)

# View the confusion matrix
table(lda_predictions$class, test_data$HadHeartAttack)








