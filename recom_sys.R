#---------------------------------------------------------#
# title: "Recommendation System for: MovieLens"
# author: "Rolf Beutner, Germany, Bamberg"
# date: "04/15/2021 19:50"
#---------------------------------------------------------#



#---------------------------------------------------------#
# Create edx set, validation set (final hold-out test set)
#---------------------------------------------------------#

# Note: this process could take a couple of minutes



if(!require(dplyr     )) install.packages("dplyr"     , repos = "http://cran.us.r-project.org")
if(!require(tidyverse )) install.packages("tidyverse" , repos = "http://cran.us.r-project.org")
if(!require(caret     )) install.packages("caret"     , repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(dplyr, warn.conflicts = FALSE)
library(tidyverse)
library(caret)
library(data.table)

# Suppress summarize info - annoying in report and does not help
options(dplyr.summarise.inform = FALSE)

# ``` ---------------------------------------------------------------------------------------------\
#
# Overview ----
# 
# This report is related to the MovieLens Project of the HarvardX: PH125.9x Data Science: Capstone. 
#
# It consists of seven parts:
#
# It starts with this **Overview**, followed by a short **Introduction**.
#
# A **Summary** section  describes how to download and prepare the datasets **edx** and **validation**. 
#
# Next section **Investigation** describes the performed exploratory data analysis which provided an overview of the data and first approaches to optimize the machine learning algorithm.
#
# In the **Methods and Analysis** section, **train**- and **test**-datasets are generated and a machine learning algorithm is stepwise developed that predicts movie ratings based on that sets. 
#
# The **Results** section explains the final model and applies it to the validation-dataset. 
#
# The report ends with a **Conclusion** on the findings and possible further steps.
# 
# # Introduction
# 
# Today machine learning is becoming more and more important in business- and everyday life.
# It is used for business i.e. to explore data about customer bases, for single user it can personalize experiences like this for the selection of the next movie. 
# Machine learning helps extract usefiul insights from data. Effective machine learning algorithms are used to obtain these insights within a short period of time.
# 
# In this project we aim to predict movie ratings for users based on a large dataset and develop an effective machine learning algorithm, suggest movies based on previous preferences of other uses, and ratings of similar movies.
# 
# Therefore a linear model is trained to generate these predictions.
# To evaluate the quality of the predictions we use the Root Mean Square Error (RMSE) to calculate the distance between the predicted ratings and the actual ratings.
# 
# # Summary
# 
# The following dataset is used from the following directory:
#   
#   * [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m/) 
# 
# * [ml-10m.zip](http://files.grouplens.org/datasets/movielens/ml-10m.zip)
# 
# This dataset downloaded and then splitted into 2 subsets:
#   
#   * **edx** - a subset to work with and to develop the models, and 
# 
# * **validation** - a subset to test the final model
# 
# The **edx** dataset is then split further into 2 subsets:
#   
#   * **trainset** - for training the models
# 
# * **testset**  -for testing the models
# 
# ## Download
# 
# The following code downloads the data and creates the 2 subsets **edx** and **validation**.
# 
# 
# It is instrumented for switching between:
#   
#   * download and separate datasets from grouplens or
# 
# * load from local file system
# 
# * size of datasets
# 
# ** large (the final 10M target dataset) or 
# 
# ** small dataset to test the program (1M dataset, same format, 10% of size). 
# 
# The final version will download AND use the 10M dataset of course.
# 
# ```{r initial_load1, warning=FALSE, message=FALSE, echo=TRUE}

evaluate=2     # switch: 2 = download, 1/2 = pre-evaluation/prepare, 0 = load of formerly prepared data from file 
scalefactr=10  # switch: scale factor for model: 1 = 1M, 10 = 10M movielens-set

if(evaluate>0) { 
  dl <- tempfile()
  mfile  = paste("http://files.grouplens.org/datasets/movielens/ml-",scalefactr,"m.zip", sep = "")
  ratf   = paste("ml-",ifelse(scalefactr==1,"1m","10M100K"),"/ratings.dat", sep = "")
  movf   = paste("ml-",ifelse(scalefactr==1,"1m","10M100K"),"/movies.dat", sep = "") 
  if((evaluate>1) | !(file.exists(ratf) & file.exists(movf))) {
   download.file(mfile, dl)
   unzip(dl, ratf)
   unzip(dl, movf)
}
ratings <- fread(text = gsub("::", "\t", readLines(ratf)),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(movf), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# prepare datasets -------------------------------------------
# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title   = as.character(title),
                                           genres  = as.character(genres))
# if using R 4.0 or later:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                            title = as.character(title),
#                                            genres = as.character(genres))

# join both data sets ----------------------------------------
movielens <- left_join(ratings, movies, by = "movieId")

# divide data sets train/test --------------------------------
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed, ratf, movf)

 ## Save datasets to a file if freshly downloaded
 saveRDS(edx       , "edx.rds")
 saveRDS(validation, "validation.rds")
} else {
 ## load objects from file if downloaded / prepared before
 edx        <- readRDS("edx.rds")
 validation <- readRDS("validation.rds")
}

# ``` ---------------------------------------------------------------------------------------------\
#
# Investigations ----
#
## Exploratory analysis ----
# 
# The grouplens-dataset was split in the ratio 90-10 into the **edx**- and **validation**-dataset. 
# 
# For the investigations here the **edx** dataset is taken.
# 
# For the development in the next chapter then the **trainset** and **testset** will be taken
# 
# 
# The training set **edx** has 9,000,055 entries with 6 columns. 
# 
# Similarly, the test set **validation** has 999,999 entries and 6 columns.\
# The column information is shown below.
# 
# ```{r initial_inquiries0a, echo=TRUE} 

dim(edx)
dim(validation)

# ``` ---------------------------------------------------------------------------------------------\
# 
# In the following the investigations are done with the **edx** dataset.
# The **validation** dataset will be used when a model is created and shall be tested.
# 
# The ratings in the edx dataset were applied by nearly 70.000 user and more than 10600 movies   
# 
# ```{r initial_inquiries0b, echo=TRUE} 

edx %>% summarize(no_users  = n_distinct(userId), 
                  no_movies = n_distinct(movieId)) %>% knitr::kable()

# ``` ---------------------------------------------------------------------------------------------\
# 
## Quick preview of "edx" ----
# 
# Six columns with the following to be investigated more:
#   
#   * rating    - the center of interest - given by user with "userId" for movie with "movieId"
# 
# * genres    - single pipe-delimited string containing the different genre categories 1:n
# 
# * timestamp - needs to be converted
# 
# * title     - has release year as appendix, worth to split from title
# 
# ```{r initial_inquiries1, echo=TRUE}

head(edx) %>% knitr::kable() #'simple')

# ``` ---------------------------------------------------------------------------------------------\
#
## Rating of movies ----
# 
# There is a broad set of ratings per movie - at the top *Pulp Fiction* with more than 31,000 ratings
# 
# ```{r initial_inquiries2, echo=TRUE}

edx %>% group_by(title) %>%
  summarize(no_of_ratings = n()) %>%
  arrange(desc(no_of_ratings)) %>%
  head(5) %>% knitr::kable() #'simple')

# ``` ---------------------------------------------------------------------------------------------\
#
# and on the end: 126 titles with just one single rating.
# 
# ```{r initial_inquiries3, echo=TRUE}

edx %>% group_by(title) %>%
  summarize(no_of_ratings = n()) %>%
  filter(no_of_ratings==1) %>%
  count() %>% pull()

# ``` ---------------------------------------------------------------------------------------------\
# 
## Further checks: ----
#   
#   * All movies are categorized with genres,
# 
# * There is no movie which has been never rated.
# 
# ```{r initial_inquiries4, echo=TRUE} 

sum(is.na(edx$rating))
sum(is.na(edx$genres))

# ``` ---------------------------------------------------------------------------------------------\
# 
# ### Movies vs. Users rating - distribution, histograms and crosstab
# 
# We have the following distribution of the ratings and the histograms of ratings related to users and movies. So there are users who rate more than others and also movies which are rated more than others. When we put both in a cross tab to see what user has rated what movie we see a sparse - but very big matrix.
# 
# #### Histograms
# 
# The mean rating distribution for movies shows slightly skewness to left or right - that is an indicator to use it as influencing factor for our prediction algorithm.
# 
# The distribution of user ratings shows that many users have watched or rated less than half of all movies in the dataset as shown in 2nd chart below.
# 
# ```{r initial_inquiries5b, echo=TRUE}

edx %>%
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Rating distribution for movies") + 
  labs(x="Number of Ratings (log 10)", y="Number of Movies", 
       caption = "many users tend to rate movies highly (skewness to the right)") 

edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Rating distribution for users") + 
  labs(x="Number of ratings (log 10)", y="Number of users", 
       caption = "many users watched or rated < half of all movies  (skewness to the left)") 

# Plot mean movie ratings given by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Mean movie ratings given by users")  + 
  labs(x="Mean rating", y="Number of users", 
       caption = "many users tend to rate movies highly  (skewness to the right)") 

# ``` ---------------------------------------------------------------------------------------------\
# 
## Crosstab/Heatmap user vs. movies ----
# 
# When we put user vs. movies  in a cross tab to see what user has rated what movie, the result a sparse,
# but very big matrix - here only shown a sample of 100 user (rows) and 200 movies (columns):
#   
#   All nonzero values are represented by a dot. Empty spaces means no ratings were given by the user. 
# 
# ```{r initial_inquiries6, echo=TRUE}

users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 200)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:200, 1:100,. , xlab="Movies\nsparse matrix - just 100 user vs. 200 movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# ``` ---------------------------------------------------------------------------------------------\
# 
## Genres ----
# 
# The genres have a broad set too - here those where the movies have **more than 120.000 ratings**.
# The average rating per genre is significantly at different levels.
# 
# So as one can see the average ratings for **Crime/Drama** are substantially higher than for **Comedy**.
# **This will be considered later as tuning parameter**.
# 
# ```{r initial_inquiries7, echo=TRUE}

edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 12000*scalefactr) %>%    # choose n that the graphics is readable
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_errorbar(width = 0.6) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x="genre", y="average rating", caption = paste("different genres are rated significantly different (here #ratings > ",12000*scalefactr),")") + 
  ggtitle("Average ratings per genre")

# ``` ---------------------------------------------------------------------------------------------\
# 
## Timestamp and year tag ----
# 
# We put some effort in investigations around the two timestamps given by the data,
# the unix timestamp of the rating and the year of production / release of the movie 
# and we check against the genres to detect patterns.
# The date we need later as join criterion.
# 
# We see that older movies have better ratings - maybe an approach to penalize. 
# The second graph shows that the date of the rating has not much effect, see the x- and y-scale: it does just cover a short time period and has just a small offset to the median around 3.5.
# 
# ```{r initial_inquiries8, warning=FALSE, message=FALSE, echo=TRUE}

# TEST title=" xsdhvhdsn  (11) xx vcsa (1996)" 
# str_extract(str_extract(title, "[/(]\\d{4}[/)]$"),regex("\\d{4}"))
edx <- edx %>% mutate(year_prod = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"),regex("\\d{4}"))),
                      year_rat  = strftime(as.Date(as.POSIXct(timestamp,origin='1970-01-01')),"%Y"),
                      title     = str_remove(title, "[/(]\\d{4}[/)]$"),
                      date      = strftime(as.Date(as.POSIXct(timestamp,origin='1970-01-01')),"%Y-%m")) # join crit.
#saveRDS(edx, "edxY.rds") #edx <- readRDS("edxY.rds")  # for speed up / reentry point

# View production year vs rating
edx %>% group_by(year_prod) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year_prod, rating)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  geom_smooth() +
  ggtitle("Production Year <-> Rating") +
  labs(caption = "Older movies have better ratings - maybe an approach to penalize") 

edx %>% group_by(year_rat) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year_rat, rating)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  geom_smooth() +
  ggtitle("Rating Year <-> Rating") +
  labs(caption = "Note: year_prod above is more useful")  

# ``` ---------------------------------------------------------------------------------------------\
# 
### Movies per year ----
# 
# ```{r initial_inquiries9a, echo=TRUE}

movies_per_year <- edx %>%
  select(movieId, year_prod) %>% 
  group_by(year_prod) %>%   
  summarise(count = n())  %>% 
  arrange(year_prod)

movies_per_year %>%
  ggplot(aes(x = year_prod, y = count)) +
  geom_line(color="black") +
  ylab('Number of Movies')  +
  ggtitle('Movies per year') +
  labs(caption = "exponential growth of the movie business up to 2010 - no more younger data")

# ``` ---------------------------------------------------------------------------------------------\
# 
### Genres per year ----
# 
# We could separate the genres-string into own attributes / features.
# We will check  later - first analysis showed: the **1:n relation blows up to more than 23 mio rows**
#   and the genres per year seem to follow the movies per year.
# So here must be investigated more work, i.e. normalize per movie per year.
# 
# ```{r initial_inquiries9b_LOOONG, echo=FALSE}

# year_genres <- edx %>% 
#   separate_rows(genres, sep = "\\|") # that is time consuming - blows up to 23mio rows / 190MB file!!
# #saveRDS(year_genres, "year_genres.rds") #year_genres <- readRDS("year_genres.rds")  # for speed up / reentry point
# 
# year_genres %>% 
#   select(movieId, year_prod, genres) %>% 
#   group_by(year_prod, genres) %>% 
#   summarise(count = n()) %>% 
#   arrange(desc(year_prod))
# 
# year_genreM <- year_genres %>%
#   filter(count >= 2500*(scalefactr))  # to make it not so noisy
#   
# # Different periods - different popularity of genres
# ggplot(year_genreM, aes(x = year_prod, y = count)) + 
#   geom_line(aes(color = factor(genres))) + 
#   ylab('Number of Movies')  +
#   ggtitle('Genres per year') +
#   labs(caption = "genres in different periods - maybe take into account later")  

# ``` ---------------------------------------------------------------------------------------------\
# 
# Methods and Analysis ----
# 
# The **edx**-dataset is very big as shown above. 
# 
# To find out the **best recommendation for a new movie** based on movies or people "close" to the asking person could lead to the idea to use the methods for solving a "nearest neighbor problem". My attempt with a nearest neighbor algorithm failed due to lack of resources.
# 
# So the approach recommended in the book for large datasets was taken: **a linear model**.
# 
# https://rafalab.github.io/dsbook/large-datasets.html#recommendation-systems 
# 
# As prerequisite to develop the machine learning models we will 
# create now the **train**- and **test**-datasets from the **edx** dataset.
# 
## Create Train- and Test-dataset
#
# The **edx**-dataset is split in the ratio 90-10 into **trainset** and **testset** for the further development of the algorithm. 
#
# ```{r initial_load2, echo=TRUE}

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
trainset <- edx[-test_index,]  # 90%
temp     <- edx[test_index,]   # 10%

# Make sure userId and movieId in validation set are also in edx set
testset <- temp %>% 
  semi_join(trainset, by = "movieId") %>%
  semi_join(trainset, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, testset)
trainset <- rbind(trainset, removed)

rm(temp)

# ``` ---------------------------------------------------------------------------------------------\
#
## Linear Model ----
# 
# The RMSE function as described in chapter 33.7.3 is used for measuring the accuracy of different approaches.
# 
# ```{r Loss function RSME, echo=TRUE}

## RSME ----

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# ``` ---------------------------------------------------------------------------------------------\
# 
### Model 0: Naive model ----
# 
# Following the book we start with the simplest model, the naive approach:
#   We predict the same rating for all movies independent of user, movie or genre.
# The estimate that minimizes the RMSE is the least squares estimate of mu and, 
# in this case, is the average of all ratings.
# 
# Result: 3.512 (`mean(trainset$rating)`). 
# 
# (using the code of Course Section 6: Model Fitting and Recommendation Systems /  6.2: Recommendation Systems)
#
# ```{r naive_model1, echo=TRUE}

mu_hat <- mean(trainset$rating)
mu_hat

# ``` 
# 
# If we naivly predict all unknown ratings with mu_hat we obtain the following RMSE.
# 
# We collect the results in the **rmse_results** data structure
# 
# ```{r just_average_model2, echo=TRUE}

naive_rmse <- RMSE(testset$rating, mu_hat)
naive_rmse

predictions <- rep(2.5, nrow(testset)) 
RMSE(testset$rating, predictions)   

predictions <- rep(3  , nrow(testset))  
RMSE(testset$rating, predictions)    

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse) 
rmse_results

rm(predictions) # cleanup, is big table!

# ``` ---------------------------------------------------------------------------------------------\
# 
### Model 1: Movie effect - multi-variate model ----
# 
# Now we want to improve the model and add to it movie effects.
# Different movies are rated differently as shown in previous chapter. 
# This is done by adding an error term $b_i$ to represent average ranking for movie i:
#   
#   $$ Y_{i} = \mu + b_i  => b_i = Y_{i,u} - \mu  \ \ (1) $$ 
#   
#   ```{r movie_effect_model1_1, echo=TRUE}

mu <- mean(trainset$rating) 
movie_avgs <- trainset %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))         # (1)
movie_avgs$b_i[is.na(movie_avgs$b_i)] <- 0 # remove NA

# ``` 
# 
# We can see that these estimates vary not much, a few outliers:
#   
#   ```{r movie_effect_model1_2, echo=TRUE}

movie_avgs %>% qplot(b_i, geom ="histogram", 
                     bins = 10, data = ., color = I("black"),
                     ylab = "Number of movies", main = "Histogram of movies compared with the computed b_i")

# ``` 
# 
# Check how much it improves the model, we test it with the validation-set,
# compare it by RSME with reality and add the result to the summary table.
# 
# ```{r movie_effect_model1_3, echo=TRUE}

predicted_ratings <- mu + testset %>%
  left_join(movie_avgs, by='movieId') %>% 
  pull(b_i)

model_1_rmse <- RMSE(predicted_ratings, testset$rating)
model_1_rmse

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                 RMSE = model_1_rmse ))
rmse_results %>% knitr::kable() #'simple')

# ``` ---------------------------------------------------------------------------------------------\
# 
### Model 2: User effect - multi-variate model ----
# 
# Add the user bias term $b_u$ to improve the model. 
# This minimizes the effect of extreme ratings made by users that love or hate every movie
# 
# $$ Y_{i,u} = \mu + b_i + b_u  => b_u = Y_{i,u} - mu - b_i  \ \  (2) $$
#   
#   ```{r movie_user_effect_model2_1, echo=TRUE}

user_avgs <- trainset %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))                      # (2)
user_avgs$b_u[is.na(user_avgs$b_u)] <- 0 # remove NA

# ``` ---------------------------------------------------------------------------------------------\
# 
# We can see that these estimates changes indeed:
#   
#   ```{r movie_user_effect_model2_2, echo=TRUE}

user_avgs %>% qplot(b_u, geom ="histogram", 
                    bins = 30, data = ., color = I("black"),
                    ylab = "Number of users", main = "Histogram of users compared with the computed b_u")

# ``` ---------------------------------------------------------------------------------------------\
#
# Check how much it improves the model, we test it with the validation-set,
# compare it by RSME with reality and add the result to the summary table.
#
# ```{r movie_user_effect_model2_3, echo=TRUE}

predicted_ratings <- testset %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs , by='userId' ) %>%
  mutate(pred = mu + b_i + b_u      ) %>%              # (2)
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, testset$rating)
model_2_rmse

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                 RMSE = model_2_rmse))
rmse_results %>% knitr::kable() #'simple')

rm(predicted_ratings) # cleanup, is big table!

# ``` ---------------------------------------------------------------------------------------------\
# 
### Model 3: Genre effect - multi-variate model ----
# 
# Next we add a genres bias term $b_g$ to possibly further improve our model. This term considers that same genres get similar ratings, some genres tend to get better ratings than others.
# The updated model is:
#   
#   $$ Y_{i,u,g} = \mu + b_i + b_u + b_g => b_g = Y_{i,u,g} - mu - b_i - b_u \ \ (3) $$
#   
#   ```{r movie_user_genre_effect_model3_1, echo=TRUE}

genre_avgs <- trainset %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs , by='userId' ) %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))                     # (3)
genre_avgs$b_g[is.na(genre_avgs$b_g)] <- 0 # remove NA

# ```
# 
# We can see that these estimates vary slightly:
#   
#   ```{r movie_user_genre_effect_model3_2, echo=TRUE}

genre_avgs %>% qplot(b_g, geom ="histogram", 
                    bins = 30, data = ., color = I("black"),
                    ylab = "Number of genres", main = "Histogram of genres compared with the computed b_g")

# ```
# 
# Check how much it improves the model, we test it with the validation-set,
# compare it by RSME with reality and add the result to the summary table.
# 
# ```{r movie_user_genre_effect_model3_3, echo=FALSE}

predicted_ratings <- testset %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs , by='userId' ) %>%
  left_join(genre_avgs, by='genres' ) %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%                          # (3)
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings, testset$rating)
model_3_rmse

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Genre Effects Model",  
                                 RMSE = model_3_rmse))
rmse_results %>% knitr::kable() #'simple')

rm(predicted_ratings) # cleanup, is big table!

# ``` ---------------------------------------------------------------------------------------------\
# 
### Model 4: Time effect - multi-variate model ----
# 
# Next we add a time bias term $b_t$ to possibly further improve our model. 
# This term considers that at similar times ratings are similar, as per spirit of that time, as fad.
# The updated model is:
#   
#   $$ Y_{i,u,g,t} = \mu + b_i + b_u + b_g + b_t => b_t = Y_{i,u,g,t} - \mu - b_i - b_u - b_g  \ \  (4) $$
#   
#   ```{r movie_user_genre_time_effect_model4_1, echo=TRUE}

temp_avgs <- trainset %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs , by='userId' ) %>%
  left_join(genre_avgs, by='genres' ) %>%
  group_by(date) %>% 
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))    # (4)
temp_avgs[is.na(temp_avgs)] <- 0 # remove NA

# ```
# 
# We can see that these estimates vary slightly:
#   
# ```{r movie_user_genre_time_effect_model4_2, echo=TRUE}

temp_avgs %>% qplot(b_t, geom ="histogram", 
                    bins = 30, data = ., color = I("black"),
                    ylab = "Number of date/times", main = "Histogram of date/time compared with the computed b_t")

# ```
# 
# That histogram above does not show any improvement a time based term could offer, 
# it seems it is showing just noise but no skewness or concentration of data. 
# But we apply it to the model and check.
# 
# We have to prepare the testset set too with the "date" to be able to join,
# then check how much it improves the model, we test it with the testset-set,
# compare it by RSME with reality and add the result to the summary table.
# 
# ```{r movie_user_genre_time_effect_model4_3, echo=TRUE}

testset <- testset %>%
  mutate(date = strftime(as.Date(as.POSIXct(timestamp,origin='1970-01-01')),"%Y-%m"))

predicted_ratings <- testset %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs , by='userId' ) %>%
  left_join(genre_avgs, by='genres' ) %>%
  left_join(temp_avgs , by='date'   ) %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
  pull(pred)

model_4_rmse <- RMSE(predicted_ratings, testset$rating)
model_4_rmse

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Genres + Time Effects Model",  
                                 RMSE = model_4_rmse))
rmse_results %>% knitr::kable() #'simple')

rm(predicted_ratings) # cleanup, is big table!

# ``` ---------------------------------------------------------------------------------------------\
#
# Indeed it does not improve the model substantially, only at 4th place behind the decimal point. 
# We will not use this parameter further.
# 
## Regularized approach ----
# 
# This section is following the book, chapter for **regularization**
#   
#   https://rafalab.github.io/dsbook/large-datasets.html#regularization
# 
# Regularization allows us to penalize large estimates from small samples. 
# The idea is to add a penalty for large values of movie/user/genre (b_i/b_u/b_g) to the sum of squares that we minimize,
# means we penalize big distances from mean by an influencing parameter.
# 
# A more precise estimate of $b_u$, $b_i$ and $b_g$ is evaluated by solving a least squares problem.
# 
# **Lambda** is this tuning parameter for the penalty and by cross validation we get the optimal(=minimal) value.
# 
# We try the following in 3 approaches:
#   
#   * starting just with $b_i$ (movie effect)
# 
# * then add $b_u$ (user effect)
# 
# * and finally add $b_g$ (genres effect)
# 
### Model 5: Regularized movie effect ----
# 
# We use the movie effect $b_i$ in our model.
# We do a loop with lambda and check where the result is minimal.
# 
# $$ \frac{1}{N} \sum_{u,i}(Y_{u,i} - \mu - b_i)^2 + \lambda (\sum_{i} b_i^2) $$
#   
# ```{r regularized_effects5_1, include=FALSE}

mu <- mean(trainset$rating)
just_the_sum <- trainset %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

# determine best lambda (penalty factor) from a sequence - as per chapter 33.9.3
lambdas <- seq(from=0, to=10, by=0.25)

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- testset %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, testset$rating))
})

# ```
# 
# We can see the typical curve and where it has a minimum - quick plot of RMSE vs lambdas
# 
# ```{r regularized_effects5_2, echo=TRUE}

qplot(lambdas, rmses) + ggtitle("RMSE <-> lambda") + 
  labs(subtitle="minimize the penalty factor lambda for b_i") 

# ```
# 
# and the minimum /optimal lambda is at:
#   
# ```{r regularized_effects5_3, echo=TRUE}

lambda <- lambdas[which.min(rmses)]
lambda # 2.5

model_5_rmse <- min(rmses)
model_5_rmse 

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie Effects Model",  
                                 RMSE = model_5_rmse))
rmse_results %>% knitr::kable() #'simple')

# ``` ---------------------------------------------------------------------------------------------\
# 
### Model 6: Regularized movie and user effect ----
# 
# We use additionally to the movie $b_i$ the user $b_u$ in our model.
# We do as before a loop with lambda and check where the result is minimal.
# 
# $$ \frac{1}{N} \sum_{u,i}(Y_{u,i} - \mu - b_i - b_u)^2 + \lambda (\sum_{i} b_i^2 + \sum_u b_u^2) $$
#   
#   This is taking time!!!
#   
# ```{r regularized_effects6_1, include=FALSE}

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(trainset$rating)

  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_i[is.na(b_i)] <- 0 # remove NA
  
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_u[is.na(b_u)] <- 0 # remove NA
  
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
})
# ```
# 
# We can see the typical curve and where it has a minimum - quick plot of RMSE vs lambdas
# 
# ```{r regularized_effects6_2, echo=TRUE}

qplot(lambdas, rmses) + ggtitle("RMSE <-> lambda") + 
  labs(subtitle="minimize the penalty factor lambda for b_i + b_u") 

# ```
# 
# and the minimum /optimal lambda is at:
#   
#   ```{r regularized_effects6_3, echo=TRUE}

lambda <- lambdas[which.min(rmses)]
lambda # 5.25

model_6_rmse <- min(rmses)
model_6_rmse

# !!!! The penalized estimates provide an improvement over the least squares estimates !!!

# add to result summary 
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effects Model",  
                                 RMSE = model_6_rmse))
rmse_results %>% knitr::kable() #'simple')

# ``` ---------------------------------------------------------------------------------------------\
# 
### Model 7: Regularized movie, user and genres effect ----
# 
# This is finally the "full blown up model" - we use additionally the genres $b_g$.
# 
# $$ \frac{1}{N} \sum_{u,i,g}(Y_{u,i,g} - \mu - b_i - b_u  - b_g)^2 + \lambda (\sum_{i} b_i^2 + \sum_u b_u^2 + \sum_g b_g^2) $$
#   
#   WARNING: This needs time!!!
#   
# ```{r regularized_effects7_1_LOOONG, include=FALSE}

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(trainset$rating)
  
  b_i <- trainset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_i[is.na(b_i)] <- 0 # remove NA
  
  b_u <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_u[is.na(b_u)] <- 0 # remove NA
  
  b_g <- trainset %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  b_g[is.na(b_g)] <- 0 # remove NA
  
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId" ) %>%
    left_join(b_g, by = "genres" ) %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, testset$rating))
})

# ```
# 
# We can see the typical curve and where it has a minimum - quick plot of RMSE vs lambdas
# 
# ```{r regularized_effects7_2, echo=TRUE}

qplot(lambdas, rmses) + ggtitle("RMSE <-> lambda") + 
  labs(subtitle="minimize the penalty factor for b_i + b_u + b_g") 

# ```
# 
# and the minimum /optimal lambda is at:
#   
# ```{r regularized_effects7_3, echo=TRUE}

lambda <- lambdas[which.min(rmses)]
lambda # 5  # slightly higher

model_7_rmse <- min(rmses)
model_7_rmse # very slighty lower

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Genres Effects Model",  
                                 RMSE = model_7_rmse))
rmse_results %>% knitr::kable() #'simple')

# ``` ---------------------------------------------------------------------------------------------\
# 
### Final lambda ----
# 
# This value will be used in the next section for the final model.
# 
# ```{r regularized_effects7_4, echo=TRUE}

final_lambda = lambda
final_lambda

# ``` ---------------------------------------------------------------------------------------------\
# 
# Results ----
#
## Execute final model 7 now with the original data
#   
# Here is the code explicitely shown to demonstrate how the algorithm works:
#
# * in 4 steps.
#
# * choose minimized **final_lambda** from best model 7 
#
# * train with the **edx** dataset (step 1-3)
#
# * test  with the **validation** dataset (step 4)
#
# ```{r results1, echo=TRUE}

# 1. step: compute regularize movie bias term
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating             - mu)/(n() + final_lambda))
b_i[is.na(b_i)] <- 0 # remove NA

# 2. step: compute regularize user bias term
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i       - mu)/(n() + final_lambda))
b_u[is.na(b_u)] <- 0 # remove NA

# 3. step: compute regularize genre bias term
b_g <- edx %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n() + final_lambda))
b_g[is.na(b_g)] <- 0 # remove NA

# 4. step: compute predictions on **validation** based on the terms above = TEST
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId" ) %>%
  left_join(b_g, by = "genres" ) %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# ```
# 
## RMSE of best model = model 7:
# 
# ```{r results2, echo=TRUE}

BESTRMSE = RMSE(predicted_ratings, validation$rating)
BESTRMSE

# Generate output file with predicted ratings to fulfill the requirement:  
#  "a script in R format that generates your predicted movie ratings and RMSE score"  
# is done in the R file, not in the Rmd file.
if(TRUE) {
  ratef = "predicted_movie_ratings.csv" 
  rmsef = "RMSE_score.csv"
  validation <- validation %>% mutate(predicted_rating         = predicted_ratings,
                                      predicted_rating_rounded = round(predicted_ratings/0.5)*0.5)
  write.csv(validation,ratef, na = "")
  write.table(BESTRMSE,rmsef, na = "", row.names=FALSE,col.names=FALSE)

  print(paste("RESULTS:", ratef," and ", rmsef , " were written into ",getwd())) 
}
