#############################################
# Project : Anime Hybrid Recommendation System
#############################################

# Context
# This data set contains information on user preference data from 73,516 users on 12,294 anime.
# Each user is able to add anime to their completed list and give it a rating and
# this data set is a compilation of those ratings.
#
# Content
# Anime.csv
#
# anime_id - myanimelist.net's unique id identifying an anime.
# name - full name of anime.
# genre - comma separated list of genres for this anime.
# type - movie, TV, OVA, etc.
# episodes - how many episodes in this show. (1 if movie).
# rating - average rating out of 10 for this anime.
# members - number of community members that are in this anime's
# "group".
# Rating.csv
#
# user_id - non identifiable randomly generated user id.
# anime_id - the anime that this user has rated.
# rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).
# Acknowledgements
# Thanks to myanimelist.net API for providing anime data and user ratings.

#############################################
# Mission 1: Data Preparing
#############################################

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
anime = pd.read_csv('Data/anime.csv')
rating = pd.read_csv('Data/rating.csv')

anime.head()
rating.head()

rating.describe().T
anime.describe().T

# We have nagetive ratings,so we should fix to score 0-10.
# And,We have two rating column,we have to change one of these.

rating = rating[rating["rating"] > 0]

rating = rating.rename({'rating': 'userRating'}, axis='columns')

rating.head()

# We want to work on 20000 user and we need anime names,and genres. So,We merge dataframes.

df_ = pd.merge(anime, rating, on=['anime_id', 'anime_id'])
df = df_.copy()
df = df[df.user_id <= 20000]
df.head()

df["userRating"].describe().T

# We are creating a pivot table for the dataframe
# with the user_ids in the index, the anime names in the columns and the ratings as values.

user_recom_df = df.pivot_table(index=["user_id"], columns=["name"], values="userRating")
user_recom_df.head()

# Choose a random user id.
random_user = int(pd.Series(user_recom_df.index).sample(1, random_state=23).values)
# 29244

# We create a new dataframe named
# random_user_df, which consists of observation units of the selected user.
random_user_df = user_recom_df[user_recom_df.index == random_user]
random_user_df.shape
random_user_df.head()

# We assign the animes that the selected user has voted to a list called animes watched.
animes_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
animes_watched

# Select the lines of the animes that the selected user watched
# from user anime_df and create a new dataframe named animes_watched_df.
animes_watched_df = user_recom_df[animes_watched]
animes_watched_df.head()
animes_watched_df.shape

# Create a new dataframe named user_anime_count, which
# contains the information about how many animes each user has watched. And we create a new df.
user_anime_count = animes_watched_df.T.notnull().sum()
user_anime_count = user_anime_count.reset_index()
user_anime_count.columns = ["user_id", "anime_count"]

user_anime_count[user_anime_count["anime_count"] == 33].count()

# We consider those who watch 60 percent or more of the animes voted by the selected user as similar users.
# We create a list called users_same_animes from the ids of these users.
perc = len(animes_watched) * 60 / 100
users_same_animes = user_anime_count[user_anime_count["anime_count"] > perc]["user_id"]
len(users_same_animes)

# 227
# Filter the animes_watched_df dataframe
# to find the ids of the users that are similar to the selected user in the user_same_animes list.
final_df = pd.concat([animes_watched_df[animes_watched_df.index.isin(users_same_animes)],
                      random_user_df[animes_watched]])
final_df.head()

# final_df.to_csv("anime.csv")

# Create a new corr_df dataframe where users' correlations with each other will be found.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

corr_df["user_id_1"].value_counts()

# Create a new dataframe named top_users
# by filtering out the users with high correlation (over 0.50) with the selected user.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.50)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "user_id"}, inplace=True)

# Merge the top_users dataframe with the rating dataset
top_users_ratings = top_users.merge(rating[["user_id", "anime_id", "userRating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["user_id"] != random_user]

top_users_ratings.head()

#############################################
# Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies
#############################################

# Create a new variable named weighted_rating, which is the product of each user's corr and rating.
top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["userRating"]
top_users_ratings.head()

# Create a new dataframe named recommendation_df containing
# the anime id and the average value of the weighted ratings of all users for each movie.
recommendation_df = top_users_ratings.groupby("anime_id").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Select animes with a weighted rating greater than 5 in recommendation_df
# and rank them by weighted rating. Save the first 5 observations a animes_to_be_recommend.
recommendation_df[recommendation_df["weighted_rating"] > 5]
animes_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 5].sort_values("weighted_rating", ascending=False)


# Bring the names of the 5 recommended animes.
animes_to_be_recommend.merge(anime[["anime_id", "name"]])[:5]







