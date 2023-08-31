# -*- coding: utf-8 -*-
"""

@author: Brian Piotrowski
"""

import pandas as pd
import numpy as np
import os 
#os.chdir("C:\\Users\\Brian Piotrowski\\Desktop\\Working\\thesis_submit\\")
from twitter_code_from_twitter import format_as_df, pull_ids
from twitterscraper_tweepy import create_set_folder, set_timeframe, set_max_tweets,execute,clear_tweets
from sentiments import sentiments_df, sentiments
from regressions2 import load_ts,date_tweets,merge_non_traded_days,create_daily_df,avg_sent,adf,plurality_dum,pos_neg_dum,ols,print_ols,plot_tweet_dist,table_to_latex,var_sent
import matplotlib.pyplot as plt

#takes a list of twitter usernames and returns a df of twitter ids
def get_ids(usernames):
     return format_as_df(pull_ids(usernames))

#setting parameters for pulling the tweets  
def set_pull_parameters():  
    global START_DATE
    global END_DATE
    global TWEETS_PER_USER
    
    #timeframe for eligible tweets to be pulled
    set_timeframe(START_DATE, END_DATE)

    #max tweets to be pulled per user
    set_max_tweets(TWEETS_PER_USER)
    clear_tweets()

#runs the entire pull and sentiment analysis on a list of usernames
def get_tweets_sents(users):
    #set paramters
    #set_pull_parameters() 
    
    #get ids
    ids = get_ids(users)
    
    #pulling tweets for all ids
    tweets = execute(ids)
    
    #assigning sentiments to tweets
    tweets_sent = sentiments_df(tweets, "text")
    return tweets_sent

def prep_joint_df(tweets,labels):
    global START_DATE
    tweets = date_tweets(tweets)
    tweets = merge_non_traded_days(tweets, broad_dollar)
    df = create_daily_df(START_DATE, 365)
    df = avg_sent(df, tweets, labels)
    
    return df
    df = create_daily_df("2022-07-01", 365)
    
    return df


#set parameters here 
START_DATE ="2022-07-01"
END_DATE = "2023-07-01"
TWEETS_PER_USER = 45


set_pull_parameters()

#lists of users
banks_lookup = pd.read_csv("Data\\banks_lookup.csv")
finlist_lookup = pd.read_csv("Data\\finlist_lookup.csv")
news_lookup = pd.read_csv("Data\\news_lookup.csv")

tweet_est = (len(banks_lookup)+len(finlist_lookup)+len(news_lookup))*TWEETS_PER_USER
print("will end up pulling up to {} tweets in total".format(str(tweet_est)))

#get tweets from central banks
#subdirectory where results will be stored
create_set_folder("2023_pull_bank")
banks = get_tweets_sents(banks_lookup["username"])
banks["type"] = "bank"

clear_tweets()

#get tweets from finlist
#subdirectory where results will be stored
create_set_folder("2023_pull_fin")
finlist = get_tweets_sents(finlist_lookup["username"])
finlist["type"] = "fin"

clear_tweets()

#get tweets from news
#subdirectory where results will be stored
set_timeframe("2022-07-01", "2023-07-28")
set_max_tweets(100)
create_set_folder("2023_pull_news")
news = get_tweets_sents(news_lookup["username"])
news["type"] = "news"


#loading dependant variables
broad_dollar = load_ts("data/DTWEXBGS_23.csv","dollar")
sp500 = load_ts("data//SP500_23.csv","sp500")
#df.dropna(inplace=True)


#merging the tweets datasets into one
finlist.drop("id_x",axis=1,inplace=True)
banks.drop("id_x",axis=1,inplace=True)
finlist.drop("username_y",axis=1,inplace=True)
finlist.rename({"username_x":"username"},inplace=True)


tweets = pd.concat([banks,finlist])
labels = ["bank","fin"]
tweets = date_tweets(tweets)
tweets.index = range(len(tweets))
tweets = merge_non_traded_days(tweets, broad_dollar)


df = df = create_daily_df("2022-07-01", 365)



#merge broad dollar and sp500 into df
df = df.merge(broad_dollar,how="left", on = "Date")
df = df.merge(sp500,how="left", on = "Date")

#delete holidays and weekends (tweets sent on these days have already been moved above)
df.drop(df[np.isnan(df["dollar"])].index, inplace=True)
df.index = range(len(df))

#Creating first difference of broad dollar index and sp500
df["dollar_diff"] = df["dollar"].diff()
df["sp500_diff"] = df["sp500"].diff()

df = avg_sent(df, tweets, labels)

df1 = pos_neg_dum(df, .2, labels)
df1 = plurality_dum(df, labels)
df.to_csv("FINAL_dataset_23_bank_finlist.csv")

plot_tweet_dist(tweets, df, "tweets_dist_23","2023")



df1 = var_sent(df,tweets,labels)



spx = df.sp500_diff[1:]
dollar = df.dollar_diff[1:]

#reindex y and z (to fit odd indexing from X)
spx.index = range(1,len(spx)+1)
dollar.index = range(1,len(dollar)+1)

#testing for stationarity within 1st diff of broad dollar index
#using Augmented Dickey-Fuller test for unit root
adf(dollar)
adf(spx)
#conclusion: null hypothesis of a unit root is rejected => the 1st diff of the broad dollar index is stationary for this time period (no autocorrelation), and the time series component has been removed

df = plurality_dum(df,labels)

#these appear labeled with a 2
df = pos_neg_dum(df,.2,labels)

#regression with emotion dummies: pos, neg, pos2, neg2
m1 = ols(df,dollar,['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
'central_bank_neg_dummy2'])
print_ols(m1,"m1")

n1 = ols(df,spx,['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
'central_bank_neg_dummy2'])
print_ols(m1,"m1")






