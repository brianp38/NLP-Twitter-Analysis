# -*- coding: utf-8 -*-
"""

@author: Brian Piotrowski
"""

import os
import pandas  as pd
import dotenv
from dotenv import load_dotenv, find_dotenv
import tweepy
 
import winsound
from datetime import datetime,timedelta


os.chdir("C:\\Users\\Brian Piotrowski\\Desktop\\Working\\thesis\\")
load_dotenv()

BEARER_TOKEN = os.environ.get("BEARER_KEY")
folder = "undefined"

#after how many users the program will automatically save the progress (useful in case of program crash)
saftey_saves = 5

#max tweets per user to be pulled
MAX_TWEETS = 5

START_TIME = "2022-01-01T00:00:00Z"
END_TIME = "2023-01-01T00:00:00Z"
#opening API client
client = tweepy.Client(BEARER_TOKEN,wait_on_rate_limit=True)

#final df to be built
tweets = pd.DataFrame(columns=['author_id', 'created_at',
                               'edit_history_tweet_ids',
                               'id', 'lang','public_metrics', 'text']) 

def clear_tweets():
    """
    clears the global tweets df so new pulls don't add on to old ones

    Returns
    -------
    None.

    """
    global tweets
    tweets = pd.DataFrame(columns=['author_id', 'created_at',
                                   'edit_history_tweet_ids',
                                   'id', 'lang','public_metrics', 'text'])

def create_set_folder(name):
    """
    creates and sets output folder
    name = name to be used for output folder
    """
    global folder
    folder = name
    try:
        os.mkdir(f"data\\{name}")
        os.mkdir(f"data\\{name}\\running")
    except BaseException as e:
        print(e)
    print(f"sucessfully set folder to {folder}")

#showing which folder is being used
def print_folder():
    print(folder)
    

def set_max_tweets(MAX):
    global MAX_TWEETS
    MAX_TWEETS = MAX
    print(f"sucessfully set MAX_TWEETS to {MAX_TWEETS}")


def set_timeframe(start,end):
    """
    Sets the timeframe over which tweets will be pulled
    Parameters
    ----------
    start : str entered in the format: "YYYY-MM-DD".
    end : str entered in the format: "YYYY-MM-DD".

    Returns
    -------
    None.

    """
    global START_TIME
    global END_TIME
    
    START_TIME = f"{start}T00:00:00Z"
    END_TIME = f"{end}T00:00:00Z"

#takes a user ID and returns the tweets within 2022
def get_tweets_user(user):
    z = client.get_users_tweets(user,tweet_fields = ["created_at","lang","public_metrics","author_id"],
                                 max_results=MAX_TWEETS,
                                 exclude= ["replies","retweets"],
                                 start_time=START_TIME,
                                 end_time=END_TIME)
    df = pd.DataFrame(z.data)
    return df



def get_tweets_list(handles,lookup):
    """
    takes a list of IDs and systematically pulls tweets for each of them, adding to "tweets" df
    will pause after each 15 pulls and will wait 15min for the api request limit to expire

    Parameters
    ----------
    handles : list
        Twitter ids to pull.
    lookup : pd.DataFrame
        lookup file matching usernames to ids.

    Returns
    -------
    None.

    """
    global tweets
    counter = 0
    length = len(handles)
    for i in handles:
        try:
            x=get_tweets_user(i)
            print("------------------------------------------------------------------")
            print("retrieved {} tweets from:\n\n {}\n\n ID: {}".format(str(len(x)),str(lookup[lookup["id"]==i]["name"]),str(i)))
            counter += 1
            print(f"{counter}/{length} completed")
            print("Next batch will run at: {}".format(str((datetime.now()+timedelta(minutes=15)).time())[:5]))
            tweets = pd.concat([tweets,x])
        except BaseException as e:
            tweets.to_csv(f"data\\{folder}\\tweets2_{i}.csv")
            message = "failed on:\n {}\n\n ID: {}\n reason: {}".format(str(lookup[lookup["id"]==i]["name"]),str(i),e)
            print(message)
            
            #write error report to file
            f = open(f"data\\{folder}\\error_report_{i}.csv","w")
            f.write(message)
            f.close()
            
            #make warning sound
            for _ in range(2):
                winsound.Beep(2400, 500)
                
        #save running csv at a regular interval     
        if counter % saftey_saves == 0:
            tweets.to_csv(f"data\\{folder}\\running\\tweets2_RUNNING{i}.csv")
        
    
def main():
    #loading users of interest (includes both IDs and usernames)
    banks_lookup = pd.read_csv("Data\\banks_lookup.csv",)
    finlist_lookup = pd.read_csv("Data\\finlist_lookup.csv")
    news_lookup = pd.read_csv("Data\\news_lookup.csv")
    
    #lists of only IDs
    bank_ids = list(banks_lookup["id"])
    finlist_ids = list(finlist_lookup["id"])[4:]
    news_ids = list(news_lookup["id"])
    
    create_set_folder("news_pull")

if __name__ == "__main__":
    main()

def execute(ids_lookup):
    """
    takes a lookup df of IDs and pulls tweets for them as per parameters above

    Parameters
    ----------
    ids_lookup : pd.DataFrame()
        as outputted by pull_ids()

    Returns
    -------
    tweets : df
        All tweets pulled for the period

    """
    global tweets
    #how long will it take to run the code?
    ids = list(ids_lookup["id"])
    batches = round(len(ids)/15-.5)
    print("Code will finish at: {}".format(str((datetime.now()+timedelta(minutes=15*batches)).time())[:5]))
    print("Up to {} tweets will be pulled (up to {} per user)".format(MAX_TWEETS*len(ids),MAX_TWEETS))
    print("Tweets will be pulled for the timeframe: {} to {}".format(START_TIME[:10],END_TIME[:10]))
    input("Press enter to execute code.")
    
  
      
    get_tweets_list(ids,ids_lookup)
    
    tweets.index = range(len(tweets))
    
    
    for _ in range(3):
        winsound.Beep(1600, 200)
    
    #cleaning the resulting dataset
    
    #putting back usernames in addition to user IDs
    ids_lookup["id"] = ids_lookup["id"].astype("uint64")
    tweets = tweets.merge(ids_lookup[["id","username"]], left_on = "author_id", 
                          right_on = "id", how="left")
    tweets = tweets.drop("id_y",axis=1)
    
    #extracting the "impression count" from the dictionary of public metrics for each tweet
    tweets["imp_count"] = list(map(lambda x: x["impression_count"],tweets["public_metrics"]))
    tweets["like_count"] = list(map(lambda x: x["like_count"],tweets["public_metrics"]))
    tweets["retweet_count"] = list(map(lambda x: x["retweet_count"],tweets["public_metrics"]))
    
    
    tweets.to_csv(f"data\\{folder}\\tweets.csv")
    
    return tweets
