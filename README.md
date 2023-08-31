# NLP-Twitter-Analysis
Brian Piotrowski's master's thesis at the University of Zurich, 2023. 

"Global market sentiment and exchange rates: evidence from Twitter"
Advised by Prof. Dr. Mathias Hoffmann & Nicola Benigni

Abstract:
This paper investigates the extent to which market sentiment expressed on Twitter correlates to
exchange rate and stock market movements. The main variable of interest is the broad-dollar index–a
trade-weighted index of the US dollar against a basket of other major currencies. Global central bank
sentiment is found to be a meaningful indicator of dollar depreciations as measured by this index.
The paper goes on to compare the effects of the posts of individual central banks on domestic and
foreign stock indexes. The paper attempts to use idiosyncratic central bank sentiment as a proxy for
the stochastic discount factor. This second goal proves to be inconclusive due to data availability
constraints.



Description of Files:

- main.py - a driver program that can facilitate a complete Twitter pull, sentiment classification, and analysis
	- NOTE: this program will not be able to lookup usernames or pull tweets unless the .env file has valid authentication credentials for the Twitter API

- regressions.py - all regressions used for this project. **This is the most important file for understanding my work**

The following files all support "main" and are best accessed through "main" rather than directly. They are used as libraries:
- sentiments.py 		           	  => for sentiment analysis
- twitterscraper_tweepy.py	    	=> for pulling tweets
- twitter_code_from_twitter.py 		=> for getting twitter ids from a list of usernames
-.env				                    	=> an enviornment file. Twitter API authentication keys need to be entered here to pull tweets

"output" has the relevant files created from this project
"data" contains files used in main and regressions




Unless otherwise noted, all code is original and written by me, Brian Piotrowski. 
Professor Hoffmann's team and the University of Zurich have my permission to use and repurpose any and all of this code and the resulting datasets. 
