- main.py - a driver program that can facilitate a complete Twitter pull, sentiment classification, and analysis
	- NOTE: this program will not be able to lookup usernames or pull tweets unless the .env file has valid authentication credentials for the Twitter API

- regressions.py - all regressions used for this project. **This is the most important file for understanding my work**





The following files all support "main" and are best accessed through "main" rather than directly. They are used as libraries:
- sentiments.py 			- for sentiment analysis
- twitterscraper_tweepy.py		- for pulling tweets
- twitter_code_from_twitter.py 		- for getting twitter ids from a list of usernames
-.env					- an enviornment file. Twitter API authentication keys need to be entered here to pull tweets

"output" has the relevant files created from this project
"data" contains files used in main and regressions




Unless otherwise noted, all code is original and written by me, Brian Piotrowski. 
Professor Hoffmann's team and the University of Zurich have my permission to use and repurpose any and all of this code and the resulting datasets. 