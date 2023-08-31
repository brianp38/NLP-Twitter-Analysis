# -*- coding: utf-8 -*-
"""

@author: Brian Piotrowski
"""
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import winsound


#english only
#roberta = "cardiffnlp/twitter-roberta-base-sentiment"

#multilingual
roberta = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

#sentiment analysis with roberta
#using pre-trained twitter model from Barbieri et al. (2022)
def sentiments_df(df,col):
  """
    sentiment analysis with roBERTa
    using a pre-trained twitter model from HuggingFace
    
    was fine-tuned on the following languages: Ar, En, Fr, De, Hi, It, Sp, Pt
    (can also handle other languages, though less reliably)

    Parameters
    ----------
    df : DateFrame
        A dataframe with a column containing text to be analyzed.
    col : string
        The name of the column containing the text to be evaluated.

    Returns
    -------
    df : DataFrame
        The input dataframe with the three sentiment columns (positive, neutral, negative).

    """
  df[col].fillna('', inplace=True)
  

  #labels = ['Negative', 'Neutral', 'Positive']


  #example = df["Content"][2]


  df["negative"] = ""
  df["neutral"] = ""
  df["positive"] = ""
  

  length = len(df)
  
  for i in range(0,length):
      try:
          
          encoded = tokenizer(df[col][i], return_tensors='pt')
          
          # sentiment analysis
          # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
          output = model(**encoded)
          
          scores = output[0][0].detach().numpy()
          scores = softmax(scores)
          
          df["negative"][i] = scores[0].item()
          df["neutral"][i] = scores[1].item()
          df["positive"][i] = scores[2].item()
          
          if i % 50 == 0:
              print("{}% completed".format(round(i/length*100)))
      except BaseException as e:
          
          message = "failed on {} for reason: {}\ntext: {}".format(str(i),e,df[col][i])
          print(message)
          
          #make warning sound
          for _ in range(2):
              winsound.Beep(2400, 500)

            

  df["negative"] = df["negative"].astype("float")
  df["neutral"] = df["neutral"].astype("float")
  df["positive"] = df["positive"].astype("float")

  for i in range(0,len(df)):
    if df[col][i] == "":
      df["negative"][i] = pd.NA
      df["neutral"][i] = pd.NA
      df["positive"][i] = pd.NA


  return df

def sentiments(sent):
    """
    sentiment analysis with roBERTa
    using a pre-trained twitter model from HuggingFace
    
    was fine-tuned on the following languages: Ar, En, Fr, De, Hi, It, Sp, Pt
    (can also handle other languages, though less reliably)

    Parameters
    ----------
    sent : string
        a sentence to be analyzed.

    Returns
    -------
    dict
        a dictionary of the the resulting sentiments.
        keys = negative, neutral, positive

    """
    encoded = tokenizer(sent, return_tensors='pt')
    
    # sentiment analysis
    
    output = model(**encoded)   
    
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    #returns a dict of sentiment scores
    return {"negative":scores[0].item(),"neutral":scores[1].item(),"positive":scores[2].item()}


def main():
    df_main = pd.read_csv("data//2_main_df_no_sentiments2.csv", index_col =0)
    df_sentiment = sentiments(df_main,"Content")

    df = df_sentiment[["Date","Date_time","sp500","dollar","User","cbank","Content","positive","neutral","negative"]]

    df.to_csv("data//3_sentiments.csv")
    
    
if __name__ == "__main__":
    main()

   


