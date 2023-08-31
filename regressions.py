# -*- coding: utf-8 -*-
"""
@author: Brian Piotrowski
"""

#scikit-learn
import sklearn as sk
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.correlation import plot_corr
import copy

#matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#datetime
import datetime
from datetime import timedelta

#yahoo finance
import yfinance as yf

#normally loaded packages
import pandas as pd
import numpy as np
import os

#defining the directory
#os.chdir("C:\\Users\\Brian Piotrowski\\Desktop\\Working\\thesis_submit\\")

def tree(leaves,X,y):
    """
    Creates a regression tree based on parameters.

    Parameters
    ----------
    leaves : int
        Maximum number of leaves (nodes) for the tree.
    X : pd.DataFrame()
        Independant variables.
    y : series
        Dependant variable. 

    Returns
    -------
    tree : Tree
        

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeRegressor(max_leaf_nodes=leaves)
    tree.fit(X_train, y_train)
    
    predict = tree.predict(X_test)
    print(f"Regreesion tree with max of {leaves} leaves:")
    print("R^2 (in sample): " + str(tree.score(X_test,y_test)))
    #print("R^2 (out-of-sample): " + str(tree.oob_score_))

    print("MSE (in-sample): "+str(mean_squared_error(y_train,tree.predict(X_train))))
    print("MSE (out-of-sample): "+str(mean_squared_error(y_test,predict)),"\n")
    
    kf6 = KFold(n_splits=6)
    print("\nCross validation score: ", -cross_val_score(tree, X_train, y_train, cv=kf6, scoring='neg_mean_squared_error').mean())
    
    plt.plot(y_test, predict, "red")
    #plt.plot(range(0,1,.1).values,range(0,1,.1).values, "blue")
    #plt.plot(X_test,predict, "red")

    plt.show()

    return tree

def forest(trees,X,y):
    
    """
    Creates a forest of regression trees based on parameters.

    Parameters
    ----------
    trees : int
        Number of trees (estimators) for the forest.
    X : pd.DataFrame()
        Independant variables.
    y : series
        Dependant variable. 

    Returns
    -------
    forest : Forest
        

    """
    
    #simple forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    forest = RandomForestRegressor(n_estimators=trees, oob_score=True, max_features="sqrt", random_state=41)
    forest.fit(X_train,y_train)

    predict = forest.predict(X_test)
    print("Random forest with ",trees," trees:")
    print("R^2 (in sample): " + str(forest.score(X_test,y_test)))
    print("R^2 (out-of-sample): " + str(forest.oob_score_))

    print("MSE (in-sample): "+str(mean_squared_error(y_train,forest.predict(X_train))))
    print("MSE (out-of-sample): "+str(mean_squared_error(y_test,predict)),"\n")
    
    kf6 = KFold(n_splits=6)
    print("\nCross validation score: ", -cross_val_score(forest, X_train, y_train, cv=kf6, scoring='neg_mean_squared_error').mean())
    
    
    compare = pd.DataFrame(y_test, columns=["actual"])
    compare["predicted"] = (pd.DataFrame(predict))
    #compare.columns = ["actual","predicted"]

    #return forest
    return compare


def merge_non_traded_days(tweets, time_series_w_na):
    """
    Change the date of tweets occuring on weekends and holidays so that they are now dated as occuring on the next business day.

    Parameters
    ----------
    tweets : df
        Tweets dataframe as generated from previous functions.
    time_series_w_na : df
        A dataframe that includes a column named "Date" which includes nan values for holidays and weekends.

    Returns
    -------
    Tweets df with a modified "Date" column.

    """
    final_date = max(time_series_w_na["Date"])
    for i in range(len(tweets)):
        date = tweets["Date"][i]
        
        while len(time_series_w_na[time_series_w_na["Date"]==date])==0:
            if date >= final_date:
                break
            date+=datetime.timedelta(days=1)
            
        tweets.loc[i,"Date"] =date
        
    return tweets

def create_daily_df(start_date,days):
    """
    sets up an empty df of dates

    Parameters
    ----------
    start_date : str
        start date entered in the format: "YYYY-MM-DD".
    days : int
        number of days from the start the analysis will run.

    Returns
    -------
    df : df
        empty dataframe of dates.

    """    
    dates = []
    start = datetime.date(year=int(start_date[:4]),month=int(start_date[6:7]),day=int(start_date[8:10]))
    #end = datetime.date(year=int(end_date[:4]),month=int(end_date[6:7]),day=int(end_date[8:10]))
    
    for i in range(days):
        dates.append((start+i*datetime.timedelta(days=1)))
    
    
    #add this list to the df (will built this df bottom-up)
    df = pd.DataFrame(columns=["Date"])
    df["Date"] = dates
    
    return df

def adf(series):
    """
    Augmented-Dickey Fuller test and outprint of results

    Parameters
    ----------
    series : series
        series of data to be tested for stationarity.

    Returns
    -------
    None.

    """
    adf = adfuller(series)
    print(f"ADF stat: {adf[0]}")
    print(f"p-value: {adf[1]}")
    for key,value in adf[4].items():
        print(f"{key}: {value}")

def load_ts(path,col_name = "x"):
    """
    loads and cleans timeseries data of a specific format

    Parameters
    ----------
    path : str
        Path to the 2 column timeseries dataset to be used cols = ["Date","data"].
    
    col_name : str, optional
        name to be given to the data colum
        
    Returns
    -------
    df : df
        Cleaned dataframe ready for use

    """

    broad_dollar = pd.read_csv(path)
    broad_dollar.columns = ["Date",col_name]
    broad_dollar["Date"] = pd.to_datetime(broad_dollar["Date"])
    broad_dollar["Date"] = [d.date() for d in broad_dollar["Date"]]
    
    #dropping missing values (holidays)
    broad_dollar.drop(broad_dollar[broad_dollar[col_name]=="."].index, inplace=True)
    #convertign to a usable data type (float)
    broad_dollar[col_name] = broad_dollar[col_name].astype(float)
    return broad_dollar

def avg_sent(df,tweets,labels):
    """
    creates columns to represent the avg sentiment among different types seperatly

    Parameters
    ----------
    df : df
        dataframe that contains a column of dates called "Date".
    tweets : df
        tweets df. Must contain a "type" column that corresponds to the labels
    labels : list
        list of the type names.

    Returns
    -------
    df : df
        the dates df with info on the average sentiment among each type per day.

    """
  
    for i in labels:
        for j in ["positive","neutral","negative"]:
            df[f"{i}_{j}"] = np.nan
        df[f"imp_{i}"] = 0
        df[f"count_{i}"] = 0

    #averaging tweets by day and by type
    for i in range(len(df)):
        date = df["Date"][i]
        
        for j in labels:
            x = tweets[tweets["type"]==j][tweets["Date"]==date]
            for k in ["positive","neutral","negative"]:
                df.loc[i, f"{j}_{k}"] = x[f"{k}"].mean()
            
            df[f"imp_{j}"][i] = x["imp_count"].sum()
            df[f"count_{j}"][i] = len(x)    
    return df

def avg_sent_country(df,tweets,labels, countries):
    """
    creates columns to represent the avg sentiment among different types seperatly

    Parameters
    ----------
    df : df
        dataframe that contains a column of dates called "Date".
    tweets : df
        tweets df. Must contain a "type" column that corresponds to the labels
    labels : list
        list of the type names.

    Returns
    -------
    df : df
        the dates df with info on the average sentiment among each type per day.

    """
  
    for i in labels:
        for j in ["positive","neutral","negative"]:
            df[f"{i}_{j}"] = np.nan

        df[f"imp_{i}"] = 0
        df[f"count_{i}"] = 0
        
        for country in countries:
            for j in ["positive","neutral","negative"]:
                df[f"{country}_{j}"] = 0
        

    #averaging tweets by day and by type
    for i in range(len(df)):
        date = df["Date"][i]
        
        for j in labels:
            x = tweets[tweets["type"]==j][tweets["Date"]==date]
            
            for country in countries:
                y = x[x["username"]==country]
                for k in ["positive","neutral","negative"]:
                    df.loc[i, f"{country}_{k}"] = y[f"{k}"].mean()
                    
                x = x[x.username != country]
            for k in ["positive","neutral","negative"]:
                df.loc[i, f"{j}_{k}"] = x[f"{k}"].mean()
            
            df[f"imp_{j}"][i] = x["imp_count"].sum()
            df[f"count_{j}"][i] = len(x)  
            
            
    return df


def var_sent(df,tweets,labels):
    """
    creates columns to represent the varience of sentiment among different types seperatly.

    Parameters
    ----------
    df : df
        dataframe that contains a column of dates called "Date".
    tweets : df
        tweets df. Must contain a "type" column that corresponds to the labels
    labels : list
        list of the type names.

    Returns
    -------
    df : df
        the dates df with info on the sentiment varience of tweets among each type per day.

    """
  
    for i in labels:
        for j in ["positive","neutral","negative"]:
            df[f"{i}_{j}_var"] = np.nan


    #averaging tweets by day and by type
    for i in range(len(df)):
        date = df["Date"][i]
        
        for j in labels:
            x = tweets[tweets["type"]==j][tweets["Date"]==date]
            for k in ["positive","neutral","negative"]:
                df.loc[i, f"{j}_{k}_var"] = x[f"{k}"].var()
               
    return df

def date_tweets(tweets):
    """
    adds a "Date" column of properly formated datetime dates.

    Parameters
    ----------
    tweets : df
        tweets df.

    Returns
    -------
    tweets : df
        tweets df with "Date" column.

    """
    
    tweets["created_at"]= pd.to_datetime(tweets["created_at"])
    tweets["Date"] = [d.date() for d in tweets["created_at"]]
    
    return tweets


def plurality_dum(df,labels):
    """
    
    Creates dummies based on the highest of the three emotion categories.

    Parameters
    ----------
    df : df
        df with columns labled f"{type}_{emotion}" where emotion is in ["positive","negative","neutral"].
    labels : list
        list of the type names.
    Returns
    -------
    df : df
        the original df with the additional dummy categories

    """
    df.index = range(len(df))
    
    for j in labels:
    
        for i in ["pos","neu","neg"]:
            df[f"{j}_{i}_dummy"] = 0
    
        for i in range(len(df)):
            if (df[f"{j}_positive"][i] > df[f"{j}_neutral"][i]) & (df[f"{j}_positive"][i] > df[f"{j}_negative"][i]):
                df[f"{j}_pos_dummy"][i] = 1
            if (df[f"{j}_neutral"][i] > df[f"{j}_negative"][i]) & (df[f"{j}_neutral"][i] > df[f"{j}_positive"][i]):
                df[f"{j}_neu_dummy"][i] = 1
            if (df[f"{j}_negative"][i] > df[f"{j}_neutral"][i]) & (df[f"{j}_negative"][i] > df[f"{j}_positive"][i]):
                df[f"{j}_neg_dummy"][i] = 1

    return df


def pos_neg_dum(df,threshold,labels):
    """
    creates dummies for pos/neg where dummy=1 iff emotion>threshold. 
    If emotion>threshold, it further checks that emotion> (opposite emotion). 
    This ensures that dummies are mutually exclusive even if theshold< 0.5.

    Parameters
    ----------
    df : df
        daily df.
    threshold : float
        threshold over which an emotion must be to get a dummy=1. [0,1]
    labels : list
        list of user categories.

    Returns
    -------
    df : list
        list of the type names.

    """
    for j in labels:        
        for i in ["pos","neg"]:
            df[f"{j}_{i}_dummy2"] = 0
        
        for i in range(len(df)):
            if (df[f"{j}_positive"][i] > threshold) & (df[f"{j}_positive"][i] > df[f"{j}_negative"][i]):
                df[f"{j}_pos_dummy2"][i] = 1
            if (df[f"{j}_negative"][i] > threshold) & (df[f"{j}_negative"][i] > df[f"{j}_positive"][i]):
                df[f"{j}_neg_dummy2"][i] = 1
    
    #count of 
    #print(f"positive count (>{threshold}): " +str(df["positive"][df["positive"]>threshold].count()))
    #print(f"negative count (>{threshold}): "+str(df["negative"][df["negative"]>threshold].count()))
    return df
    
    

def interact(df,cols):
    """
    Create all possible interaction terms for the given columns.

    Parameters
    ----------
    df : pd.DataFrame
        daily df with all needed columns..
    cols : list
        columns to be interacted.

    Returns
    -------
    X_int : pd.DataFrame
        input df with the interaction columns added.

    """
    X_int = copy.deepcopy(df)
    for i in cols: 
        X_int[f"cbank_X_{i}"] = df["cbank"]*df[f"{i}"]
    
    return X_int

def ols(df,outcome,cols,test_size=.2,return_ridge=False,return_ols=False):
    """
    runs OLS on a training sample and validates on an out-of-sample test

    Parameters
    ----------
    df : df
        daily df with all needed columns.
    outcome : string
        column to be used as the dependant variable.
    cols : list
        columns to be used as explanitory variables.

    Returns
    -------
    ols2 : ols model from statsmodels package
        resulting regression.

    """
    df.index = range(len(df))
    z = df[outcome]
    X_train, X_test, y_train, y_test = train_test_split(df[cols][1:], z[1:], test_size=test_size, random_state=42)
    
    ols = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=1.2).fit(X_train, y_train)
    lasso = Lasso(alpha=1.2).fit(X_train, y_train)
    mse_ols = mean_squared_error(y_test, ols.predict(X_test))
    print("OLS MSE (out of sample): "+str(mse_ols))
    print("Ridge MSE (out of sample): "+str(mean_squared_error(y_test, ridge.predict(X_test))))
    var = np.var(z[1:])
    print("Lasso MSE (out of sample): "+str(mean_squared_error(y_test, lasso.predict(X_test))))
    var = np.var(z[1:])
    
    
    print("ridge coef. : "+ str(ridge.intercept_ ) +" "+ str(ridge.coef_))
    print("lasso coef. : "+ str(lasso.intercept_ ) +" "+ str(lasso.coef_))
    print("varience of outcome variable: "+ str(var))
    print("Out of sample MSE represents {} of the variance".format(mse_ols/var))
    X_train = sm.add_constant(X_train)
    ols2 = sm.OLS(y_train,X_train).fit()
    print(ols2.summary())
    if return_ridge:
        return ridge
    elif return_ols:
        return ols
    else:
        return ols2

def plot_tweet_dist(tweets,df,name,year):
    """
    
    Used to show the asymetry of the tweets distribution by date. 
    Saves file to output folder
    
    Parameters
    ----------
    tweets : pd.DataFrame
        tweets dataframe.
    df : pd.DataFrame
        daily dataframe.
    name : str
        name to be used for resulting file.
    year : str
        Year to be displayed in the resulting graph.

    Returns
    -------
    None.

    """

    dates = df["Date"]
    p = []
    for day in dates:
        p.append(len(tweets[tweets["Date"]==day]))
    
    plt.figure(figsize=(10,6))
    
    plt.title(f"Distribution of Tweets ({year} dataset)")
    plt.xlabel("Date")
    plt.ylabel("Total tweets")
               
    plt.axhline(y = sum(p)/len(p), color = 'r', linestyle = '-')
    plt.plot(dates,p)
    #plt.show()
    plt.savefig(f"output\\{name}.png", dpi =300, bbox_inches="tight")

def best_alpha(df,outcome,cols,model =""):
    """
    
    Iterates through a range of ridge regression alphas and identifies the optimal alpha for out-of-sample prediction.
    Prints a graph showing the convergence of the coefficients to zero along with a vertical line at the alpha that predicts best.

    Parameters
    ----------
    df : pd.DataFrame
        daily df with all needed columns.
    outcome : str
        name of the dependant variable column within df.
    cols : list
        columns to be used as explanitory variables.
    model : str, optional
        Model name to be shown in the graph's title. The default is "".

    Returns
    -------
    best_a : int
        The alpha value that results in the lowest out-of-sample MSE.

    """
    n_alphas = 200
    alphas = np.logspace(-10, 3, n_alphas)
    
    z = df[outcome]
    X_train, X_test, y_train, y_test = train_test_split(df[cols][1:], z[1:], test_size=0.2, random_state=42)
    
    
    coefs = []
    mses = pd.DataFrame(columns=["alpha","mse"])
    b =[]
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X_train, y_train)
        coefs.append(ridge.coef_)
        
        b.append(mean_squared_error(y_test, ridge.predict(X_test)))
        #mses.append(b,ignore_index=True)
    mses["alpha"] = alphas
    mses["mse"] = b
    
    #ax = plt.gca()
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    #ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel("alpha")
    plt.ylabel("coefficients")
    best_a = mses[mses["mse"]==min(mses["mse"])]["alpha"].mean()
    plt.axvline(best_a,color = "r", label = "min. MSE")
    plt.title(f"Ridge coefficients as a function of the regularization {model}")
    plt.axis("tight")
    plt.show()
    print("Best alpha (min MSE): "+ str(best_a))
    
    fig2, ax2 = plt.subplots(figsize=(10,6))
    plt.plot(alphas,b)
    plt.axvline(best_a,color = "r", label = "min. MSE")
    
    
    return best_a

    
def table_to_latex(df, name):
    """
    outputs a table to a latex markup file

    Parameters
    ----------
    df : pd.DateFrame
        table to be outprinted.
    name : str
        name for the resulting file (without extension).

    Returns
    -------
    None.

    """
    f = open(f"latex\\{name}.txt","w+")
    f.write(df.to_latex())
    f.close()
    
def print_ols(model,name):
    """
    outputs a model to a latex markup file

    Parameters
    ----------
    model : ols model from statsmodels package
    name : str
        name to be given to the resulting file.

    Returns
    -------
    None.

    """
    f = open(f"latex\\{name}.txt","w+")
    f.write(model.summary().as_latex())
    f.close()

def load_ts_yf(path, col_name):
    """
    Loads a yahoo finance csv and prepares it for merging.

    Parameters
    ----------
    path : str
        path to locally saved csv file.
    col_name : str
        name to be given to the column of data.

    Returns
    -------
    x : pd.DataFrame
        

    """
    x = pd.read_csv(path)
    x = x[["Date","Adj Close"]]
    x.columns = ["Date",col_name]
    x["Date"] = pd.to_datetime(x["Date"])
    x["Date"] = [d.date() for d in x["Date"]]
    
    return x

def load_ts_yf_api(ticker, col_name):
    """
    Retrieves a table from the Yahoo Finace API and prepares it for merging.

    Parameters
    ----------
    ticker : str
        Stock or index ticker.
    col_name : str
        name to be given to the column of data.

    Returns
    -------
    x : pd.DataFrame
        

    """
    startDate = datetime.datetime(2022, 1, 1)
    endDate = datetime.datetime(2023, 1, 1)
    b = yf.Ticker(ticker)
    x = b.history(start=startDate,end=endDate)
    x["Date"] = x.index
    x.index.name ="index"
    
    x = x[["Date","Close"]]
    x.columns = ["Date",col_name]
    x["Date"] = pd.to_datetime(x["Date"])
    x["Date"] = [d.date() for d in x["Date"]]
    
    return x

def plot_ex(df,currency,country):
    """
    Plots sentiment ratios and exchange rates for two countries.
    The goal is to observe a shared direction between the changes in ratios and rates

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of ratios (marked as column 0) and exchange rates (marked as column "ex_rate").
    currency : str
        Currency abbreviation.
    country : str
        Country name abbreviation.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10,6))
    #fig,ax1 = plt.subplots()
    #ax2=ax1.twinx()
    df=df.dropna()
    
    x = [11,12]
    y1 = df[0]
    y2 =df["ex_rate"]
    
    plt.title(f"{currency} vs USD")
    plt.xlabel("Month")
    #plt.ylabel("Sentiment")
               
    plt.plot(x,y1,color ="r",label = f"US sent. - {country} sent.")
    plt.plot(x,y2,color ="b", label = f"{currency} / USD")
    leg = plt.legend(loc='upper right')
    plt.show()
    #plt.savefig(f"output\\{country}_ex_plot_neg.png", dpi =300, bbox_inches="tight")


def corr_matrix(df, variables, name):
    """
    
    Plotting and saving a visualization of the correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        
    variables : list
        List of column names that will be compared.
        
    name : str
        name of the resulting file to be saved

    Returns
    -------
    None.

    """
    
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    cmap = mpl.cm.Reds
    norm = mpl.colors.Normalize(vmin=-.2, vmax=1)
    
    plot_corr(df[variables].dropna().corr(),
              xnames = variables,
              ynames = variables,
              ax = ax,
              cmap=cmap
              )
    cax = fig.add_axes([.75, .3, 0.05, 0.5])

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical')
    
    fig.savefig("output\\{name}.png", dpi =300, bbox_inches="tight")
    

def main():
    """
    
    Regressions to be run on data that has already been pulled and marked with sentiments.


    """
    
    #loading dependant variables
    broad_dollar = load_ts("data/DTWEXBGS.csv","dollar")
    sp500 = load_ts("data//SP500.csv","sp500")
    stoxx_eu = load_ts_yf("data//STOXX.csv","stoxx_eu")
    axjo_au = load_ts_yf_api("^AXJO", "axjo_au")
    ftse_gb = load_ts_yf_api("^FTSE", "ftse_gb")
    gsptse_ca = load_ts_yf("data//^GSPTSE.csv", "gsptse_ca")


    #loading and cleaning tweets dataframe
    tweets = pd.read_csv("data//tweets_sent_2022.csv", index_col = 0)
    tweets.drop("Unnamed: 0",axis=1,inplace=True)
    tweets = date_tweets(tweets)
    
    #merging non-traded days into the following business day
    tweets = merge_non_traded_days(tweets, broad_dollar)
    
    #creating a df of all the dates in 2022. This is what will be built off of
    df = create_daily_df("2022-01-01", 365)
    
    #merging the average sentiment and average variance in sentiment for each day into the dates df. 
    #This is done per poster-type.
    labels = ["fin_poster","central_bank"]
    df = avg_sent(df, tweets, labels)
    df = var_sent(df, tweets, labels)
    
    #summing variance terms across positive and negative sentiment
    df["bank_var"] = df["central_bank_positive_var"] + df["central_bank_negative_var"]
    df["fin_var"] =df["fin_poster_positive_var"] + df["fin_poster_negative_var"]
    
    #"countries" is a list of the central banks of the five main countries of interest
    countries = ["ecb","federalreserve","bankofengland","RBAInfo","bankofcanada"]
    
    #df_country is a mirror of df except that it holds banks of the five countries of interest seperate
    #and assigns them their own dummy variables
    df_country = avg_sent_country(create_daily_df("2022-01-01", 365), 
                                  tweets, labels, 
                                  countries)
    
    #merge broad dollar and sp500 into df
    df = df.merge(broad_dollar,how="left", on = "Date")
    df = df.merge(sp500,how="left", on = "Date")
    
    #merge broad dollar and all stock indexes into df_country
    df_country = df_country.merge(broad_dollar,how="left", on = "Date")
    df_country = df_country.merge(sp500,how="left", on = "Date")
    df_country = df_country.merge(stoxx_eu,how="left", on = "Date")
    df_country = df_country.merge(axjo_au,how="left", on = "Date")
    df_country = df_country.merge(ftse_gb,how="left", on = "Date")
    df_country = df_country.merge(gsptse_ca,how="left", on = "Date")
    
    #delete holidays and weekends (tweets sent on these days have already been moved above)
    #the missing data (nan) in the broad dollar series is used to identify weekends and US bank holidays
    df.drop(df[np.isnan(df["dollar"])].index, inplace=True)
    df_country.drop(df_country[np.isnan(df_country["dollar"])].index, inplace=True)
    
    #Creating first difference of broad dollar index and sp500 for df
    df["dollar_diff"] = df["dollar"].diff()
    df["sp500_diff"] = df["sp500"].diff()
    
    #Creating first difference of broad dollar index and all stock indexes for df_country
    df_country["dollar_diff"] = df_country["dollar"].diff()
    df_country["sp500_diff"] = df_country["sp500"].diff()
    df_country["eu_diff"] = df_country["stoxx_eu"].diff()
    df_country["au_diff"] = df_country["axjo_au"].diff()
    df_country["gb_diff"] = df_country["ftse_gb"].diff()
    df_country["ca_diff"] = df_country["gsptse_ca"].diff()

    

    #df.to_csv("FINAL_dataset.csv")


    #testing for stationarity within 1st diff of broad dollar index
    #using Augmented Dickey-Fuller test for unit root
    adf(df["dollar_diff"][1:])
    #conclusion: null hypothesis of a unit root is rejected => the 1st diff of the broad dollar index is stationary for this time period (no autocorrelation), and the time series component has been removed
    
    #stationarity tests for the first differences of all stock indexes
    adf(df_country.dropna(subset="sp500_diff")["sp500_diff"])
    adf(df_country.dropna(subset="eu_diff")["eu_diff"])
    adf(df_country.dropna(subset="gb_diff")["gb_diff"])
    adf(df_country.dropna(subset="au_diff")["au_diff"])
    adf(df_country.dropna(subset="ca_diff")["ca_diff"])
    #conclusion: null hypothesis of a unit root is rejected for all => the 1st diff of each index is stationary for this time period (no autocorrelation), and the time series component has been removed
    
    #adding plurality dummies (identifies highest scoring sentiment category and creates dummy)
    #these will be used for the financial posters category but not for the central banks
    df = plurality_dum(df,labels)
    df_country = plurality_dum(df_country,labels)
    
    
    #interact("pos_dummy","neg_dummy")
    #result: interaction terms with the central bank dummy are not useful because the vast majority of central bank posts are classified as "neutral". Only 2% of central bank tweets are non-neutral (2 pos+ 3 neg)/299 total

    #adding threshold dummies (theshold = 0.2)
    #these all have a "2" at the end of their column name in the resulting df (ie "central_bank_pos_dummy2")
    #these will be used for the central banks since this type of dummy is more sensitive
    df = pos_neg_dum(df,.2,labels)
    df_country = pos_neg_dum(df_country,.2,labels)
    
    #the same method is used to give dummies based on thesholds to individual central banks within df_country
    #like above, these also end in "2" to indicate that they are threshold and not plurality dummies
    df_country = pos_neg_dum(df_country,.2,countries)
    
    """
    Final regressions: These are the regressions presented in the paper
    
    """
    
    #OLS regressions    
    
    #Regression 1
    #regression with emotion dummies: pos, neg, pos2, neg2
    ols1 = ols(df,"dollar_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'],return_ols=True)
    #print_ols(ols1,"analysis_set_dummies")
  
    #creating correlation matrix for Regression 1 (saved to output folder)
    corr_matrix(df,["bank_var","fin_var",'fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'], "corr")
  
    #Regression 2
    #regression with variences only
    ols2 = ols(df.dropna(subset = ["bank_var","fin_var"]),"dollar_diff",["bank_var","fin_var"],return_ols=True)
    #print_ols(ols2, "var_only")
    
    #Regression 3
    #regression with all emotion dummies and variences
    ols3 = ols(df.dropna(subset = ["bank_var","fin_var"]),"dollar_diff",["bank_var","fin_var",'fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'],return_ols=True)
    #print_ols(ols3,"var_and_avg")
    
    
    
    
    #Ridge regression versions of the three OLS regressions
    
    #Regression 1 - Ridge version
    #regression with emotion dummies: pos, neg, pos2, neg2
    ridge1 = ols(df,"dollar_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'],return_ridge=True)
    
  
    #Regression 2 - Ridge version
    #regression with variences only
    ridge2 = ols(df.dropna(subset = ["bank_var","fin_var"]),"dollar_diff",["bank_var","fin_var"],return_ridge=True)

    #Regression 3 - Ridge version
    #regression with all emotion dummies and variences
    ridge3 = ols(df.dropna(subset = ["bank_var","fin_var"]),"dollar_diff",["bank_var","fin_var",'fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'],return_ridge=True)
    
    
    """
    Ridge alpha testing
    this is testing a range of alphas for best out-of-sample predictability 
    
    run "help(best_alpha)" for more info
    
    """

    alphas1 = best_alpha(df,"dollar_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'],"(model 1)")
    
    alphas2 =best_alpha(df.dropna(subset = ["bank_var","fin_var"]),"dollar_diff",["bank_var","fin_var"],"model 2")
    
    alphas3 = best_alpha(df.dropna(subset = ["bank_var","fin_var"]),"dollar_diff",["bank_var","fin_var",'fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'],"model 3")
    

    
    
    
    """
    Out of sample testing of the OLS and Ridge models on 2023 data
    """
    
    #loading 2023 data
    df2 = pd.read_csv("output//FINAL_dataset_23_bank_finlist.csv")
    df2.rename(columns={"bank_neg_dummy2":"central_bank_neg_dummy2",
                        "bank_pos_dummy2":"central_bank_pos_dummy2",
                        "bank_neg_dummy":"central_bank_neg_dummy",
                        "bank_pos_dummy":"central_bank_pos_dummy",
                        "bank_neu_dummy":"central_bank_neu_dummy",
                        "fin_neg_dummy": "fin_poster_neg_dummy",
                        "fin_pos_dummy": "fin_poster_pos_dummy",
                        "fin_neu_dummy": "fin_poster_neu_dummy",
                        "fin_neg_dummy2":"fin_poster_neg_dummy2",
                        "fin_pos_dummy2":"fin_poster_pos_dummy2",
                        "count_bank":"count_",
                        "count_fin":"count_fin_poster",                        
                        },inplace=True)
    
    #truncating to exclude the overlapping section (all 2022 dates since the 2023 dataset is from 7/2022 to 7/2023)
    df2 =df2[125:]
    df2.index = range(len(df2))
    
    #creating variables that did not exist in the other dataset
    df2["bank_var"] = df2['bank_positive_var'] + df2['bank_negative_var']
    df2["fin_var"] = df2['fin_positive_var'] + df2['fin_negative_var']
    
    
    
    #OLS out of sample tests
    #Regression 1: out of sample test
    test_error_1 = mean_squared_error(df2["dollar_diff"][1:], ols1.predict(df2[['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2']][1:]))
    
    #Regression 2: out of sample test
    test_error_2 = mean_squared_error(
        df2.dropna(subset = ["bank_var","fin_var"])["dollar_diff"][1:],
        ols2.predict(df2.dropna(subset = ["bank_var","fin_var"])
                     [["bank_var","fin_var"]][1:]))
    
    #Regression 3: out of sample test
    test_error_3 = mean_squared_error(
        df2.dropna(subset = ["bank_var","fin_var"])["dollar_diff"][1:],
        ols3.predict(df2.dropna(subset = ["bank_var","fin_var"])
                     [["bank_var","fin_var",'fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
                     'central_bank_neg_dummy2']][1:]))
    
    
    
    #Ridge out of sample tests
    #Regression 1 - Ridge: out of sample test
    test_error_1_ridge = mean_squared_error(df2["dollar_diff"][1:], ridge1.predict(df2[['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2']][1:]))
    
    #Regression 2 - Ridge: out of sample test
    test_error_2_ridge = mean_squared_error(
        df2.dropna(subset = ["bank_var","fin_var"])["dollar_diff"][1:],
        ridge2.predict(df2.dropna(subset = ["bank_var","fin_var"])
                     [["bank_var","fin_var"]][1:]))
    
    #Regression 3 - Ridge: out of sample test
    test_error_3_ridge = mean_squared_error(
        df2.dropna(subset = ["bank_var","fin_var"])["dollar_diff"][1:],
        ridge3.predict(df2.dropna(subset = ["bank_var","fin_var"])
                     [["bank_var","fin_var",'fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
                     'central_bank_neg_dummy2']][1:]))
    
    
    
    
    """
    Predictions using machine learning  (regression trees and forests)
    """
    
    forest100 = forest(100, df[['count_fin_poster',
           'central_bank_pos_dummy', 'central_bank_neu_dummy',
           'central_bank_neg_dummy', 'fin_poster_pos_dummy',
           'fin_poster_neu_dummy', 'fin_poster_neg_dummy',
           'central_bank_pos_dummy2', 'central_bank_neg_dummy2',
           'fin_poster_pos_dummy2', 'fin_poster_neg_dummy2',
           'imp_central_bank', 'imp_fin_poster']][1:], df["dollar_diff"][1:
           ]
           )
                                                                         
                                                                         
    forest10 = forest(10, df[['count_fin_poster',
           'central_bank_pos_dummy', 'central_bank_neu_dummy',
           'central_bank_neg_dummy', 'fin_poster_pos_dummy',
           'fin_poster_neu_dummy', 'fin_poster_neg_dummy',
           'central_bank_pos_dummy2', 'central_bank_neg_dummy2',
           'fin_poster_pos_dummy2', 'fin_poster_neg_dummy2',
           ]][1:], df["dollar_diff"][1:
           ]
           )         
                                                             
    tree4 = tree(4, df[['count_fin_poster',
           'central_bank_pos_dummy', 'central_bank_neu_dummy',
           'central_bank_neg_dummy', 'fin_poster_pos_dummy',
           'fin_poster_neu_dummy', 'fin_poster_neg_dummy',
           'central_bank_pos_dummy2', 'central_bank_neg_dummy2',
           'fin_poster_pos_dummy2', 'fin_poster_neg_dummy2',
           ]][1:], df["dollar_diff"][1:
           ]
           )                                                                    
    tree6 = tree(6, df[['count_fin_poster',
           'central_bank_pos_dummy', 'central_bank_neu_dummy',
           'central_bank_neg_dummy', 'fin_poster_pos_dummy',
           'fin_poster_neu_dummy', 'fin_poster_neg_dummy',
           'central_bank_pos_dummy2', 'central_bank_neg_dummy2',
           'fin_poster_pos_dummy2', 'fin_poster_neg_dummy2',
           ]][1:], df["dollar_diff"][1:
           ]
           )
    
                                     
    #using the most sucessful ML model to predict
    test_error_tree6 = mean_squared_error(
        df2["dollar_diff"][1:],
        tree6.predict(df2[['count_fin_poster',
               'central_bank_pos_dummy', 'central_bank_neu_dummy',
               'central_bank_neg_dummy', 'fin_poster_pos_dummy',
               'fin_poster_neu_dummy', 'fin_poster_neg_dummy',
               'central_bank_pos_dummy2', 'central_bank_neg_dummy2',
               'fin_poster_pos_dummy2', 'fin_poster_neg_dummy2',
               ]][1:]))                                                                     
                                                                         
    
                                                                        
   

                                            
    #Country specific regressions
    
    #Regression 4
    #includes all the variables from Regression 1, adds the country dummies
    ols1_country = ols(df_country,"dollar_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2','ecb_pos_dummy2',
    'ecb_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2', 'bankofengland_pos_dummy2',
    'bankofengland_neg_dummy2', 'RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    
    
    #Regression 5
    #includes all the variables from Regression 1, adds only the negative emotion country dummies 
    ols1_country_neg = ols(df_country,"dollar_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2',
    'ecb_neg_dummy2', 
    'federalreserve_neg_dummy2', 
    'bankofengland_neg_dummy2', 
    'RBAInfo_neg_dummy2',
    'bankofcanada_neg_dummy2'])
    
    #Regression 6
    #includes all the variables from Regression 1, adds only the dummies for countries that were statistically significant in Regression4
    ols_aus_us_ca = ols(df_country,"dollar_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2','RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    
    
    
    
    
    
    """
    The following five regressions make up Table 7 of the final paper
    Table 7: Regressions on major country stock indexes
    """
    
    #Table 7: US
    ols1_country_us = ols(df_country.dropna(subset="sp500"),"sp500",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2','ecb_pos_dummy2',
    'ecb_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2', 'bankofengland_pos_dummy2',
    'bankofengland_neg_dummy2', 'RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    #print_ols(ols1_country_us,"countries_us")
    
    #Table 7: EU
    ols1_country_eu = ols(df_country.dropna(subset="eu_diff"),"eu_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2','ecb_pos_dummy2',
    'ecb_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2', 'bankofengland_pos_dummy2',
    'bankofengland_neg_dummy2', 'RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    #print_ols(ols1_country_eu,"countries_eu")
    
    #Table 7: UK
    ols1_country_gb = ols(df_country.dropna(subset="gb_diff"),"gb_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2','ecb_pos_dummy2',
    'ecb_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2', 'bankofengland_pos_dummy2',
    'bankofengland_neg_dummy2', 'RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    #print_ols(ols1_country_gb,"countries_gb")
    
    #Table 7: AU
    ols1_country_au = ols(df_country.dropna(subset="au_diff"),"au_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2','ecb_pos_dummy2',
    'ecb_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2', 'bankofengland_pos_dummy2',
    'bankofengland_neg_dummy2', 'RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    #print_ols(ols1_country_au,"countries_au")
    
    #Table 7: CA
    ols1_country_ca = ols(df_country.dropna(subset="ca_diff"),"ca_diff",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2','ecb_pos_dummy2',
    'ecb_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2', 'bankofengland_pos_dummy2',
    'bankofengland_neg_dummy2', 'RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    #print_ols(ols1_country_ca,"countries_ca")
    
    
    
    
    
    #explaining UK correlations
    #creates a chart to examine why the Bank of England's posts are so significant to multiple economies in table 7
    uk = tweets[tweets.username=="bankofengland"]
    plt.figure(figsize=(10,6))
    
    plt.title("Bank of England")
    plt.xlabel("Date")
    plt.ylabel("Sentiment")
    
    #only considering posts that exceed the non-neutral thresholds
    data = uk[uk["positive"]>.2]
    data1 = uk[uk["negative"]>.2]

    plt.scatter(data["Date"],data["positive"],color ="r")
    plt.scatter(data1["Date"],data1["negative"],color ="b")
    plt.show()
    
    
    """
    This portion corresponds to table 8.
    It is an attempt to compare country-specific sentiment to exchange rate differences
    """
    #average quarterly sentiment by country (central bank)
    quart_sent= pd.DataFrame(index=range(5)) 
    
    for i in countries:
        for j in ["pos","neg"]:
            quart_sent[f"{i}_avg_sent_{j}"]= 0 
    
    #quarterly classifications. 
    # *Data is too sparce to use these*
    for i in countries:
        quart_sent[f"{i}_avg_sent_pos"][1] = df_country[f"{i}_positive"][:62].mean()
        quart_sent[f"{i}_avg_sent_neg"][1]  = df_country[f"{i}_negative"][:62].mean()
        
        quart_sent[f"{i}_avg_sent_pos"][2] = df_country[f"{i}_positive"][62:125].mean()
        quart_sent[f"{i}_avg_sent_neg"][2]  = df_country[f"{i}_negative"][62:125].mean()
        
        quart_sent[f"{i}_avg_sent_pos"][3] = df_country[f"{i}_positive"][125:189].mean()
        quart_sent[f"{i}_avg_sent_neg"][3]  = df_country[f"{i}_negative"][125:189].mean()
        
        quart_sent[f"{i}_avg_sent_pos"][4] = df_country[f"{i}_positive"][189:].mean()
        quart_sent[f"{i}_avg_sent_neg"][4]  = df_country[f"{i}_negative"][189:].mean()
        
    
    
    #monthly classifications for Q4 of 2022
    #unfortunatly, this was the only period of the dataset with sufficient data for this analysis
    month_sent= pd.DataFrame(index=range(4)) 
    
    for i in countries:
        for j in ["pos","neg"]:
            month_sent[f"{i}_avg_sent_{j}"]= 0 


    for i in countries:
        month_sent[f"{i}_avg_sent_pos"][0] = df_country[f"{i}_positive"][168:189].mean()
        month_sent[f"{i}_avg_sent_neg"][0]  = df_country[f"{i}_negative"][168:189].mean()
        
        month_sent[f"{i}_avg_sent_pos"][1] = df_country[f"{i}_positive"][189:209].mean()
        month_sent[f"{i}_avg_sent_neg"][1]  = df_country[f"{i}_negative"][189:209].mean()
        
        month_sent[f"{i}_avg_sent_pos"][2] = df_country[f"{i}_positive"][209:229].mean()
        month_sent[f"{i}_avg_sent_neg"][2]  = df_country[f"{i}_negative"][209:229].mean()
        
        month_sent[f"{i}_avg_sent_pos"][3] = df_country[f"{i}_positive"][229:].mean()
        month_sent[f"{i}_avg_sent_neg"][3]  = df_country[f"{i}_negative"][229:].mean()
    
    
    #loading exchange rate data (OECD, 2023)
    ex_rates = pd.read_csv("data//EX_rates.csv")
    #ex_rates = ex_rates[for ex_rates["LOCATION"] in ["AUS","CAN"]]
    
    
    """
    This section creates the pairwise comparisons of currencies used in table 8
    """
    
    #AUS to USD
    bank = "RBAInfo"
    country = "AUS"
    aus_us_pos = pd.DataFrame(month_sent["federalreserve_avg_sent_pos"]/month_sent[f"{bank}_avg_sent_pos"])
    aus_us_pos["inverse"]= month_sent[f"{bank}_avg_sent_pos"]/month_sent["federalreserve_avg_sent_pos"]
    
    aus_us_neg = pd.DataFrame(month_sent[f"{bank}_avg_sent_neg"]/month_sent["federalreserve_avg_sent_neg"])
    aus_us_neg["inverse"]= month_sent[f"{bank}_avg_sent_neg"]/month_sent["federalreserve_avg_sent_neg"]
    
    aus_us_pos["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"]
    aus_us_neg["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"]
    
    #CAD to USD
    bank = "bankofcanada"
    country = "CAN"
    cad_us_pos = pd.DataFrame(month_sent["federalreserve_avg_sent_pos"]/month_sent[f"{bank}_avg_sent_pos"])
    cad_us_pos["inverse"] = month_sent[f"{bank}_avg_sent_pos"]/month_sent["federalreserve_avg_sent_pos"]
    
    cad_us_neg = pd.DataFrame(month_sent[f"{bank}_avg_sent_neg"]/month_sent["federalreserve_avg_sent_neg"])
    cad_us_neg["inverse"] = month_sent["federalreserve_avg_sent_neg"]/month_sent[f"{bank}_avg_sent_neg"]
    
    cad_us_pos["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"].reset_index().drop(columns="index")
    cad_us_neg["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"].reset_index().drop(columns="index")
    
    #Euro to USD
    bank = "ecb"
    country = "EA19"
    eu_us_pos = pd.DataFrame(month_sent["federalreserve_avg_sent_pos"]/month_sent[f"{bank}_avg_sent_pos"])
    eu_us_pos["inverse"] = month_sent[f"{bank}_avg_sent_pos"]/month_sent["federalreserve_avg_sent_pos"]
    
    eu_us_neg = pd.DataFrame(month_sent[f"{bank}_avg_sent_neg"]/month_sent["federalreserve_avg_sent_neg"])
    eu_us_neg["inverse"] = month_sent["federalreserve_avg_sent_neg"]/month_sent[f"{bank}_avg_sent_neg"]
    
    eu_us_pos["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"].reset_index().drop(columns="index")
    eu_us_neg["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"].reset_index().drop(columns="index")
    
    #GBP to USD
    bank = "bankofengland"
    country = "GBR"
    uk_us_pos = pd.DataFrame(month_sent["federalreserve_avg_sent_pos"]/month_sent[f"{bank}_avg_sent_pos"])
    uk_us_pos["inverse"] = month_sent[f"{bank}_avg_sent_pos"]/month_sent["federalreserve_avg_sent_pos"]
    
    uk_us_neg = pd.DataFrame(month_sent[f"{bank}_avg_sent_neg"]/month_sent["federalreserve_avg_sent_neg"])
    uk_us_neg["inverse"] = month_sent["federalreserve_avg_sent_neg"]/month_sent[f"{bank}_avg_sent_neg"]
    
    uk_us_pos["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"].reset_index().drop(columns="index")
    uk_us_neg["ex_rate"] = ex_rates[ex_rates["LOCATION"]==country]["Value"].reset_index().drop(columns="index")
    
    
    
    
    #the goal of the following plots is to observe a shared direction between the changes in ratios and exchange rates

    #plotting sentiment ratios to exchange rates for the negative sentiment comparisions
    plot_ex(uk_us_neg, "GBP","UK")
    plot_ex(eu_us_neg, "EUR","EU")
    plot_ex(aus_us_neg, "AUD","AU")
    plot_ex(uk_us_neg, "CAD","CA")
    
    #plotting sentiment ratios to exchange rates for the positive sentiment comparisions
    plot_ex(uk_us_pos, "GBP","UK")
    plot_ex(eu_us_pos, "EUR","EU")
    plot_ex(aus_us_pos, "AUD","AU")
    plot_ex(uk_us_pos, "CAD","CA")





    
    
    
    """
    Other Regressions. These regressions do not appear in the final paper
    """
    
    
    #Regression 1.1
    #using the impression counts (how many people have seen the tweet) in addition to Regression 1
    m1 = ols(df,"dollar_diff",["imp_central_bank","imp_fin_poster",'fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'])
    
    
    
    #Regression 2.1
    #bank var only
    bank_var_only = ols(df.dropna(subset = ["bank_var"]),"dollar_diff",["bank_var"])
    
    #Regression 3.1
    #with only bank variables
    ols4 =ols(df.dropna(subset = ["bank_var"]),"dollar_diff",["bank_var",
                                                              'central_bank_pos_dummy2',
                                                              'central_bank_neg_dummy2'])
    #Regression 4.1
    #a variant of Regression 4 that uses the S&P 500 as the dependant variable
    ols1_country_sp500 = ols(df_country.dropna(subset=["sp500"]),"sp500",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2','ecb_pos_dummy2',
    'ecb_neg_dummy2', 'federalreserve_pos_dummy2',
    'federalreserve_neg_dummy2', 'bankofengland_pos_dummy2',
    'bankofengland_neg_dummy2', 'RBAInfo_pos_dummy2', 'RBAInfo_neg_dummy2',
    'bankofcanada_pos_dummy2', 'bankofcanada_neg_dummy2'])
    
    #Regression 5.1
    #a variant of Regression 5 where the positive dummies for fin_poster and central_bank are also removed
    ols1_country_neg_only = ols(df_country,"dollar_diff",[
    "fin_poster_neg_dummy",
    "central_bank_neg_dummy",
    'ecb_neg_dummy2', 
    'federalreserve_neg_dummy2', 
    'bankofengland_neg_dummy2', 
    'RBAInfo_neg_dummy2',
    'bankofcanada_neg_dummy2'])
    
    #Regression 5.1.1
    #a variant of Regression 5 where the positive dummies for fin_poster and central_bank are also removed
    #now also changes the depenant variable to the S&P 500
    ols1_country_sp= ols(df_country.dropna(subset=["sp500"]),"sp500",['fin_poster_pos_dummy', 'fin_poster_neg_dummy', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2',
    'ecb_neg_dummy2', 
    'federalreserve_neg_dummy2', 
    'bankofengland_neg_dummy2', 
    'RBAInfo_neg_dummy2',
    'bankofcanada_neg_dummy2'])
    
    
    

    #banks only
    banks_only = ols(df,"dollar_diff",['central_bank_pos_dummy2',
    'central_bank_neg_dummy2'])
    
    #Thresholds only
    m1_2 = ols(df,"dollar_diff",['fin_poster_pos_dummy2', 'fin_poster_neg_dummy2', 'central_bank_pos_dummy2',
    'central_bank_neg_dummy2'])
    #print_ols(m1,"analysis_set_dummies2")
    
    
if __name__ == "__main__":
    main()                            
                                                                         

    
    

    
    