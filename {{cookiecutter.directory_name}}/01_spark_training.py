# BUSINESS SCIENCE UNIVERSITY ----
# LEARNING LAB 66: SPARK IN PYTHON ----
# **** ----

# INSTALLATION REQUIREMENTS ----

# IF YOU WANT TO SET UP THE SAME CONDA ENVIRONMENT AS ME
# RUN IN TERMINAL:
# conda env create -f environment.yml

# 1. INSTALL PYTHON
# 2. INSTALL JAVA
# 3. INSTALL PYSPARK==3.2.0 (THIS ALSO INSTALLS SPARK V3.2.0 IN YOUR ENV)

# DATABRICKS USERS ----
# - Try the Databricks VSCode Extension

# LIBRARIES ----

import pandas as pd
import numpy as np
import plotly.express as px

# USING PYSPARK==3.2 (BUILT IN PANDAS)
from pyspark import pandas as ps
from pyspark.sql import SparkSession


# SETTING UP SPARK SESSION ----

spark = SparkSession.builder \
    .master("local[12]") \
    .appName("lab_66_pyspark") \
    .config("spark.driver.bindAddress","localhost") \
    .config("spark.ui.port", "4050") \
    .config("spark.driver.memory", "16g") \
    .config("spark.memory.fraction", "0.9") \
    .getOrCreate()

spark

spark.version

# SPARK WEB UI ----
# - Should be running on localhost:4050
# - Will use this to examine memory usage, jobs, and data cache

# NASDAQ ANALYSIS ----
# - About the data: 
#   I pulled from R using my tidyquant package 
#   This was done in Lab 65: Spark in R
#   It contains 5.8M rows & 4000 time series groups

nasdaq_df = ps.read_csv(r'/Users/lucasbagge/Documents/Projects/business\ sciecne\ /lab_66_pyspark/01-data-raw/nasdaq_data.csv')
nasdaq_df.head()

nasdaq_df.shape

nasdaq_df['symbol'].value_counts().shape

nasdaq_df.head().info()

# Change data types 
nasdaq_df['symbol'] = nasdaq_df['symbol'].astype("str")
nasdaq_df['date'] = nasdaq_df['date'].astype("datetime64")
nasdaq_df['adjusted'] = nasdaq_df['adjusted'].astype("float64")

nasdaq_df.head().info()
nasdaq_df.head()

# What is spark doing?
nasdaq_df.spark.explain()

nasdaq_df

# CHECKPOINTING ----
# - Can help to speed up spark by Caching operations
# - IF YOU GET WARNING NOT ENOUGH MEMORY HERE: Adjust your .config("spark.driver.memory", "16g") 

spark 

nasdaq_df = nasdaq_df.spark.local_checkpoint()

nasdaq_df.head()

# LAG OPERATION ----

nasdaq_shifted_df = nasdaq_df \
    .drop('date') \
    .groupby("symbol") \
    .shift(1) \
    .rename(columns={'adjusted':'lag1'}) \
    .sort_index()

# Create another local checkpoint to speed up 
nasdaq_shifted_df = nasdaq_shifted_df.spark.local_checkpoint()

nasdaq_shifted_df.head()

# COMBINE LAG WITH ADJUSTED PRICES ----
# - IF YOU GET ERROR: NEED TO SET OPTION TO COMPUTE ON MULTIPLE DATA FRAMES

ps.set_option('compute.ops_on_diff_frames', True)

nasdaq_lag_df = nasdaq_df.copy()
nasdaq_lag_df['lag1'] = nasdaq_shifted_df[['lag1']]

nasdaq_lag_df.head()

nasdaq_lag_df = nasdaq_lag_df.spark.local_checkpoint()
    
# SUMMARIZE RETURNS

nasdaq_agg_df = nasdaq_lag_df \
    .assign(returns = lambda x: (x['adjusted'] / x['lag1']) - 1) \
    .groupby('symbol') \
    .aggregate(
        {
            'returns': ['mean', 'std', 'count'],
            'date': ['max', 'min']
        }
    ) \
    .reset_index()

nasdaq_agg_df.head()

nasdaq_agg_df.columns = ["_".join(a) for a in nasdaq_agg_df.columns.to_flat_index()]

nasdaq_agg_df.head()

nasdaq_agg_df = nasdaq_agg_df.spark.local_checkpoint()



# JOIN NASDAQ INDEX META DATA ----

nasdaq_index_df = ps.read_csv(r'/Users/lucasbagge/Documents/Projects/business\ sciecne\ /lab_66_pyspark/01-data-raw/nasdaq_index.csv')

nasdaq_index_df.head()

nasdaq_index_df = nasdaq_index_df[['symbol', 'company', 'market.cap']]

nasdaq_index_df.head().info()

nasdaq_index_df['market.cap'] = nasdaq_index_df['market.cap'].astype('float64')

nasdaq_index_df['symbol'] = nasdaq_index_df['symbol'].astype('str')

nasdaq_index_df['company'] = nasdaq_index_df['company'].astype('str')

nasdaq_index_df.head().info()

# JOIN

nasdaq_agg_df.head().info()

nasdaq_index_df.head().info()

nasdaq_agg_df.columns


nasdaq_agg_join_df = nasdaq_agg_df \
    .rename({'symbol_':'symbol'}, axis = 1) \
    .set_index('symbol') \
    .join(nasdaq_index_df.set_index('symbol'), how = "left", lsuffix = "_l", rsuffix = "_r") \
    .reset_index()

nasdaq_agg_join_df.head()



# SCREEN & VISUALIZE RETURNS ----

type(nasdaq_agg_join_df)

nasdaq_screened_df = nasdaq_agg_join_df \
    .rename({"market.cap":"market_cap"}, axis=1) \
    .query("returns_std < 0.10") \
    .query("returns_count > 3*365") \
    .query("market_cap > 1e9") \
    .assign(reward_metric = lambda x: x['returns_mean'] / x['returns_std'] * 2500) \
    .to_pandas() 

nasdaq_screened_df

type(nasdaq_screened_df)

   
nasdaq_screened_df \
    .pipe(
        func           = px.scatter,
        x              = 'returns_std',
        y              = 'returns_mean',
        color          = 'reward_metric',
        hover_data     = ['company', 'symbol', 'market_cap'],
        render_mode    = 'svg',
        template       = 'plotly_dark'
    )

# FILTER & VISUALIZE TOP STOCKS

best_stocks_df = nasdaq_screened_df \
    .sort_values('reward_metric', ascending = False) \
    .head(10)

best_stocks_df['symbol'].values

_filter = nasdaq_df['symbol'].isin(best_stocks_df['symbol'].values)

nasdaq_df[_filter] \
    .to_pandas() \
    .pipe(
        func           = px.line,
        x              = 'date',
        y              = 'adjusted',
        color          = 'symbol',
        facet_col      = 'symbol',
        facet_col_wrap = 2, 
        render_mode    = 'svg',
        template       = 'plotly_dark'
    ) \
    .update_yaxes(matches=None, title = None) \
    .update_xaxes(title = None) \
    .update_layout(showlegend=True, font = dict(size=8)) \
    .update_traces(line = dict(width=0.7))

# STORE RESULTS ----

nasdaq_agg_join_df.to_parquet("02-data-processed/nasdaq_agg_join.parquet")

ps.read_parquet("02-data-processed/nasdaq_agg_join.parquet")

nasdaq_agg_join_df.to_pandas().to_pickle("02-data-processed/nasdaq_agg_join.pkl")

pd.read_pickle("02-data-processed/nasdaq_agg_join.pkl")

nasdaq_df.to_pandas().to_pickle("02-data-processed/nasdaq.pkl")

# CONCLUSIONS ----
# - Organizations need these skills (ref. job postings listing "Spark")

# 1. Data Engineering can now be done with Pandas (using PySpark). You BETTER LEARN pandas!!

# 2. Production isn't just models, but the ability to take your results and package them in a way users can use them. Dash is a great way to do this! You BETTER LEARN DASH!

# How are you going to learn these skills?

