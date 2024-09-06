import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from scipy import stats

# Load environment variables from .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_using_sqlalchemy(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query using SQLAlchemy.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Create a connection string
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, engine)

        # Handle missing values by filling them with 0
        df.fillna(0, inplace=True)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define SQL query to get the top 10 handsets
query_handsets = """
SELECT "Handset Type" AS handset, COUNT(*) AS usage_count
FROM public.xdr_data
GROUP BY "Handset Type"
ORDER BY usage_count DESC
LIMIT 10;
"""

# Load data from PostgreSQL
df_handsets = load_data_using_sqlalchemy(query_handsets)

# Inspect the data
if df_handsets is not None:
    print("Top 10 handsets:")
    print(df_handsets.head(10))
else:
    print("Failed to load data.")

# Define SQL query to identify top 3 manufacturers
query_manufacturers = """
SELECT "Handset Manufacturer" AS manufacturer, COUNT(*) AS usage_count
FROM public.xdr_data
GROUP BY "Handset Manufacturer"
ORDER BY usage_count DESC
LIMIT 3;
"""

df_manufacturers = load_data_using_sqlalchemy(query_manufacturers)

if df_manufacturers is not None:
    print("Top 3 manufacturers:")
    print(df_manufacturers.head())
else:
    print("Failed to load data.")

# Identify top 5 handsets per top 3 handset manufacturer
if df_manufacturers is not None:
    manufacturers = df_manufacturers['manufacturer'].tolist()
    for manufacturer in manufacturers:
        query_top_handsets = f"""
        SELECT "Handset Type" AS handset, COUNT(*) AS usage_count
        FROM public.xdr_data
        WHERE "Handset Manufacturer" = '{manufacturer}'
        GROUP BY "Handset Type"
        ORDER BY usage_count DESC
        LIMIT 5;
        """
        df_top_handsets = load_data_using_sqlalchemy(query_top_handsets)
        
        if df_top_handsets is not None:
            print(f"Top 5 handsets for {manufacturer}:")
            print(df_top_handsets.head())
        else:
            print(f"Failed to load data for manufacturer {manufacturer}.")

# Define SQL query to aggregate user behavior
query_user_behavior = """
SELECT
    "MSISDN/Number" AS user,
    COUNT(*) AS session_count,
    SUM("Dur. (ms)") AS total_session_duration, -- Assuming "Dur. (ms)" is session duration
    SUM("Total DL (Bytes)") AS total_download,
    SUM("Total UL (Bytes)") AS total_upload,
    SUM("Total DL (Bytes)") + SUM("Total UL (Bytes)") AS total_data_volume,
    SUM("Social Media DL (Bytes)") AS social_media_dl,
    SUM("Social Media UL (Bytes)") AS social_media_ul,
    SUM("Google DL (Bytes)") AS google_dl,
    SUM("Google UL (Bytes)") AS google_ul,
    SUM("Email DL (Bytes)") AS email_dl,
    SUM("Email UL (Bytes)") AS email_ul,
    SUM("Youtube DL (Bytes)") AS youtube_dl,
    SUM("Youtube UL (Bytes)") AS youtube_ul,
    SUM("Netflix DL (Bytes)") AS netflix_dl,
    SUM("Netflix UL (Bytes)") AS netflix_ul,
    SUM("Gaming DL (Bytes)") AS gaming_dl,
    SUM("Gaming UL (Bytes)") AS gaming_ul,
    SUM("Other DL (Bytes)") AS other_dl,
    SUM("Other UL (Bytes)") AS other_ul
FROM public.xdr_data
GROUP BY "MSISDN/Number";
"""

# Load user behavior data
df_user_behavior = load_data_using_sqlalchemy(query_user_behavior)

# Inspect the data
if df_user_behavior is not None:
    print("User behavior analysis:")
    print(df_user_behavior.head())
else:
    print("Failed to load data.")



print(df_user_behavior.info())
print(df_user_behavior.describe())


# Check for missing values
print(df_user_behavior.isnull().sum())

# Replace missing values with mean for numeric columns
df_user_behavior.fillna(df_user_behavior.mean(), inplace=True)



z_scores = stats.zscore(df_user_behavior.select_dtypes(include=['float64', 'int64']))
abs_z_scores = np.abs(z_scores)
df_user_behavior = df_user_behavior[(abs_z_scores < 3).all(axis=1)]