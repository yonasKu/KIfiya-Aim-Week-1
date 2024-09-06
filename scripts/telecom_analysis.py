import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

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

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define your SQL query to get the top 10 handsets
query_handsets = """
SELECT "Handset Type" AS handset, COUNT(*) AS usage_count
FROM public.xdr_data
GROUP BY "Handset Type"
ORDER BY usage_count DESC
LIMIT 10
"""

# Load data from PostgreSQL
df_handsets = load_data_using_sqlalchemy(query_handsets)

# Inspect the data
if df_handsets is not None:
    print("Top 10 handsets:")
    print(df_handsets.head())
else:
    print("Failed to load data.")

# Define SQL queries to identify top 3 manufacturers and top 5 handsets per manufacturer
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