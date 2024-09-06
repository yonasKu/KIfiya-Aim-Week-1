
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Establish a connection to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Load data using pandas
        df = pd.read_sql_query(query, connection)

        # Close the database connection
        connection.close()

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define your SQL query
query = """
SELECT handset, COUNT(*) as usage_count
FROM xdr_table
GROUP BY handset
ORDER BY usage_count DESC
LIMIT 10;
"""

# Load data from PostgreSQL
df_handsets = load_data_from_postgres(query)

# Inspect the data
if df_handsets is not None:
    print(df_handsets.head())
else:
    print("Failed to load data.")
