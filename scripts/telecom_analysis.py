import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
    Connect to PostgreSQL and load data based on the provided SQL query.
    """
    try:
        # Create connection string
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_string)
        df = pd.read_sql_query(query, engine)
        df.fillna(0, inplace=True)  # Handle missing values
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def preprocess_data(df):
    """
    Preprocess the dataframe by handling missing values, outliers, and normalizing data.
    """
    # Check the columns from the database schema
    print("Columns in the dataframe:", df.columns)
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Remove outliers by capping values at the 99th percentile for session duration, download, and upload
    for column in ["Dur. (ms)", "Total DL (Bytes)", "Total UL (Bytes)"]:
        upper_limit = df[column].quantile(0.99)
        df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])

    # Normalize relevant columns
    scaler = StandardScaler()
    df[["Dur. (ms)", "Total DL (Bytes)", "Total UL (Bytes)"]] = scaler.fit_transform(
        df[["Dur. (ms)", "Total DL (Bytes)", "Total UL (Bytes)"]]
    )

    return df


def segment_users(df):
    """
    Segment users into deciles based on session duration ('Dur. (ms)').
    """
    # Use "Dur. (ms)" column for segmentation and drop duplicate bin edges
    df['decile'] = pd.qcut(df['Dur. (ms)'], 10, labels=False, duplicates='drop')
    return df


def perform_pca(df, columns, n_components=2):
    """
    Perform PCA on the provided columns.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df[columns])
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = explained_variance.sum()

    return pca_result, explained_variance, total_explained_variance


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_users(df, n_clusters=5):
    """
    Perform KMeans clustering on users and add cluster labels to the dataframe.
    """
    # Check if necessary columns are present
    required_columns = ['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'HTTP DL (Bytes)']  # Updated to match schema
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Extract the relevant data
    data_for_clustering = df[required_columns].copy()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Return the dataframe with cluster labels and the KMeans model
    return df, kmeans



def visualize_correlation(df, columns):
    """
    Visualize the correlation matrix of the specified columns.
    """
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')
    plt.title('Correlation Matrix')
    plt.show()
