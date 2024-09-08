import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_using_sqlalchemy(query):
    try:
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_string)
        df = pd.read_sql_query(query, engine)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        return df
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def clean_and_aggregate(df):
    grouped = df.groupby('IMEI').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Handset Type': 'first'
    }).reset_index()
    return grouped

def plot_throughput_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=df)
    plt.xticks(rotation=45)
    plt.title('Distribution of Average Throughput per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average Throughput (kbps)')
    st.pyplot(plt)

def plot_tcp_retransmission_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=df)
    plt.xticks(rotation=45)
    plt.title('Distribution of Average TCP Retransmission per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average TCP Retransmission (Bytes)')
    st.pyplot(plt)

def perform_kmeans(df):
    features = df[['Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    return df, cluster_centers

# Streamlit app layout
st.title('Telecommunication Data Dashboard')

query = "SELECT * FROM xdr_data;"
df = load_data_using_sqlalchemy(query)

if df is not None:
    st.sidebar.header('Select an Option')
    option = st.sidebar.selectbox('Choose a Task', 
                                  ['Distribution of Average Throughput',
                                   'Distribution of Average TCP Retransmission',
                                   'K-Means Clustering'])
    
    aggregated_df = clean_and_aggregate(df)
    
    if option == 'Distribution of Average Throughput':
        st.subheader('Average Throughput Distribution')
        plot_throughput_distribution(aggregated_df)

    elif option == 'Distribution of Average TCP Retransmission':
        st.subheader('Average TCP Retransmission Distribution')
        plot_tcp_retransmission_distribution(aggregated_df)

    elif option == 'K-Means Clustering':
        st.subheader('K-Means Clustering')
        clustered_df, cluster_centers = perform_kmeans(aggregated_df)
        st.write(clustered_df.head())
        st.write("Cluster Centers (Original Scale):")
        st.write(pd.DataFrame(cluster_centers, columns=['Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)']))
else:
    st.error("Failed to load data.")
