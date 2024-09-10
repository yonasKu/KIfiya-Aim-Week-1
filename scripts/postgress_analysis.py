
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


def experience_analysis(df):

    st.header('Experience Analysis')


    # Task 3.1: Clean and Aggregate
    def clean_and_aggregate(df):
        df.fillna({
            'TCP DL Retrans. Vol (Bytes)': df['TCP DL Retrans. Vol (Bytes)'].mean(),
            'Avg RTT DL (ms)': df['Avg RTT DL (ms)'].mean(),
            'Avg Bearer TP DL (kbps)': df['Avg Bearer TP DL (kbps)'].mean(),
            'Handset Type': df['Handset Type'].mode()[0]
        }, inplace=True)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna({
            'TCP DL Retrans. Vol (Bytes)': df['TCP DL Retrans. Vol (Bytes)'].mean(),
            'Avg RTT DL (ms)': df['Avg RTT DL (ms)'].mean(),
            'Avg Bearer TP DL (kbps)': df['Avg Bearer TP DL (kbps)'].mean()
        }, inplace=True)
        
        aggregated = df.groupby('IMEI').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Handset Type': 'first'
        }).reset_index()
        
        return aggregated

    aggregated_df = clean_and_aggregate(df)
    print("Data after cleaning and aggregation:")
    print(aggregated_df.head())

    # Task 3.2: Compute and list top/bottom/frequent values
    def compute_statistics(df):
        top_10_tcp = df['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
        bottom_10_tcp = df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
        frequent_tcp = df['TCP DL Retrans. Vol (Bytes)'].mode()
        
        top_10_rtt = df['Avg RTT DL (ms)'].nlargest(10)
        bottom_10_rtt = df['Avg RTT DL (ms)'].nsmallest(10)
        frequent_rtt = df['Avg RTT DL (ms)'].mode()
        
        top_10_throughput = df['Avg Bearer TP DL (kbps)'].nlargest(10)
        bottom_10_throughput = df['Avg Bearer TP DL (kbps)'].nsmallest(10)
        frequent_throughput = df['Avg Bearer TP DL (kbps)'].mode()
        
        return {
            'top_10_tcp': top_10_tcp,
            'bottom_10_tcp': bottom_10_tcp,
            'frequent_tcp': frequent_tcp,
            'top_10_rtt': top_10_rtt,
            'bottom_10_rtt': bottom_10_rtt,
            'frequent_rtt': frequent_rtt,
            'top_10_throughput': top_10_throughput,
            'bottom_10_throughput': bottom_10_throughput,
            'frequent_throughput': frequent_throughput
        }

    stats_summary = compute_statistics(aggregated_df)
    print("Statistics summary:")
    print(stats_summary)

    # Task 3.3: Distribution of throughput and TCP retransmission
    def compute_and_plot_distributions(df):
        aggregated_data = df.groupby('Handset Type').agg({
            'Avg Bearer TP DL (kbps)': 'mean',
            'TCP DL Retrans. Vol (Bytes)': 'mean'
        }).reset_index()

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=df)
        plt.title('Distribution of Average Throughput per Handset Type')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        sns.boxplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=df)
        plt.title('Distribution of Average TCP Retransmission per Handset Type')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        print("Aggregated Data by Handset Type:")
        print(aggregated_data)

    compute_and_plot_distributions(df)

    # Task 3.4: K-Means Clustering
    def perform_kmeans_clustering(df, n_clusters=3):
        features = df[['Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)']].copy()
        features.fillna(features.mean(), inplace=True)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        df['Cluster'] = kmeans.fit_predict(scaled_features)
        
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=['Avg Bearer TP DL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)'])
        
        print("Cluster Centers (Original Scale):")
        print(cluster_centers_df)
        
        return df

    clustered_df = perform_kmeans_clustering(df)

    def describe_clusters(df):
        cluster_description = df.groupby('Cluster').agg({
            'Avg Bearer TP DL (kbps)': ['mean', 'std'],
            'TCP DL Retrans. Vol (Bytes)': ['mean', 'std'],
            'Avg RTT DL (ms)': ['mean', 'std'],
            'IMEI': 'count'
        }).reset_index()

        print("\nCluster Descriptions:")
        print(cluster_description)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='Avg Bearer TP DL (kbps)', y='TCP DL Retrans. Vol (Bytes)', hue='Cluster', data=df, palette='Set1')
        plt.title('K-Means Clustering of Users')
        plt.xlabel('Average Throughput (kbps)')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.legend(title='Cluster')
        plt.show()

    describe_clusters(clustered_df)

