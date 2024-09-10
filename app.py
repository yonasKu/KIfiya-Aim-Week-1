# Import standard libraries
import os
import sys

# Import data manipulation libraries
import pandas as pd
import numpy as np

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import statistical and machine learning libraries
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import Streamlit for web-based app
import streamlit as st

# Add custom scripts to the system path
sys.path.append(os.path.abspath('./scripts'))

# Import custom function to load data
from telecom_analysis import load_data_using_sqlalchemy

# Import custom function to load data
# 

sys.path.append(os.path.abspath('./scripts'))

from telecom_analysis import load_data_using_sqlalchemy


def main():
    st.title('Telecom Data Analysis')

    # Sidebar for task selection
    st.sidebar.header('Select Tasks')
    tasks = st.sidebar.multiselect('Choose Tasks', ['Overview Analysis', 'User Engagement Analysis', 'Experience Analysis'])

    # Load data
    query = "SELECT * FROM xdr_data;"
    df = load_data_using_sqlalchemy(query)

    if df is not None:
        st.write("Successfully loaded the data")

        if 'Overview Analysis' in tasks:
            overview_analysis(df)
        if 'User Engagement Analysis' in tasks:
            user_engagement_analysis(df)
        if 'Experience Analysis' in tasks:
            experience_analysis(df)
    else:
        st.error("Failed to load data.")


def overview_analysis(df):
    st.header('Overview Analysis')
    
    # Top 10 Handsets
    query_handsets = """
    SELECT "Handset Type" AS handset, COUNT(*) AS usage_count
    FROM public.xdr_data
    GROUP BY "Handset Type"
    ORDER BY usage_count DESC
    LIMIT 10;
    """
    df_handsets = load_data_using_sqlalchemy(query_handsets)

    if df_handsets is not None:
        st.subheader("Top 10 Handsets")
        st.write(df_handsets)
    else:
        st.error("Failed to load handsets data.")

    # Top 3 Manufacturers
    query_manufacturers = """
    SELECT "Handset Manufacturer" AS manufacturer, COUNT(*) AS usage_count
    FROM public.xdr_data
    GROUP BY "Handset Manufacturer"
    ORDER BY usage_count DESC
    LIMIT 3;
    """
    df_manufacturers = load_data_using_sqlalchemy(query_manufacturers)

    if df_manufacturers is not None:
        st.subheader("Top 3 Manufacturers")
        st.write(df_manufacturers)
        
        # Top 5 Handsets per Manufacturer
        for manufacturer in df_manufacturers['manufacturer']:
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
                st.subheader(f"Top 5 Handsets for {manufacturer}")
                st.write(df_top_handsets)
            else:
                st.error(f"Failed to load data for manufacturer {manufacturer}.")

    # User Behavior Analysis
    query_user_behavior = """
    SELECT
        "MSISDN/Number" AS user,
        COUNT(*) AS session_count,
        SUM("Dur. (ms)") AS total_session_duration,
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
    df_user_behavior = load_data_using_sqlalchemy(query_user_behavior)

    if df_user_behavior is not None:
        st.subheader("User Behavior Data")
        st.write(df_user_behavior.head())

        # Data Cleaning
        df_user_behavior.fillna(df_user_behavior.mean(), inplace=True)
        z_scores = stats.zscore(df_user_behavior.select_dtypes(include=['float64', 'int64']))
        df_user_behavior = df_user_behavior[(np.abs(z_scores) < 3).all(axis=1)]

        # Segment users into deciles
        df_user_behavior['decile'] = pd.qcut(df_user_behavior['total_session_duration'], 10, labels=False) + 1
        decile_data = df_user_behavior.groupby('decile')[['total_download', 'total_upload']].sum()
        st.subheader("Data per Decile")
        st.write(decile_data)

        # Metrics
        metrics = df_user_behavior[['total_session_duration', 'total_download', 'total_upload', 'total_data_volume']].agg(['mean', 'median', 'std', 'min', 'max'])
        st.subheader("Metrics")
        st.write(metrics)

        # Data Visualization
        st.subheader("Histograms")
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        columns = ['total_session_duration', 'total_download', 'total_upload', 'total_data_volume']
        for i, column in enumerate(columns):
            ax = plt.subplot(2, 2, i + 1)
            df_user_behavior[column].hist(ax=ax)
            ax.set_title(f'Histogram of {column}')
        st.pyplot(fig)

        st.subheader("Box Plots")
        for column in columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=df_user_behavior[column], ax=ax)
            ax.set_title(f'Box Plot of {column}')
            st.pyplot(fig)

        # Scatter plots
        st.subheader("Scatter Plots")
        application_columns = ['social_media_dl', 'google_dl', 'email_dl', 'youtube_dl', 'netflix_dl', 'gaming_dl', 'other_dl']
        for app in application_columns:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df_user_behavior[app], y=df_user_behavior['total_data_volume'], ax=ax)
            ax.set_title(f'Relationship between {app} and Total Data Volume')
            st.pyplot(fig)

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        correlation_matrix = df_user_behavior[application_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix of Application Data')
        st.pyplot(fig)

        # PCA
        st.subheader("PCA - Dimensionality Reduction")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_user_behavior[application_columns])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        fig, ax = plt.subplots()
        ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        ax.set_title('PCA - Dimensionality Reduction')
        st.pyplot(fig)

        explained_variance = pca.explained_variance_ratio_
        st.write(f'Explained variance by the 2 components: {explained_variance.sum():.2f}')
        st.write(f'Explained variance ratio (how much information each component contains): {explained_variance}')
        for i, var in enumerate(explained_variance, start=1):
            st.write(f'Principal Component {i} explains {var:.2%} of the variance')
        cumulative_variance = explained_variance.sum()
        st.write(f'Cumulative explained variance by the 2 components: {cumulative_variance:.2%}')

        st.write(df_user_behavior.info())
        st.write(df_user_behavior.describe())
    else:
        st.error("Failed to load user behavior data.")

def user_engagement_analysis(df):
    st.header('User Engagement Analysis')
    
    # Task 2.1 - Aggregate metrics per customer ID (MSISDN)
    user_engagement = df.groupby('MSISDN/Number').agg({
        'Dur. (ms)': 'sum',  # Session duration
        'HTTP DL (Bytes)': 'sum',  # Download bytes
        'HTTP UL (Bytes)': 'sum',  # Upload bytes
        'Bearer Id': 'count'  # Session frequency
    }).rename(columns={
        'Dur. (ms)': 'total_session_duration',
        'HTTP DL (Bytes)': 'total_download',
        'HTTP UL (Bytes)': 'total_upload',
        'Bearer Id': 'session_frequency'
    })

    # Calculate total traffic (download + upload)
    user_engagement['total_traffic'] = user_engagement['total_download'] + user_engagement['total_upload']

    # Display the top 10 customers per engagement metric
    st.subheader("Top 10 Customers by Session Frequency")
    st.write(user_engagement.sort_values('session_frequency', ascending=False).head(10))

    st.subheader("Top 10 Customers by Session Duration")
    st.write(user_engagement.sort_values('total_session_duration', ascending=False).head(10))

    st.subheader("Top 10 Customers by Total Traffic")
    st.write(user_engagement.sort_values('total_traffic', ascending=False).head(10))

    # Normalize the metrics
    scaler = StandardScaler()
    normalized_metrics = scaler.fit_transform(user_engagement[['session_frequency', 'total_session_duration', 'total_traffic']])

    # Apply K-Means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_engagement['engagement_cluster'] = kmeans.fit_predict(normalized_metrics)

    user_engagement['cluster'] = kmeans.labels_
    
    # Visualize the clusters
    st.subheader("K-Means Clustering of User Engagement")
    fig, ax = plt.subplots()
    sns.pairplot(user_engagement, vars=['session_frequency', 'total_session_duration', 'total_traffic'], hue='cluster', palette='Set2')
    st.pyplot(fig)

    # Group by cluster and calculate summary statistics for each cluster
    st.subheader("Cluster Summary Statistics")
    cluster_summary = user_engagement.groupby('cluster').agg({
        'session_frequency': ['min', 'max', 'mean', 'sum'],
        'total_session_duration': ['min', 'max', 'mean', 'sum'],
        'total_traffic': ['min', 'max', 'mean', 'sum']
    })
    st.write(cluster_summary)

    # Aggregate user total traffic per application
    applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
    user_engagement_apps = pd.DataFrame()

    for app in applications:
        app_dl = f'{app} DL (Bytes)'
        app_ul = f'{app} UL (Bytes)'
        
        if app_dl in df.columns and app_ul in df.columns:
            user_engagement_apps[f'{app}_total_traffic'] = df.groupby('MSISDN/Number')[[app_dl, app_ul]].sum().sum(axis=1)
            
            # Top 10 users per application by traffic
            st.subheader(f"Top 10 Users by {app} Traffic")
            top_users = user_engagement_apps[[f'{app}_total_traffic']].sort_values(f'{app}_total_traffic', ascending=False).head(10)
            st.write(top_users)
        else:
            st.warning(f"Columns for {app} not found in DataFrame.")

    # Plot the top 3 most used applications
    total_traffic = {app: user_engagement_apps[f'{app}_total_traffic'].sum() for app in applications if f'{app}_total_traffic' in user_engagement_apps.columns}
    top_3_apps = sorted(total_traffic.items(), key=lambda x: x[1], reverse=True)[:3]

    # Create a DataFrame for plotting
    top_3_df = pd.DataFrame(top_3_apps, columns=['Application', 'Total Traffic'])

    st.subheader("Top 3 Most Used Applications")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Application', y='Total Traffic', data=top_3_df, palette='viridis')
    st.pyplot(fig)

    # Determine optimal k using the Elbow Method
    st.subheader("Elbow Method for Optimal k")
    inertia = []
    k_range = range(1, min(len(user_engagement), 11))  # Ensure k does not exceed number of samples

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_metrics)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertia, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    st.pyplot(fig)

    # Based on the elbow plot, choose the optimal k
    optimal_k = 3  # Replace with the value you determine from the elbow plot

    # Fit k-means with the optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    user_engagement['cluster'] = kmeans.fit_predict(normalized_metrics)

    # Compute min, max, average, and total non-normalized metrics for each cluster
    st.subheader("Cluster Metrics")
    cluster_metrics = user_engagement.groupby('cluster').agg({
        'session_frequency': ['min', 'max', 'mean', 'sum'],
        'total_session_duration': ['min', 'max', 'mean', 'sum'],
        'total_traffic': ['min', 'max', 'mean', 'sum']
    })
    st.write(cluster_metrics)

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
    st.subheader("Data after Cleaning and Aggregation")
    st.write(aggregated_df.head())

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
    st.subheader("Statistics Summary")
    st.write("Top 10 TCP Retransmission:", stats_summary['top_10_tcp'])
    st.write("Bottom 10 TCP Retransmission:", stats_summary['bottom_10_tcp'])
    st.write("Frequent TCP Retransmission:", stats_summary['frequent_tcp'])
    st.write("Top 10 RTT:", stats_summary['top_10_rtt'])
    st.write("Bottom 10 RTT:", stats_summary['bottom_10_rtt'])
    st.write("Frequent RTT:", stats_summary['frequent_rtt'])
    st.write("Top 10 Throughput:", stats_summary['top_10_throughput'])
    st.write("Bottom 10 Throughput:", stats_summary['bottom_10_throughput'])
    st.write("Frequent Throughput:", stats_summary['frequent_throughput'])

    # Task 3.3: Distribution of throughput and TCP retransmission
    def compute_and_plot_distributions(df):
        aggregated_data = df.groupby('Handset Type').agg({
            'Avg Bearer TP DL (kbps)': 'mean',
            'TCP DL Retrans. Vol (Bytes)': 'mean'
        }).reset_index()

        st.subheader('Distribution of Throughput and TCP Retransmission')
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=df, ax=ax[0])
        ax[0].set_title('Distribution of Average Throughput per Handset Type')
        ax[0].tick_params(axis='x', rotation=45)

        sns.boxplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=df, ax=ax[1])
        ax[1].set_title('Distribution of Average TCP Retransmission per Handset Type')
        ax[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        st.write("Aggregated Data by Handset Type:")
        st.write(aggregated_data)

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
        
        st.subheader("Cluster Centers (Original Scale)")
        st.write(cluster_centers_df)
        
        return df

    clustered_df = perform_kmeans_clustering(df)

    def describe_clusters(df):
        cluster_description = df.groupby('Cluster').agg({
            'Avg Bearer TP DL (kbps)': ['mean', 'std'],
            'TCP DL Retrans. Vol (Bytes)': ['mean', 'std'],
            'Avg RTT DL (ms)': ['mean', 'std'],
            'IMEI': 'count'
        }).reset_index()

        st.subheader("Cluster Descriptions")
        st.write(cluster_description)

        st.subheader("K-Means Clustering of Users")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(x='Avg Bearer TP DL (kbps)', y='TCP DL Retrans. Vol (Bytes)', hue='Cluster', data=df, palette='Set1', ax=ax)
        ax.set_title('K-Means Clustering of Users')
        ax.set_xlabel('Average Throughput (kbps)')
        ax.set_ylabel('Average TCP Retransmission (Bytes)')
        ax.legend(title='Cluster')
        st.pyplot(fig)

    describe_clusters(clustered_df)
if __name__ == "__main__":
    main()