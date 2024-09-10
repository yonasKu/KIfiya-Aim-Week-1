import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# Import custom function to load data
sys.path.append(os.path.abspath('./scripts'))
from telecom_analysis import load_data_using_sqlalchemy

def main():
    st.title('Telecom Data Analysis')

    # Sidebar for task selection
    st.sidebar.header('Select Task')
    task = st.sidebar.selectbox('Choose Task', ['Overview Analysis', 'User Engagement Analysis'])

    # Load data
    query = "SELECT * FROM xdr_data;"
    df = load_data_using_sqlalchemy(query)

    if df is not None:
        st.write("Successfully loaded the data")

        if task == 'Overview Analysis':
            overview_analysis(df)
        elif task == 'User Engagement Analysis':
            user_engagement_analysis(df)
       
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
    

if __name__ == "__main__":
    main()
