#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify Data Analysis Dashboard
-------------------------------
An interactive dashboard for exploring Spotify dataset using Streamlit.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(
    page_title="Spotify Data Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load the Spotify dataset."""
    try:
        df = pd.read_csv('data/spotify_2021.csv', encoding='latin1')
        return df
    except:
        try:
            df = pd.read_csv('data/spotify_2023.csv', encoding='latin1')
            return df
        except:
            st.error("Dataset not found. Please make sure to place either spotify_2021.csv or spotify_2023.csv in the data directory.")
            return None

def run_dashboard():
    """Main function to run the Streamlit dashboard."""
    st.title("🎵 Spotify Data Analysis Dashboard")
    st.markdown("""This dashboard provides interactive visualizations and insights from the Spotify dataset. Explore top songs, artists, audio features, and more.""")

    df = load_data()
    
    if df is None:
        return

    # Check and convert column datatypes
    if 'streams' in df.columns:
        df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

    st.sidebar.title("Navigation")
    pages = ["Dataset Overview", "Top Songs & Artists", "Audio Feature Analysis", 
             "Clustering Analysis"]
    selection = st.sidebar.radio("Go to", pages)

    if selection == "Dataset Overview":
        display_dataset_overview(df)
    elif selection == "Top Songs & Artists":
        display_top_songs_artists(df)
    elif selection == "Audio Feature Analysis":
        display_audio_feature_analysis(df)
    elif selection == "Clustering Analysis":
        display_clustering_analysis(df)

def display_dataset_overview(df):
    """Display dataset overview."""
    st.header("Dataset Overview")
    
    # Basic dataset information
    st.subheader("Basic Information")
    st.write("**Available Columns:**", list(df.columns))
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of songs:** {df.shape[0]}")
        st.write(f"**Number of features:** {df.shape[1]}")
    
    with col2:
        artist_col = next((col for col in ['artist_name', 'artist(s)_name', 'artist'] if col in df.columns), None)
        if artist_col:
            st.write(f"**Total number of artists:** {df[artist_col].nunique()}")
        
        if 'genre' in df.columns:
            st.write(f"**Total number of genres:** {df['genre'].nunique()}")
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Summary statistics
    st.subheader("Summary Statistics")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe())
    else:
        st.write("No numeric columns found for summary statistics.")
    
    # Missing values
    st.subheader("Missing Values")
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Feature', 'Missing Values']
    missing_data = missing_data[missing_data['Missing Values'] > 0]
    
    if missing_data.empty:
        st.write("No missing values found in the dataset.")
    else:
        st.dataframe(missing_data)

def display_top_songs_artists(df):
    """Display top songs and artists."""
    st.header("Top Songs & Artists")
    
    # Identify key columns
    track_col = next((col for col in ['track_name', 'track'] if col in df.columns), None)
    artist_col = next((col for col in ['artist_name', 'artist(s)_name', 'artist'] if col in df.columns), None)
    popularity_col = next((col for col in ['streams', 'popularity'] if col in df.columns), None)
    
    if not all([track_col, artist_col, popularity_col]):
        st.error(f"Missing required columns. Available columns: {df.columns.tolist()}")
        return
    
    tab1, tab2 = st.tabs(["Top Songs", "Top Artists"])
    
    with tab1:
        st.subheader(f"Top Songs by {popularity_col.capitalize()}")
        top_songs = df.sort_values(popularity_col, ascending=False).head(20)
        st.dataframe(top_songs[[track_col, artist_col, popularity_col]], height=400)

        # Bar plot for the top 10 most popular songs
        fig = px.bar(top_songs.head(10), x=popularity_col, y=track_col, 
                     orientation='h', color=popularity_col,
                     color_continuous_scale='Viridis',
                     title=f'Top 10 Songs by {popularity_col.capitalize()}',
                     labels={track_col: 'Song', popularity_col: popularity_col.capitalize()})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Top Artists")
        
        # Count of songs by artist
        artist_counts = df[artist_col].value_counts().reset_index().head(10)
        artist_counts.columns = ['Artist', 'Number of Songs']
        
        # Average popularity by artist
        artist_popularity = df.groupby(artist_col)[popularity_col].mean().reset_index()
        artist_popularity.columns = ['Artist', f'Average {popularity_col.capitalize()}']
        top_artists_by_popularity = artist_popularity.sort_values(f'Average {popularity_col.capitalize()}', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Artists with Most Songs**")
            fig1 = px.bar(artist_counts, x='Number of Songs', y='Artist', 
                         orientation='h', color='Number of Songs',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.write(f"**Artists with Highest Average {popularity_col.capitalize()}**")
            fig2 = px.bar(top_artists_by_popularity, x=f'Average {popularity_col.capitalize()}', y='Artist', 
                         orientation='h', color=f'Average {popularity_col.capitalize()}',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig2, use_container_width=True)

def display_audio_feature_analysis(df):
    """Display audio feature analysis."""
    st.header("Audio Feature Analysis")
    
    # Try to detect audio feature columns with both naming conventions
    potential_features = {
        "Danceability": ["danceability_%", "danceability"],
        "Energy": ["energy_%", "energy"],
        "Acousticness": ["acousticness_%", "acousticness"],
        "Instrumentalness": ["instrumentalness_%", "instrumentalness"],
        "Liveness": ["liveness_%", "liveness"],
        "Valence": ["valence_%", "valence"],
        "Speechiness": ["speechiness_%", "speechiness"],
        "Tempo": ["bpm", "tempo"]
    }
    
    # Find which features are available in the dataset
    feature_mapping = {}
    for display_name, possible_cols in potential_features.items():
        for col in possible_cols:
            if col in df.columns:
                feature_mapping[display_name] = col
                break
    
    if not feature_mapping:
        st.error("No audio features found in the dataset. Available columns: " + ", ".join(df.columns))
        return
    
    # Get the popularity/streams column
    popularity_col = next((col for col in ['streams', 'popularity'] if col in df.columns), None)
    
    # Distribution of audio features
    st.subheader("Distribution of Audio Features")
    display_features = list(feature_mapping.keys())
    selected_display_feature = st.selectbox("Select a feature to visualize:", display_features)
    selected_feature = feature_mapping[selected_display_feature]

    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig1 = px.histogram(df, x=selected_feature, nbins=30, 
                          title=f"Distribution of {selected_display_feature}")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Box plot
        fig2 = px.box(df, y=selected_feature, 
                    title=f"Box Plot of {selected_display_feature}")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Feature correlation if popularity/streams column exists
    if popularity_col:
        st.subheader("Feature Correlations")
        
        # Create a dataframe with just the features for correlation
        feature_cols = list(feature_mapping.values())
        correlation_columns = feature_cols + [popularity_col]
        correlation_df = df[correlation_columns].copy()
        
        # Handle missing values for correlation calculation
        correlation_df = correlation_df.apply(pd.to_numeric, errors='coerce')
        correlation_df = correlation_df.fillna(correlation_df.mean())
        
        # Calculate correlation with popularity/streams
        corr_matrix = correlation_df.corr()
        corr_with_popularity = corr_matrix[popularity_col].drop(popularity_col).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Correlation with {popularity_col.capitalize()}**")
            display_names = [name for name, col in feature_mapping.items() if col in corr_with_popularity.index]
            values = [corr_with_popularity.loc[feature_mapping[name]] for name in display_names]
            
            corr_data = pd.DataFrame({
                'Feature': display_names,
                'Correlation': values
            }).sort_values('Correlation', ascending=False)
            
            fig3 = px.bar(
                corr_data, 
                x='Correlation', 
                y='Feature',
                orientation='h',
                color='Correlation',
                color_continuous_scale='RdBu',
                title=f'Audio Features Correlation with {popularity_col.capitalize()}'
            )
            fig3.update_layout(xaxis_title="Correlation Coefficient", yaxis_title="Audio Feature")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.write("**Feature Relationships**")
            x_display = st.selectbox("Select X-axis feature:", display_features, index=0)
            y_display = st.selectbox("Select Y-axis feature:", display_features, index=min(1, len(display_features)-1))
            
            x_feature = feature_mapping[x_display]
            y_feature = feature_mapping[y_display]
            
            fig4 = px.scatter(df, x=x_feature, y=y_feature, color=popularity_col,
                            hover_name=next((col for col in ['track_name', 'track'] if col in df.columns), None),
                            hover_data=[next((col for col in ['artist_name', 'artist(s)_name'] if col in df.columns), None)],
                            color_continuous_scale='Viridis',
                            title=f'{x_display} vs {y_display} by {popularity_col.capitalize()}')
            
            st.plotly_chart(fig4, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig5 = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        fig5.update_layout(height=700)
        st.plotly_chart(fig5, use_container_width=True)

def display_clustering_analysis(df):
    """Display clustering analysis."""
    st.header("Clustering Analysis")
    st.write("This analysis helps identify natural groupings of songs based on their audio features.")
    
    # Try to detect audio feature columns with both naming conventions
    potential_features = {
        "danceability": ["danceability_%", "danceability"],
        "energy": ["energy_%", "energy"],
        "acousticness": ["acousticness_%", "acousticness"],
        "instrumentalness": ["instrumentalness_%", "instrumentalness"],
        "liveness": ["liveness_%", "liveness"],
        "valence": ["valence_%", "valence"],
        "speechiness": ["speechiness_%", "speechiness"],
        "tempo": ["bpm", "tempo"]
    }
    
    # Find which features are available in the dataset
    features_for_clustering = []
    for feature_name, possible_cols in potential_features.items():
        for col in possible_cols:
            if col in df.columns:
                features_for_clustering.append(col)
                break
    
    if len(features_for_clustering) < 2:
        st.error("Not enough audio features found for clustering analysis.")
        return
    
    st.write(f"Using the following features for clustering: {', '.join(features_for_clustering)}")
    
    # Convert columns to numeric and handle missing values
    X = df[features_for_clustering].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values with column mean
    X = X.fillna(X.mean())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Let user select number of clusters
    n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=4)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to DataFrame
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_clustered['PCA1'] = pca_result[:, 0]
    df_clustered['PCA2'] = pca_result[:, 1]
    
    # Display cluster scatter plot
    track_col = next((col for col in ['track_name', 'track'] if col in df.columns), None)
    artist_col = next((col for col in ['artist_name', 'artist(s)_name'] if col in df.columns), None)
    
    hover_data = [col for col in [track_col, artist_col] if col is not None]
    
    fig = px.scatter(df_clustered, x='PCA1', y='PCA2', color='Cluster',
                   hover_data=hover_data,
                   title=f'Song Clusters Based on Audio Features (PCA)',
                   color_continuous_scale=px.colors.qualitative.G10)
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    cluster_features = df_clustered.groupby('Cluster')[features_for_clustering].mean()
    st.dataframe(cluster_features)
    
    # Show example songs from each cluster
    st.subheader("Example Songs from Each Cluster")
    
    selected_cluster = st.selectbox("Select a cluster to view example songs:", 
                                 range(n_clusters), format_func=lambda x: f"Cluster {x}")
    
    popularity_col = next((col for col in ['streams', 'popularity'] if col in df.columns), None)
    display_cols = [col for col in [track_col, artist_col, popularity_col] if col is not None]
    
    cluster_songs = df_clustered[df_clustered['Cluster'] == selected_cluster]
    if popularity_col:
        cluster_songs = cluster_songs.sort_values(popularity_col, ascending=False)
    
    st.dataframe(cluster_songs[display_cols].head(10))

if __name__ == "__main__":
    run_dashboard() 