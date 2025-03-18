#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify Data Analysis
--------------------
This script analyzes the Spotify dataset to identify trends and patterns in music.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
sns.set_theme(style="whitegrid")

def load_data(file_path):
    """Load the Spotify dataset and return a pandas DataFrame."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Trying with 'latin1' encoding...")
        try:
            return pd.read_csv(file_path, encoding='latin1')
        except UnicodeDecodeError:
            print("Latin1 decoding also failed. Trying with 'cp1252' encoding...")
            try:
                return pd.read_csv(file_path, encoding='cp1252')
            except Exception as e:
                print("Failed to read the CSV file with multiple encodings. Error details:")
                print(e)
                raise

def explore_data(df):
    """Perform initial data exploration."""
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def clean_data(df):
    """Clean and preprocess the dataset."""
    numeric_columns_with_commas = ['streams', 'in_deezer_playlists', 'in_shazam_charts']

    for col in numeric_columns_with_commas:
        if col in df.columns:
            print(f"\nCleaning column: {col}")
            print(df[col].head(5))

            # Handle comma-separated numbers
            invalid_rows = df[~df[col].astype(str).str.replace(',', '').str.replace('.', '').str.isnumeric()]
            if not invalid_rows.empty:
                print(f"\nDropping {len(invalid_rows)} invalid rows from column '{col}'")
                df = df.drop(invalid_rows.index)

            df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

    # Identify numeric feature columns
    numeric_features = [
        'danceability_%', 'energy_%', 'acousticness_%', 'instrumentalness_%',
        'liveness_%', 'valence_%', 'speechiness_%', 'bpm', 'streams'
    ]
    
    # Only keep columns that exist in the dataframe
    numeric_features = [col for col in numeric_features if col in df.columns]
    
    if numeric_features:
        # Drop rows with missing values in important numeric columns
        df.dropna(subset=numeric_features, inplace=True)

        # Convert to numeric
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Standardize artist column name
    possible_artist_columns = ['artist', 'artists', 'artist_name', 'artist(s)_name']
    found_artist_column = None
    for col in possible_artist_columns:
        if col in df.columns:
            found_artist_column = col
            break

    if found_artist_column and found_artist_column != 'artist_name':
        df.rename(columns={found_artist_column: 'artist_name'}, inplace=True)
        print(f"\nRenamed '{found_artist_column}' to 'artist_name' for consistency.")

    print("Data cleaned successfully!")
    return df

def analyze_popularity(df):
    """Analyze popularity/streams and its relationship with audio features."""
    # Identify popularity column (streams or popularity)
    popularity_col = 'streams' if 'streams' in df.columns else 'popularity'
    
    if popularity_col not in df.columns:
        print("No popularity or streams column found. Skipping popularity analysis.")
        return
    
    # Identify available audio feature columns
    potential_features = [
        'danceability_%', 'energy_%', 'acousticness_%', 'instrumentalness_%',
        'liveness_%', 'valence_%', 'speechiness_%', 'bpm', 'tempo',
        'danceability', 'energy', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'speechiness'
    ]
    
    available_features = [col for col in potential_features if col in df.columns]
    
    if not available_features:
        print("No audio features found. Skipping correlation analysis.")
        return
    
    # Add popularity column to the features for correlation
    correlation_cols = available_features + [popularity_col]
    
    # Compute correlation matrix
    correlation_matrix = df[correlation_cols].corr()

    print(f"\nCorrelation with {popularity_col.capitalize()}:")
    print(correlation_matrix[popularity_col].sort_values(ascending=False))

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f"Correlation Matrix with {popularity_col.capitalize()}")
    plt.savefig(f"images/correlation_matrix.png", bbox_inches="tight")
    plt.close()

def analyze_audio_features(df):
    """Analyze the distribution of audio features."""
    # Identify available audio feature columns
    potential_features = [
        'danceability_%', 'energy_%', 'acousticness_%', 'instrumentalness_%',
        'liveness_%', 'valence_%', 'speechiness_%',
        'danceability', 'energy', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'speechiness'
    ]
    
    audio_features = [col for col in potential_features if col in df.columns]
    
    if not audio_features:
        print("No audio features found. Skipping audio feature analysis.")
        return
        
    # Create subplot grid based on number of features
    n_features = len(audio_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    for i, feature in enumerate(audio_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig("images/audio_features_distribution.png", bbox_inches="tight")
    plt.close()

def genre_analysis(df):
    """Analyze genres in the dataset."""
    if "genre" in df.columns:
        # Number of songs per genre
        genre_counts = df["genre"].value_counts().head(10)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=genre_counts.values, y=genre_counts.index)
        plt.title("Top 10 Genres by Number of Songs", fontsize=15)
        plt.xlabel("Number of Songs")
        plt.savefig("images/top_genres.png", bbox_inches="tight")
        plt.close()
        
        # Find popularity column
        popularity_col = 'streams' if 'streams' in df.columns else 'popularity'
        if popularity_col in df.columns:
            # Average popularity by genre
            genre_popularity = df.groupby("genre")[popularity_col].mean().sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=genre_popularity.values, y=genre_popularity.index)
            plt.title(f"Top 10 Genres by Average {popularity_col.capitalize()}", fontsize=15)
            plt.xlabel(f"Average {popularity_col.capitalize()}")
            plt.savefig("images/genre_popularity.png", bbox_inches="tight")
            plt.close()
    else:
        print("No genre column found. Skipping genre analysis.")

def artist_analysis(df):
    """Analyze artists in the dataset."""
    # Check if artist column exists
    if "artist_name" not in df.columns:
        print("No artist_name column found. Skipping artist analysis.")
        return
        
    # Top artists by number of songs
    artist_counts = df["artist_name"].value_counts().head(10)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=artist_counts.values, y=artist_counts.index)
    plt.title("Top 10 Artists by Number of Songs", fontsize=15)
    plt.xlabel("Number of Songs")
    plt.savefig("images/top_artists_count.png", bbox_inches="tight")
    plt.close()

    # Find popularity column
    popularity_col = 'streams' if 'streams' in df.columns else 'popularity'
    if popularity_col in df.columns:
        # Top artists by average popularity/streams
        top_artists_popularity = df.groupby("artist_name")[popularity_col].mean().sort_values(ascending=False).head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_artists_popularity.values, y=top_artists_popularity.index)
        plt.title(f"Top 10 Artists by Average {popularity_col.capitalize()}", fontsize=15)
        plt.xlabel(f"Average {popularity_col.capitalize()}")
        plt.savefig("images/top_artists_popularity.png", bbox_inches="tight")
        plt.close()

def main():
    """Main function to execute the analysis."""
    # Create directories for data and images
    os.makedirs("data", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # Try both possible file names
    file_paths = ["data/spotify_2021.csv", "data/spotify_2023.csv"]
    
    df = None
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"Found dataset: {file_path}")
            df = load_data(file_path)
            break
    
    if df is None:
        print("Error: No dataset found. Please place either spotify_2021.csv or spotify_2023.csv in the data directory.")
        return
    
    # Load, explore and clean data
    df = explore_data(df)
    df = clean_data(df)
    
    print("\nColumns in DataFrame:", df.columns.tolist())

    # Run analyses
    analyze_popularity(df)
    analyze_audio_features(df)
    genre_analysis(df)
    artist_analysis(df)
    
    print("\nAnalysis completed! Check the 'images' directory for visualization results.")

if __name__ == "__main__":
    main() 