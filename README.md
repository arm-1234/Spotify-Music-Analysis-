# Spotify Data Analysis

This project analyzes Spotify music datasets to identify trends, patterns, and interesting insights about music streaming.

## Project Overview

The analysis focuses on:
- Top songs and artists by popularity/streams
- Audio feature distributions and correlations
- Genre analysis (if available in dataset)
- Artist performance analysis

## Setup Instructions

1. Clone this repository
2. Place `spotify_2023.csv` in the `data/` directory
3. Install required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Analysis

You can run the analysis in two ways:

1. **Python Script:**
```bash
python spotify_analysis.py
```

2. **Interactive Dashboard:**
```bash
streamlit run spotify_dashboard.py
```

## Expected Dataset Structure

The analysis works with different Spotify dataset formats and automatically adapts to them. Common columns include:
- `track_name` or `track`: Name of the song
- `artist_name` or `artist(s)_name`: Name of the artist
- `streams` or `popularity`: Popularity metric
- Audio features with naming variations:
  - Percentage format: `danceability_%`, `energy_%`, etc.
  - Regular format: `danceability`, `energy`, etc.
- Optional: `genre`: Music genre

## Output

The script generates visualizations in the `images/` directory, including:
- Correlation heatmaps between popularity/streams and audio features
- Distribution plots for audio features
- Top artists and genres by various metrics

## Dashboard Features

The Streamlit dashboard provides:
- Interactive data exploration
- Song and artist rankings
- Audio feature analysis
- Music clustering by audio characteristics

## Requirements

- Python 3.6+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- scikit-learn 
