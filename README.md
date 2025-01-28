# Movie-ML

This is a **Movie Recommender System** built with Python, Streamlit, and machine learning techniques. It uses movie data to provide personalized recommendations based on a user-input movie title. The application also fetches movie posters and trailers using external APIs.

## Features
- Recommends 10 similar movies based on the input movie title.
- Displays movie posters and links to trailers.
- Utilizes **fuzzy matching** to find the closest movie title match.
- Supports light and dark themes for the user interface.

## Technologies Used
- **Python**
- **Streamlit** for the web application interface
- **pandas** and **NumPy** for data manipulation
- **scikit-learn** for machine learning (CountVectorizer and Cosine Similarity)
- **NLTK** for text stemming
- **FuzzyWuzzy** for title matching
- **Google API** for fetching YouTube trailers
- **The Movie Database (TMDb) API** for movie posters

## Dataset
The app uses the [TMDb 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), consisting of two CSV files:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`
