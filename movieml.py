import pandas as pd
import numpy as np
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as strlit
import nltk
from fuzzywuzzy import process
from googleapiclient.discovery import build

# Function to convert stringified data to list
def convert(x):

    list = []

    for i in ast.literal_eval(x):
        list.append(i['name']) # Extracting the name of the field from the dictionary

    return list

# Function to convert stringified data to list and extract the first 3 cast members
def convertCast(x):

    list = []
    counter = 0

    for i in ast.literal_eval(x):
        if counter < 3:
            list.append(i['name']) # Extracting first 3 cast members from the dictionary
            counter += 1
    
    return list

# Function to extract the name of the director from the crew column
def getDirName(x):

    list = []

    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            list.append(i['name']) # Extracting the name of the director from the dictionary
            break

    return list

# Function to remove spaces from the string
def removeSpaces(x):

    list = []

    for i in x:
        list.append(i.replace(" ", "")) # Removing spaces from the string

    return list

# Function to stem the text
def stemmer(text):

    stemmer = nltk.PorterStemmer() # Create a stemmer object

    return [stemmer.stem(word) for word in text] # Return the stemmed words

# Function to fetch the poster URL of the movie
def fetchPosterURL(movie_id, apiKey):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={apiKey}"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Function to fetch the trailer URL of the movie
def fetchTrailerURL(movie_title, apiKey):

    youtube = build('youtube', 'v3', developerKey=apiKey)
    request = youtube.search().list(
        q=f"{movie_title} trailer",
        part='snippet',
        type='video',
        maxResults=1
    )
    response = request.execute()
    if response['items']:
        video_id = response['items'][0]['id']['videoId']
        return f"https://www.youtube.com/watch?v={video_id}"
    return None

# Step 1: Load and prepare the data
@strlit.cache_data
def loadData():

    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    # Merging the two dataframes on the 'title' column
    mergedData = movies.merge(credits, on='title')

    # Collecting the title, genres, id, overview, cast, crew, and keywords of the movie
    mergedData = mergedData[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    mergedData.dropna(inplace=True) # Drop rows with missing values

    # Now we need to convert the stringified columns to lists using the convert function
    mergedData['genres'] = mergedData['genres'].apply(convert) # Applying the convert function to the 'genres' column
    mergedData['keywords'] = mergedData['keywords'].apply(convert) # Applying the convert function to the 'keywords' column
    mergedData['cast'] = mergedData['cast'].apply(convertCast) # Applying the convert_cast function to the 'cast' column
    mergedData['crew'] = mergedData['crew'].apply(getDirName) # Applying the get_director_name function to the 'crew' column
    mergedData['overview'] = mergedData['overview'].apply(lambda x: x.split()) # Splitting the overview into a list of words

    # Now we need to remove spaces from the string using the remove_spaces function
    mergedData['cast'] = mergedData['cast'].apply(removeSpaces) # Applying the remove_spaces function to the 'cast' column
    mergedData['crew'] = mergedData['crew'].apply(removeSpaces) # Applying the remove_spaces function to the 'crew' column
    mergedData['genres'] = mergedData['genres'].apply(removeSpaces) # Applying the remove_spaces function to the 'genres' column
    mergedData['keywords'] = mergedData['keywords'].apply(removeSpaces) # Applying the remove_spaces function to the 'keywords' column

    # Now we need to create a new column 'tags' which will contain the combined values of the 'genres', 'keywords', 'cast', 'crew', and 'overview' columns
    mergedData['tags'] = mergedData['genres'] + mergedData['keywords'] + mergedData['cast'] + mergedData['crew'] + mergedData['overview']

    # Now we can remove the 'genres', 'keywords', 'cast', 'crew', and 'overview' columns since we have extracted the necessary information into the 'tags' column
    mergedData.drop(['genres', 'keywords', 'cast', 'crew', 'overview'], axis=1, inplace=True)
    mergedData['tags'] = mergedData['tags'].apply(lambda x: ' '.join(x)) # Joining the list of tags into a single string

    # Now we will convert all the text to lowercase
    mergedData['tags'] = mergedData['tags'].apply(lambda x: x.lower())
    # Now we will stem the text. This means that we will convert the words to their root form to reduce the dimensionality of the data and improve the performance of the model
    mergedData['tags'] = mergedData['tags'].apply(lambda x: ' '.join(stemmer(x.split()))) # Stemming the text

    return mergedData

# Step 2: Create a Count Vectorizer to convert the text data into a matrix of token counts
@strlit.cache_data
def createCountVectorizer(data):

    countVectorizer = CountVectorizer(stop_words='english') # Create a CountVectorizer object
    countMatrix = countVectorizer.fit_transform(data['tags']) # Fit and transform the data to get the count matrix

    return countMatrix

# Step 3: Compute the Cosine Similarity Matrix to get the similarity between movies
def computeCosineSimilarity(countMatrix):

    cosineSimilarity = cosine_similarity(countMatrix) # Compute the cosine similarity matrix to get the similarity between movies

    return cosineSimilarity

# Step 4: Get movie recommendations based on the cosine similarity matrix
def getRecommendations(title, data, cosineSimilarity, tmdbApiKey, youtubeApiKey):

    # Use fuzzy matching to find the closest match to the user's input
    closest_match = process.extractOne(title, data['title'].values)
    
    if closest_match is None:
        return ["Movie title not found in the dataset. Please try another title."]
    
    # Get the index of the movie with the closest matching title
    index = data[data['title'] == closest_match[0]].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    similarityScores = list(enumerate(cosineSimilarity[index]))

    # Sort the movies based on the similarity scores
    similarityScores = sorted(similarityScores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies
    topMovies = similarityScores[1:11]

    # Get the movie titles
    movieTitles = [data.iloc[movie[0]]['title'] for movie in topMovies]

    # Get the movie titles and poster URLs
    movieTitles = []
    posterURLs = []
    trailerURLs = []

    for movie in topMovies:
        movie_id = data.iloc[movie[0]]['movie_id']
        movie_title = data.iloc[movie[0]]['title']
        movieTitles.append(movie_title)
        posterURLs.append(fetchPosterURL(movie_id, tmdbApiKey))
        trailerURLs.append(fetchTrailerURL(movie_title, youtubeApiKey))

    return movieTitles, posterURLs, trailerURLs

# Step 5: Streamlit App
# Ensure session state for theme exists
if 'themebutton' not in strlit.session_state:
    strlit.session_state['themebutton'] = 'Dark'

# Sidebar radio box for theme selection
selected_theme = strlit.sidebar.radio("Select Theme", ['Light', 'Dark'])

# Update theme based on the radio box selection
if selected_theme == 'Light' and strlit.session_state['themebutton'] != 'Light':
    strlit._config.set_option(f'theme.base', "light")
    strlit._config.set_option(f'theme.backgroundColor', "white")
   # strlit._config.set_option(f'theme.primaryColor', "white") # Blue
   # strlit._config.set_option(f'theme.secondaryBackgroundColor', "black") # Light Blue
    strlit._config.set_option(f'theme.textColor', "black")
    strlit.session_state['themebutton'] = 'Light'
    strlit.rerun()

elif selected_theme == 'Dark' and strlit.session_state['themebutton'] != 'Dark':
    strlit._config.set_option(f'theme.base', "dark")
    strlit._config.set_option(f'theme.backgroundColor', "black")
  #  strlit._config.set_option(f'theme.primaryColor', "#c98bdb") # Purple
  #  strlit._config.set_option(f'theme.secondaryBackgroundColor', "#1e1e1e") # Dark Grey
    strlit._config.set_option(f'theme.textColor', "white")
    strlit.session_state['themebutton'] = 'Dark'
    strlit.rerun()

strlit.title("Movie Recommender System")

# Load movie data and compute similarity matrix
movies = loadData()
countMatrix = createCountVectorizer(movies)
cosineSimilarity = computeCosineSimilarity(countMatrix)

# Collect user input
title = strlit.text_input("Enter the title of the movie:")

# API keys
tmdbApiKey = "ef3f0c3d6c83a3a1c9d27f94f6dde9ac"
youtubeApiKey = "AIzaSyB3f7sELh9ASvDQ_VI0bYEq5nsAjgjWAxQ"

# Display recommendations when the button is clicked
if strlit.button("Get Personalized Recommendations"):
    if title:
        with strlit.spinner('Fetching recommendations...'):
            recommendations, posterURLs, trailerURLs = getRecommendations(title, movies, cosineSimilarity, tmdbApiKey, youtubeApiKey)
        strlit.write("Recommended Movies:")
        cols = strlit.columns(5)  # Create 5 columns
        for i, (movie, poster, trailer) in enumerate(zip(recommendations, posterURLs, trailerURLs)):
            with cols[i % 5]:  # Cycle through columns
                strlit.markdown(f"[{movie}]({trailer})")
                if poster:
                    strlit.image(poster, width=150)
                else:
                    strlit.write("Poster not available")
    else:
        strlit.write("Please enter a movie title to get recommendations.")