import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load the data from the CSV file into a pandas dataframe
movies_data = pd.read_csv('/content/movies.csv')

# Select relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace null values with an empty string for selected features
movies_data[selected_features] = movies_data[selected_features].fillna('')

# Combine all selected features into a single column
movies_data['combined_features'] = movies_data[selected_features].agg(' '.join, axis=1)

# Convert text data to feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Get similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# Get the movie name from the user
movie_name = input('Enter your favorite movie name: ')

# Find the closest match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, movies_data['title'])
close_match = find_close_match[0]

# Find the index of the movie with the title
index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]

# Get a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))

# Sort the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# Print the names of similar movies based on the index
print('Movies suggested for you:\n')

for i, movie in enumerate(sorted_similar_movies[:30], 1):
    index = movie[0]
    title_from_index = movies_data.loc[index, 'title']
    print(i, '.', title_from_index)
