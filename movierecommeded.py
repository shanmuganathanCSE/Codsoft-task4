import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')
movies['content'] = movies['genre'] + " " + movies['description']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, cosine_sim=cosine_sim):
    if title not in movies['title'].values:
        return "Movie not found in database."

    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

movie_name = input("Enter a movie you like: ")
recommendations = recommend(movie_name)
print("\nRecommended Movies:")
for movie in recommendations:
    print(movie)
