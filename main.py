from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
import requests
import random
import re
import pickle
import pandas as pd

from dotenv import load_dotenv
import os

# laddar in tränade AI-modeller och data
vectorizer = pickle.load(open("models/vectorizer_west.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model_west.pkl", "rb"))
df = pd.read_pickle("models/netflix_data_west.pkl")

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

Bootstrap5(app)
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Populära filmer vid dålig knn-match
POPULAR_MOVIES = [
    'Avengers: Infinity War', 'Mystery Lab', 'Click', 'Dallas Buyers Club', 'Grown Ups',
    'Thomas & Friends: Marvelous Machinery: World of Tomorrow', 'LEGO Marvel Super Heroes: Black Panther',
    'Innocent', 'House of Cards', 'ADAM SANDLER 100% FRESH', 'Revolutionary Road',
    'LEGO Marvel Super Heroes: Avengers Reassembled!', 'Django Unchained',
    'Batman: The Killing Joke', 'Better Call Saul', 'Marvel Anime: Wolverine', "Lion's Heart",
    'The Road to El Camino: Behind the Scenes of El Camino: A Breaking Bad Movie', 'Peaky Blinders', "The Dark Knight",
    "The Matrix", "Interstellar", "Parasite"
]

# simpel text-preprocessor som gör små bokstäver och tar bort specialtecken
def preprocess_query(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


# Tar user-query och returnerar top_n rekommenderade filmer
def recommend_movies(user_query, top_n=5):
    query_vector = vectorizer.transform([user_query]) # Vektoriserar texten
    distances, indices = knn_model.kneighbors(query_vector, n_neighbors=top_n) # hämtar near neighbors

    recommendations = df.iloc[indices[0]].copy()
    recommendations["similarity"] = 1 - distances[0]  # omvandlar distans till likhet
    return recommendations[["title", "similarity"]]



# APIar OMDb för filmrekommendationers poster
def get_movie_poster(title, api_key):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            print(f"OMDb API error ({response.status_code}): {response.text}")
            return "/static/content/default.png"

        data = response.json()
        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
            return data.get('Poster') 
        else:
            print(f"OMDb: No poster found for '{title}'")
            return "/static/content/default.png" # Fallback bild om poster inte hittas

    except Exception as e:
        print(f"Error fetching poster for '{title}': {e}")
        return "/static/content/default.png"


@app.route("/", methods=["GET", "POST"])
def home():
    movie_data = []
    user_query = ""
    message = ""

    if request.method == "POST":
        user_query = request.form.get("query", "")
        recommendations = recommend_movies(user_query)

        # Fallback vid dålig matchning, visar 5 random från manuellt utvalda populära filmer
        if recommendations["similarity"].mean() < 0.1 or recommendations["title"].equals(df.tail(5)["title"]):
            message = "We couldn’t understand your search, but here are 5 recommendations!"
            fallback_titles = random.sample(POPULAR_MOVIES, 5)
            for title in fallback_titles:
                poster_url = get_movie_poster(title, OMDB_API_KEY)
                movie_data.append({
                    "title": title,
                    "poster_url": poster_url
                })
        else:
            
            # Visar riktiga rekommendationer med posters
            for title in recommendations["title"]:
                poster_url = get_movie_poster(title, OMDB_API_KEY)
                movie_data.append({
                    "title": title,
                    "poster_url": poster_url
                })

    return render_template("index.html", movie_data=movie_data, user_query=user_query, message=message)



if __name__ == '__main__':
    app.run(debug=True)
