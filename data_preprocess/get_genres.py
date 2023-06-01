# +---------------------+
# | Importing Depencies |
# +---------------------+
import os
import requests
import spotipy
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.spatial
import time

#need to be running on python 3.11 for these
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

# +-----------------+
# | Reading in data |
# +-----------------+
path = Path() / "Data" / "tenyear_cleaned_for_kmeans.csv" 
path_id_genre = Path() / "Data" / "id_genres.csv"
tracks_df = pd.read_csv(path)
column_names = tracks_df.columns
index = tracks_df.index

#isolating track_ids as a list
track_ids = tracks_df["id"].values.tolist()

def get_genres(track_ids):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    genres_list = []
    track_number = 0

    for id in track_ids:
        search = sp.track(id)
        artist_id = search["album"]["artists"][0]["id"]

        artist_search = sp.artist(artist_id)
        genres = artist_search['genres']

        genres_list.append(genres)
        print(f"{track_number} -- Genres for track {id} successfully added!")
        track_number +=1


    return genres_list


def main(): 
    #Setting up environment variables for spotipy API
    os.environ['SPOTIPY_CLIENT_ID'] = "d8c9d086b1714bbaad9b9d1448817add"
    os.environ['SPOTIPY_CLIENT_SECRET'] = "e6b7f13670bc4fec9b1b591acaa3eb8b"
    os.environ['SPOTIPY_REDIRECT_URI'] = "https://localhost:8888/callback"

    #requests an authorization token and saves it as a variable: the SpotifyOAuth accepts token automatically.
    response = requests.post(url="https://accounts.spotify.com/api/token"
                                ,data="grant_type=client_credentials&client_id=d8c9d086b1714bbaad9b9d1448817add&client_secret=e6b7f13670bc4fec9b1b591acaa3eb8b"
                                ,headers={"Content-Type": "application/x-www-form-urlencoded"})

    response_json = response.json()
    access_token = response_json['access_token']

    start_time = time.time()
    genres_list = get_genres(track_ids)

    #Create a dataframe with track_id and genres_list  
    id_genres_df = pd.DataFrame({"id": track_ids, "genres": genres_list})
    id_genres_df.to_csv(path_id_genre)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()