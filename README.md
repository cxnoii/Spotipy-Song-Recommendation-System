# Project Overview

Members:
* Raphael Tran
* David Dixon
* Nicholas Dao
* Patricia Roa
 
The purpose of this project is to use an unsupervised learning model to create an application to recommend songs that are similar to the song the user inputs. A KMeans clustering algorithm will be used in order to cluster songs based on audio features that will be obtained from the Spotify API. 

<p align="center">
<img src="https://github.com/cxnoii/Spotipy-Song-Recommendation-System/assets/114107454/00834c05-4068-4f1b-9d16-f4915cc4d76d" width="350" height="200">
</p>

# Background

Song and playlist prediction using machine learning is an emerging field that aims to provide personalized music recommendations to users based on their listening habits and preferences. The idea behind this technology is to analyze large amounts of data on a user's listening history, such as the songs they have played, skipped, or added to their playlists, and use this information to predict the types of songs and playlists that they will enjoy in the future.

The development of this technology is made possible by advances in machine learning and data analysis techniques. The algorithms used to predict song and playlist preferences rely on a variety of factors, including the user's age, gender, location, and musical tastes, as well as data on the popularity and characteristics of different songs and artists.

One of the key challenges in developing accurate song and playlist prediction models is the sheer volume and complexity of the data involved. With millions of songs and countless variations in musical style and genre, it can be difficult to identify patterns and trends that accurately reflect individual user preferences. To overcome this challenge, machine learning algorithms are trained on large datasets that contain vast amounts of information on user listening habits, as well as on the characteristics of different songs and artists.

Once a machine learning model has been trained, it can be used to predict the types of songs and playlists that a user is most likely to enjoy. This information can then be used to generate personalized recommendations, which can be delivered to the user through a variety of channels, such as music streaming services, mobile apps, or social media platforms.

Overall, the development of song and playlist prediction using machine learning has the potential to revolutionize the way that people discover and enjoy new music. By providing personalized recommendations based on individual listening habits and preferences, this technology can help users to discover new artists and genres, and to create playlists that reflect their unique musical tastes and moods.


# The Dataset
The Spotify dataset that we chose contains songs from the years 1921-2020. We chose to focus on a subset of the data, usings songs from the years 2010-2020 in order to relieve tension on the KMeans clustering algorithm. The Spotify API has a large variety of features documented for each track, such as the key signature, liveliness, mode, etc. Only numerical values were kept for the algorithm to consider. Listed below are the final audio features that were kept in the dataset for clustering. It should be noted that KMeans considers two variables and the two that were chosen to focus on are _danceability_ and _energy_. Recommendations vary based on the two variables chosen for clustering.

__Audio Features__:
- _id_ : Unique identifier for each track.
- _Acousticness_ : Describes how acoustic a song is. A score of 1.0 means the song is most likely to be an acoustic one.
- _Danceability_ : Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- _Duration_: Length of the song in milliseconds (ms).
- _Energy_ : Represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.
- _Instrumentalness_ : Represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is.
- _Liveness_ : Describes the probability that the song was recorded with a live audience. According to the official documentation “a value above 0.8 provides strong likelihood that the track is live”.
- _Loudness_ : Refers to how loud or soft a sound seems to a listener. The loudness of sound is determined, in turn, by the intensity of the sound waves.
- _Speechiness_ : Detects the presence of spoken words in a track. If the speechiness of a song is above 0.66, it is probably made of spoken words, a score between 0.33 and 0.66 is a song that may contain both music and words, and a score below 0.33 means the song does not have any speech.
- _Tempo_ : The overall estimated tempo of a track in beats per minute (BPM).
- _Valence_ : Spotify uses the word “valence” to measure whether a song is likely to make someone feel happy (higher valence) or sad (lower valence).

# Data Preprocessing
The following steps were taken to prepare the data for the KMeans clustering algorithm.

```python
#Selects all records from 2010 to 2020.
ten_yr_df=music_data_df.loc[(music_data_df['year']>=2010) & (music_data_df['year']<=2020)]
ten_yr_df
```


```python
#Dropping all non-numerical, boolean, and variables that are ill suited for project need.
ten_yr_df = ten_yr_df.drop(['explicit', 'mode', 'key', "release_date", "popularity", "name", "year","artists"], axis=1, inplace=False)
ten_yr_df.head(3)
```

After isolating the target audio features, the new dataset was checked for duplicate records or missing values to prevent any errors.
```python
#Checks to see if any records are duplicated
ten_yr_df.duplicated().value_counts()

out:
False    21656
dtype: int64
```

```python
#Checks to see if any values are missing
ten_yr_df.isnull().sum()

out:
acousticness        0
danceability        0
duration_ms         0
energy              0
id                  0
instrumentalness    0
liveness            0
loudness            0
speechiness         0
tempo               0
valence             0
dtype: int64
```


# KMeans Model
The objective of the KMeans model is to group similar data points together based on certain features. In this project, the audio features of a user's track was retrieved from the Spotify API and KMeans was performed with the dataset of songs from the years 2010-2020. Clusters are formed by randomly selecting a record as the centroid and as new records are introduced, the centroids of a given cluster adjust. The number of clusters that are generated is determined by k, which was optimized in this project using the elbow method.

### Calculating optimal number of clusters:

```python
# Creates an empty list to store the inertia values
k = list(range(1,12))
inertia = []

# Creates a for loop to compute the inertia with each possible value of k
for i in k:
    k_model = KMeans(n_clusters=i, random_state=42, n_init="auto")
    k_model.fit(tracks_df_scaled)
    inertia.append(k_model.inertia_)

# Creates a dictionary with the data to plot the Elbow curve
elbow_dict = {"k": k, "inertia": inertia}

# Creates a DataFrame with the data to plot the Elbow curve
elbow_df = pd.DataFrame(elbow_dict)
elbow_df.head()

#Plots k vs. inertia to visualize the elbow
elbow = elbow_df.hvplot.line(
    x='k',
    y='inertia',
    xticks=k,
    title='Elbow Curve using Original Scaled Data')
elbow
```
<p align="center">
<img src="https://github.com/cxnoii/Spotipy-Song-Recommendation-System/assets/114107454/aa05d194-a1f3-4a9c-85cc-139964d7d8ed" width=600>
</p>

The elbow method uses inertia values to plot against possible values of k. Inertia is the sum of squared distances between each data point and the centroid; this is an indicator of how well a dataset was clustered by K-Means. The optimal number for k is where the inertia values begin to flatten out, the elbow. In this project, a value of k=4 was used in the K-Means algorithm. Python's SciKit-Learn package was used in order to calculate the inertia values where it was plotted to view the elbow. 

## Preparing the data for KMeans
The application prompts you to enter the name of a song, followed by the artist's name. Before we begin clustering, our target audio features for this song must be retrieved. These features are retrieved from the Spotify API where it is then concatenated with our existing dataset of songs. The name and artists values are located at a separate endpoint from the audio features, so the track_id is retrieved in a preceeding function, where it can then be used to obtain the following:

```python
#accepts the track_id and returns our target features
def get_track_features(track_id):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    feature_search = sp.audio_features(tracks=track_id)
    acousticness = feature_search[0]['acousticness']
    danceability = feature_search[0]['danceability']
    duration_ms = feature_search[0]['duration_ms']
    energy = feature_search[0]['energy']
    instrumentalness = feature_search[0]['instrumentalness']
    liveness = feature_search[0]['liveness']
    loudness = feature_search[0]['loudness']
    speechiness = feature_search[0]['speechiness']
    tempo = feature_search[0]['tempo']
    valence = feature_search[0]['valence']
    features = {"id": track_id, "acousticness":acousticness, "danceability":danceability, "duration_ms":duration_ms
                ,"energy":energy, "instrumentalness":instrumentalness, "liveness":liveness, "loudness":loudness
                , "speechiness":speechiness, "tempo":tempo, "valence":valence}
    
    return features
```

Once the dataframes are combined, KMeans can be performed. The data is first standardized by appplying a standard scalar in order to increase the performance of the model. This ensures that our model is not too heavily skewed by any potential outliers present in the data. After initializing the algorithm, each song is assigned to a cluster group based on the model's predictions, and the function will return a dataframe of tracks in its respective cluster group.

```python
#runs kmeans with user song in it
def run_kmeans(tracks_df_w_new_song, track_id):
    #preprocessing dataframe
    column_names = tracks_df_w_new_song.columns
    index = tracks_df_w_new_song.index
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(tracks_df_w_new_song)
    tracks_df_scaled = pd.DataFrame(scaled_data, columns=column_names, index=index)

    #running kmeans algorithm
    k_model = KMeans(n_clusters=4, random_state=1)
    predictions = k_model.fit_predict(tracks_df_scaled)
    tracks_df_predictions = tracks_df_scaled.copy()
    tracks_df_predictions["ClusterGroup"] = predictions

    #now isolates data points by the cluster group of the user's song
    user_song = tracks_df_predictions.loc[track_id]
    cluster_group_number = int(user_song['ClusterGroup'])
    user_song_cluster_group = tracks_df_predictions[tracks_df_predictions['ClusterGroup'] == cluster_group_number]

    return user_song_cluster_group
```

## Demo
insert[screen recording]

# Data Sources 
[List of Songs on Spotify 1921-2020](https://www.kaggle.com/datasets/ektanegi/spotifydata-19212020?resource=download)

[Spotify Web API](https://developer.spotify.com/documentation/web-api)

[SpotiPy Documentation](https://spotipy.readthedocs.io/en/2.22.1/)
