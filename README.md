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



# Research Questions
1. What is the most popular type of music?
2. Based on user preferences, can we predict other songs that users may enjoy?

# The Dataset
The Spotify dataset that we chose contains songs from the years 1921-2020. We chose to focus on a subset of the data, usings songs from the years 2010-2020 in order to relieve tension on the KMeans clustering algorithm. The Spotify API has a large variety of features documented for each track, such as the key signature, liveliness, mode, etc. Only numerical values were kept for the algorithm to consider. Listed below are the final audio features that were kept in the dataset for clustering.

Audio Features:
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
The following steps were taken to prepare the data for the KMeans clustering algorithm:

```python
#Selects all records from 2010 to 2020.
ten_yr_df=music_data_df.loc[(music_data_df['year']>=2010) & (music_data_df['year']<=2020)]
ten_yr_df

output:
False    21656
dtype: int64
```


```python
#Dropping explicit and mode columns
ten_yr_df = ten_yr_df.drop(['explicit', 'mode', 'key', "release_date", "popularity", "name", "year","artists"], axis=1, inplace=False)
ten_yr_df.head(3)
```

```python
#Checks to see if any records are duplicated
ten_yr_df.duplicated().value_counts()
```

```python
#Checks to see if any values are missing
ten_yr_df.isnull().sum()
```


![image](https://user-images.githubusercontent.com/115199874/229994975-89a45f07-92a1-442a-ba27-436cb222a385.png)

The Application

What does the graph tell us ?
- Loudness and Tempo musics evolved in the same way
- Energy and Valence are pretty similar 
- Danceability and Speechiness are more or less close

## Correlation 
In order to go further in the analysis, we did a correlation visualization :

![image](https://user-images.githubusercontent.com/115199874/229678609-c3907610-5d72-40ae-a0ef-6c4abcaead95.png)

It seems like Energy and Loudness are highly positively correlated.
Also, Valence is positively correlated with Danceability and energy. Considering happy songs make people energetic and want to dance, the correlation make a lot sense. *Interestingly, speechiness and loudness are negatively correlated with each other.

### In-depth analysis

![image](https://user-images.githubusercontent.com/115199874/229679875-efb4de27-8ab9-48aa-aa72-becf93dc1271.png)

From the datas above, here a visualization how Energy and Loudness are positively correlated.

## What artists do people listen to?

![image](https://user-images.githubusercontent.com/115199874/229963812-4f369422-6037-4c8f-a15b-12bfb25e4aa0.png)

## What kind of music do people listen to ?

![image](https://user-images.githubusercontent.com/115199874/229966141-ca8f2e5d-3ecd-4e37-acb1-4bd38847912f.png)


# Data Sources 
[List of Songs on Spotify 1921-2020](https://www.kaggle.com/datasets/ektanegi/spotifydata-19212020?resource=download)

[Spotify Web API](https://developer.spotify.com/documentation/web-api)

[Spotipy Documentation](https://spotipy.readthedocs.io/en/2.22.1/)
