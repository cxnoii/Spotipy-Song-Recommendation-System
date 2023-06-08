from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import spotipyxx

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/KMeans.html")
def Kmeans():
    return render_template("KMeans.html")

@app.route("/modelpage.html", methods=["POST", "GET"])
def modelpage():
    return render_template("modelpage.html")

#After pressing submit with song, this url runs the spotipyxx script to return song names and album urls
@app.route("/recommend_songs", methods=['POST'])
def process_input():
    input_data = request.form['input']
    recommended_songs, album_urls = spotipyxx.recommend_songs(input_data)

    #Creates a list of tuples with recommended songs and album urls for ease of tag creations (see modelpage.html)
    song_with_url = list(zip(recommended_songs, album_urls))
    
    return render_template("/modelpage.html", results=song_with_url)

@app.route("/sources.html")
def sources():
    return render_template("sources.html")


if __name__ == '__main__':
    app.run()

