from datetime import datetime
import pandas as pd
from flask import jsonify
from flask import Flask
from flask import request

app = Flask(__name__)

# class PlaylistBase(BaseModel):
#     model_date: str
#     playlist_ids: List[int]
#     version: str


# class Songs(BaseModel):
#     songs: List[str]

@app.route('/api/recommend', methods=['POST'])
def recommend():
    songs = request.get_json()

    model = pd.read_pickle("/modelo/rules.pkl")
    data = pd.read_pickle("/modelo/data.pkl")

    musics = []
    pids = []

    for song in songs['songs']:
        model['antecedents'].apply(lambda x: musics.append(x) if song in x else None)

    for setlist in musics:
        lset = list(setlist)
        for index, row in data.iterrows():
            if set(lset).issubset(set(row['track_name'])):
                pids.append(row['pid'])

    pids = list(set(pids))

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")

    return {
        "model_date": date,
        "playlist_ids": pids,
        "version": "1.0"
    }

if __name__ == '__main__':
    app.run(debug=True)
