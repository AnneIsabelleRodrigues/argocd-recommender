from datetime import datetime
import pandas as pd
from flask import jsonify
from flask import Flask
from flask import request
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)

@app.route('/api/recommend', methods=['POST'])
def recommend():

    songs = request.get_json(force=True)

    image_tag = os.environ.get('dev-image-tag')

    try:
        model = pd.read_pickle("/modelo/rules.pkl")
        data = pd.read_pickle("/modelo/data.pkl")
    except FileNotFoundError as e:
        return jsonify({"error": "Arquivo não encontrado"}), 404
    except Exception as e:
        return jsonify({"error": f"Erro desconhecido: {e}"}), 500
    
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

    return jsonify({
        "model_date": date,
        "playlist_ids": pids,
        "version": image_tag
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=32168)
