import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
from IPython.display import HTML
from flask import Flask, render_template, request, redirect, session
from metode.sistemRekomendasi import *
from metode.matrikEvaluasi import *

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(2)

# path = ''  # local
path = '/home/farsulhaq/mbkm'  # hosting

names = ['user_id', 'item_id', 'rating', 'timestime']
columns_movies = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown", "action", "adventure", "animation", "children's",
           "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
#  load dataset moviedata
movie_data = pd.read_csv(os.path.join(path, 'datasets/ml-100k', 'u.item'),
                         sep='|', names=columns_movies, encoding="latin-1", index_col="movie_id")
# load dataset
ratings_train_old = pd.read_csv(os.path.join(
    path, 'datasets/ml-100k', 'u1.base'), sep='\t', names=names)
ratings_test = pd.read_csv(os.path.join(
    path, 'datasets/ml-100k', 'u1.test'), sep='\t', names=names)
# rating dummy
rating_matrix = pd.DataFrame(np.zeros((943, 1682)), index=list(range(
    1, 944)), columns=list(range(1, 1683))).rename_axis(index='user_id', columns="item_id")
# dataset to pivot
rating_matrix_old = ratings_train_old.pivot_table(
    index='user_id', columns='item_id', values='rating')
rating_matrix_old = rating_matrix_old.fillna(0)
# update rating dummy
rating_matrix.update(rating_matrix_old)

# ==================================Parameter================================ #
nameMetode = {
    "pcc": "Pearson Correlation Coefficient (PCC)",
    "ami": "Adjusted Mutual Information (AMI)"
}

baseMetode = {
    "pcc-pcc": {"tetanggaU": 10, "tetanggaI": 100, "r1": 0.3},
    "pcc-ami": {"tetanggaU": 10, "tetanggaI": 10, "r1": 0.3},
    "ami-pcc": {"tetanggaU": 10, "tetanggaI": 130, "r1": 0.0},
    "ami-ami": {"tetanggaU": 10, "tetanggaI": 120, "r1": 0.0},
}

banyak_users = np.unique(ratings_test["user_id"]).tolist()
data_user = rating_matrix.to_numpy()


@app.route("/")
def index_page():
    navnya = ["Home", "Metode", "Tentang Aplikasi"]
    judulnya = "Rekomendasi System"
    nama_user = "Selamat datang di"
    return render_template("index.html", navnya=navnya, judulnya=judulnya)


@app.route("/metode")
def metode_page():
    # metod=(request.args.get('metode'))
    metod_user = (request.args.get('user-based'))
    metod_item = (request.args.get('item-based'))

    # User
    metode_usernya = nameMetode[metod_user.lower()]

    # Item
    metode_itemnya = nameMetode[metod_item.lower()]

    navnya = ["Home", "Metode", ""]
    judulnya = "Rekomendasi System"
    nama_user = "Selamat datang di"

    banyak_n = list(range(1, 51))

    pesan_error = ""
    return render_template("metode.html", navnya=navnya, metode_usernya=metode_usernya, metode_itemnya=metode_itemnya, metod_user=metod_user, metod_item=metod_item, judulnya=judulnya, nama_user=nama_user, banyak_user=banyak_users, banyak_n=banyak_n, pesan_error=pesan_error)


@app.route("/rekomendasi")
def rekomendasi_page():
    navnya = ["Home", " Hasil Rekomendasi Film", ""]
    judulnya = "Hasil Rekomendasi"
    id_user = int(request.args.get('user'))
    tetangga = int(request.args.get("tetangga"))
    metod_user = (request.args.get('metod_user'))
    metod_item = (request.args.get('metod_item'))

    if id_user not in banyak_users:
        return redirect("/")
    # User
    metode_usernya = nameMetode[metod_user.lower()]
    mean_user_df, mean_centered_user_df, similarity_user_df = joblib.load(
        os.path.join(path, 'model', metod_user.lower(), 'user_k1.joblib'))

    # Item
    metode_itemnya = nameMetode[metod_item.lower()]
    mean_item_df, mean_centered_item_df, similarity_item_df = joblib.load(
        os.path.join(path, 'model', metod_item.lower(), 'item_k1.joblib'))
    # tetanggaU
    tetanggaU = baseMetode[metod_user.lower(
    )+'-'+metod_item.lower()]['tetanggaU']
    # tetanggaI
    tetanggaI = baseMetode[metod_user.lower(
    )+'-'+metod_item.lower()]['tetanggaI']
    # r1
    r1 = baseMetode[metod_user.lower()+'-'+metod_item.lower()]['r1']
    # rekomendasi
    sr = sistemRekomendasi(rating_matrix, mean_user_df, mean_centered_user_df, similarity_user_df, mean_item_df,
                           mean_centered_item_df, similarity_item_df, user=id_user, tetanggaU=tetanggaU, tetanggaI=tetanggaI, r1=r1)

    gt = ratings_test[ratings_test['user_id']
                      == id_user].loc[:, 'item_id'].tolist()
    topN = sr[(-sr[:, 1].astype('float')).argsort()][:, 0][:tetangga]

    nameTopN = movie_data.loc[(topN), 'movie_title'].tolist()

    # datatrain
    dataTrain = movie_data.loc[rating_matrix.loc[id_user, :]
                               != 0.0, 'movie_title'].tolist()
    # datagroundtruth
    dataGroundTruth = movie_data.loc[gt, 'movie_title'].tolist()
    # datairisan
    dataIrisan = np.intersect1d(nameTopN, dataGroundTruth).tolist()
    print(dataIrisan)
    ePrecision = round(
        precision(ground_truth=gt, topN=topN, n=tetangga), 4)
    eRecall = round(recall(ground_truth=gt, topN=topN, n=tetangga), 4)
    eFscore = round(f1Score(ground_truth=gt, topN=topN, n=tetangga), 4)
    eDcg = round(dcg(ground_truth=gt, topN=topN, n=tetangga), 4)
    eNdcg = round(ndcg(ground_truth=gt, topN=topN, n=tetangga), 4)

    session['rekomendasi'] = {"user": id_user, "dataTrain": dataTrain, "nameTopN": nameTopN, "dataGroundTruth": dataGroundTruth,
                              "dataIrisan": dataIrisan, "ePrecision": ePrecision, "eRecall": eRecall, "eFscore": eFscore, "eDcg": eDcg, "eNdcg": eNdcg}

    return render_template("hasilrekomendasi.html", metod_user=metod_user, metod_item=metod_item, navnya=navnya, judulnya=judulnya, user=id_user, tetangga=tetangga, banyak_data_rekomendasi=tetangga, top_n=nameTopN)


@app.route("/metrik_evaluasi")
def metrik_evaluasi_page():
    navnya = ["Home", "", "Metrik Evaluasi"]
    judulnya = "Hasil Rekomendasi"
    hasilRekomendasi = session.get('rekomendasi')
    if hasilRekomendasi is None:
        # @app.route("/")
        return redirect("/")
    return render_template("metrik_evaluasi.html", navnya=navnya, judulnya=judulnya, user=hasilRekomendasi["user"], data_train=hasilRekomendasi["dataTrain"], data_irisan=hasilRekomendasi["dataIrisan"], data_ground_truth=hasilRekomendasi["dataGroundTruth"], top_n=hasilRekomendasi["nameTopN"], precision=hasilRekomendasi["ePrecision"], recall=hasilRekomendasi["eRecall"], f1=hasilRekomendasi["eFscore"], dcg=hasilRekomendasi["eDcg"], ndcg=hasilRekomendasi["eNdcg"])


if __name__ == "__main__":
    app.run(use_reloader=True)
