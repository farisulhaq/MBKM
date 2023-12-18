import os
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics import adjusted_mutual_info_score
from IPython.display import HTML
from flask import Flask, render_template, request
from metode.sistemRekomendasi import *
from metode.matrikEvaluasi import *

app = Flask(__name__)

# path = '' # local
path = '/home/farsulhaq/mbkm'  # hosting

names = ['user_id', 'item_id', 'rating', 'timestime']
columns = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown", "action", "adventure", "animation", "children's",
           "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
movie_data = pd.read_csv(os.path.join(path, 'datasets/ml-100k', 'u.item'),
                         sep='|', names=columns, encoding="latin-1", index_col="movie_id")
ratings_train_k1_old = pd.read_csv(os.path.join(
    path, 'datasets/ml-100k', 'u1.base'), sep='\t', names=names)
ratings_test_k1 = pd.read_csv(os.path.join(
    path, 'datasets/ml-100k', 'u1.test'), sep='\t', names=names)
banyak_users = np.unique(ratings_test_k1["user_id"])
rating_matrix_k1 = pd.DataFrame(np.zeros((943, 1682)), index=list(range(
    1, 944)), columns=list(range(1, 1683))).rename_axis(index='user_id', columns="item_id")
# train
rating_matrix_k1_old = ratings_train_k1_old.pivot_table(
    index='user_id', columns='item_id', values='rating')
rating_matrix_k1_old = rating_matrix_k1_old.fillna(0)
rating_matrix_k1.update(rating_matrix_k1_old)
# calculate contingency matrix
rating_matrix = pd.DataFrame(np.zeros((943, 1682)), index=list(range(
    1, 944)), columns=list(range(1, 1683))).rename_axis(index='user_id', columns="item_id")
rating_matrix_test = pd.DataFrame(np.zeros((943, 1682)), index=list(range(
    1, 944)), columns=list(range(1, 1683))).rename_axis(index='user_id', columns="item_id")

# load dataset k-fold, train dan test
ratings_train = pd.read_csv(os.path.join(
    path, f'datasets/ml-100k/u1.base'), sep='\t', names=names)
ratings_test = pd.read_csv(os.path.join(
    path, f'datasets/ml-100k/u1.test'), sep='\t', names=names)

# merubah dataset menjadi data pivot
rating_matrix_ = ratings_train.pivot_table(
    index='user_id', columns='item_id', values='rating').fillna(0)
rating_matrix_ = rating_matrix_.fillna(0)
# update data rating dummie
rating_matrix.update(rating_matrix_)
result_rating_matrix = rating_matrix.iloc[:5, :5]
# result_rating_matrix=rating_matrix
result_rating_matrix = HTML(result_rating_matrix.to_html(
    classes='table table-stripped fortable container'))
# result = rating_matrix.to_html()

# merubah test menjadi data pivot
rating_matrix_test_ = ratings_test.pivot_table(
    index='user_id', columns='item_id', values='rating').fillna(0)
rating_matrix_test_ = rating_matrix_test_.fillna(0)
rating_matrix_test.update(rating_matrix_test_)
result_rating_matrix_test = rating_matrix_test.iloc[:5, :5]
result_rating_matrix_test = HTML(result_rating_matrix_test.to_html(
    classes='table table-stripped fortable container'))
# result_rating_matrix=text_file.write(result)

# ===================================================================================================
# Item
# rating_matrix_T = rating_matrix.copy().T
# mean_user, mean_center_user, similarity_user = joblib.load(os.path.join(path, 'model', "pcc", 'item_k1.joblib'))
# item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
# item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
# item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)


# # ===================================================================================================
# # USER
# mean_user, mean_center_user, similarity_user = joblib.load(os.path.join(path, 'model', "pcc", 'user_k1.joblib'))
# mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
# mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
# similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)

# EVALUASI


data_user = rating_matrix.to_numpy()


@app.route("/")
def index_page():
    navnya = ["Home", "Metode", "Tentang Aplikasi"]
    judulnya = "Rekomendasi System"
    nama_user = "Selamat datang di"
    # films = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
    banyak_user = []
    for i in banyak_users:
        banyak_user.append(i)
    banyak_n = []
    for i in range(1, 51):
        banyak_n.append(i)
    pesan_error = ""
    return render_template("index.html", navnya=navnya, judulnya=judulnya, nama_user=nama_user, banyak_user=banyak_user, banyak_n=banyak_n, pesan_error=pesan_error)


@app.route("/metode")
def metode_page():
    # metod=(request.args.get('metode'))
    metod_user = (request.args.get('user-based'))
    metod_item = (request.args.get('item-based'))

    # User
    if metod_user == "PCC":
        metode_usernya = "Pearson Correlation Coefficient (PCC)"
        # User
        # with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        # mean_user, mean_center_user, similarity_user = joblib.load(os.path.join(path, 'model', f'user_k1_pcc.joblib'))
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    # elif metod_user=="ITR":
    #       metode_usernya = "Improved Triangle Similarity (ITR)"
    #       # User
    #       with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        # mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        # mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        # similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    elif metod_user == "AMI":
        metode_usernya = "Adjusted Mutual Information (AMI)"
        # User
        # with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "ami", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    else:
        metode_usernya = "Pearson Correlation Coefficient (PCC)"
        # User
        # with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)

    if metod_item == "PCC":
        metode_itemnya = "Pearson Correlation Coefficient (PCC)"
        # item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'item_k1.joblib'))
        # mean_item_df_k1, mean_centered_item_df_k1, similarity_item_df_k1 = joblib.load(os.path.join(path, 'model', 'item_k1_itr.joblib'))
        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    # elif metod_item=="ITR":
    #       metode_itemnya = "Improved Triangle Similarity (ITR)"
    #       # item
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       # mean_item_df_k1, mean_centered_item_df_k1, similarity_item_df_k1 = joblib.load(os.path.join(path, 'model', 'item_k1_itr.joblib'))
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    elif metod_item == "AMI":
        metode_itemnya = "Adjusted Mutual Information (AMI)"
        # item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        # item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(os.path.join(path, 'model', 'item_k1_ami.joblib'))
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', 'ami', 'item_k1_ami.joblib'))

        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.columns, columns=rating_matrix_T.columns)
    else:
        metode_itemnya = "Pearson Correlation Coefficient (PCC)"
        # Item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'item_k1.joblib'))
        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.columns, columns=rating_matrix_T.columns)

    # if metod == "itrxitr":
    #       metodenya = "Improved Triangle Similarity (ITR) dan Improved Triangle Similarity (ITR)"
    #       # Item
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, f'item_k11.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       # mean_item_df_k1, mean_centered_item_df_k1, similarity_item_df_k1 = joblib.load(os.path.join(path, 'model', f'item_k1_itr.joblib'))
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)

    #       # USER
    #       with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
    #       mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
    #       mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
    #       similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    # elif metod == "amixami":
    #       metodenya = "Adjusted Mutual Information (AMI) dan Adjusted Mutual Information (AMI)"
    #       # ITEM
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, f'item_k11.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)

    #       # USER
    #       with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
    #       mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
    #       mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
    #       similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    # elif metod == "itrxami":
    #       # ITEM
    #       metodenya = "Improved Triangle Similarity (ITR) dan Adjusted Mutual Information (AMI)"
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, f'item_k11.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)

    #       # USER
    #       with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
    #       mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
    #       mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
    #       similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    # elif metod == "amixami":
    #       # ITEM
    #       metodenya = "Adjusted Mutual Information (AMI) dan Adjusted Mutual Information (AMI)"
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, f'item_k1.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)

    #       # USER
    #       with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
    #       mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
    #       mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
    #       similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    # else:
    #       metodenya = "Improved Triangle Similarity (ITR) dan Adjusted Mutual Information (AMI)"
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, f'item_k11.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)

    #       # USER
    #       with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
    #       mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
    #       mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
    #       similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)

    navnya = ["Home", "Metode", ""]
    judulnya = "Rekomendasi System"
    nama_user = "Selamat datang di"
    # films = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
    banyak_user = []
    for i in banyak_users:
        banyak_user.append(i)
    banyak_n = []
    for i in range(1, 51):
        banyak_n.append(i)
    pesan_error = ""
    hasil_plot = [0.003111, 0.003374, 0.004131, 0.004485, 0.004531, 0.004702, 0.004730, 0.004701, 0.004936,
                  0.005132, 0.005473, 0.006124, 0.006475, 0.006746, 0.007581, 0.008290, 0.009290, 0.010341, 0.010815, 0.011359]
    xlabel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    return render_template("metode.html", navnya=navnya, metode_usernya=metode_usernya, metode_itemnya=metode_itemnya, metod_user=metod_user, metod_item=metod_item, judulnya=judulnya, nama_user=nama_user, banyak_user=banyak_user, banyak_n=banyak_n, pesan_error=pesan_error, hasil_plot=hasil_plot, xlabel=xlabel)


@app.route("/rekomendasi")
def rekomendasi_page():
    navnya = ["Home", " Hasil Rekomendasi Film", ""]
    judulnya = "Hasil Rekomendasi"
    id_user = int(request.args.get('user'))
    tetangga = int(request.args.get("tetangga"))
    metod_user = (request.args.get('metod_user'))
    metod_item = (request.args.get('metod_item'))

    # User
    if metod_user == "PCC":
        metode_usernya = "Pearson Correlation Coefficient (PCC)"
        # User
        # with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        # mean_user, mean_center_user, similarity_user = joblib.load(os.path.join(path, 'model', f'user_k1_pcc.joblib'))
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    # elif metod_user=="ITR":
    #       metode_usernya = "Improved Triangle Similarity (ITR)"
    #       # User
    #       with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        # mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        # mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        # similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    elif metod_user == "AMI":
        metode_usernya = "Adjusted Mutual Information (AMI)"
        # User
        # with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "ami", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    else:
        metode_usernya = "Pearson Correlation Coefficient (PCC)"
        # User
        # with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)

    if metod_item == "PCC":
        metode_itemnya = "Pearson Correlation Coefficient (PCC)"
        # item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'item_k1.joblib'))
        # mean_item_df_k1, mean_centered_item_df_k1, similarity_item_df_k1 = joblib.load(os.path.join(path, 'model', 'item_k1_itr.joblib'))
        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    # elif metod_item=="ITR":
    #       metode_itemnya = "Improved Triangle Similarity (ITR)"
    #       # item
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       # mean_item_df_k1, mean_centered_item_df_k1, similarity_item_df_k1 = joblib.load(os.path.join(path, 'model', 'item_k1_itr.joblib'))
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    elif metod_item == "AMI":
        metode_itemnya = "Adjusted Mutual Information (AMI)"
        # item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        # item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(os.path.join(path, 'model', 'item_k1_ami.joblib'))
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', 'ami', 'item_k1_ami.joblib'))

        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    else:
        metode_itemnya = "Pearson Correlation Coefficient (PCC)"
        # Item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'item_k1.joblib'))
        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    # print(id_user.type())
    # print(tetangga.type())
    # print("==========================="*2)
    # data_ground_truth = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5"]
    index_data_ground_truth = []
    # print("ini untuk i")
    asem = 0
    for i in range(1, len(rating_matrix_test.loc[id_user]+1)):
        if rating_matrix_test.loc[id_user][i] != 0.0:
            index_data_ground_truth.append(i)
    data_ground_truth = []
    for i in index_data_ground_truth:
        data_ground_truth.append(movie_data.loc[i][0])
    banyak_data_ground_truth = len(data_ground_truth)
    index_data_train = []
    # for i in range(len(rating_matrix[id_user])):
    for i in range(1, len(rating_matrix.loc[id_user]+1)):
        if rating_matrix.loc[id_user, i] != 0.0:
            index_data_train.append(i)
    data_train = []
    for i in index_data_train:
        data_train.append(movie_data.loc[i][0])
    # index_data_train=rating_matrix.iloc[id_user,:]
    banyak_data_train = len(data_train)

    if id_user not in banyak_users:
        # @app.route("/")
        navnya = ["Home", "Rekomendasi Film", "Tentang Aplikasi"]
        judulnya = "Rekomendasi System"
        nama_user = "Selamat datang di"
        films = ["ini film 1", "ini film 2", "ini film 3", "ini film 4", "ini film 5",
                 "ini film 6", "ini film 7", "ini film 8", "ini film 9", "ini film 10",]
        banyak_user = []
        for i in banyak_users:
            banyak_user.append(i)
        banyak_n = []
        for i in range(1, 51):
            banyak_n.append(i)
        pesan_error = "aktif"
        return render_template("index.html", navnya=navnya, judulnya=judulnya, nama_user=nama_user, banyak_user=banyak_user, banyak_n=banyak_n, pesan_error=pesan_error, id_user=id_user)

    top_n = []
    data_used_test = rating_matrix.loc[id_user].to_numpy()
    movie_norated_test = np.where(data_used_test == 0)[0]+1
    movie_norated_test.tolist()
    pred_user_datas = np.array(
        [
            # user,
            predict(
                rating_matrix,
                mean_user,
                mean_center_user,
                similarity_user,
                user=id_user,
                item=item,
                jenis="user",
                tetangga=10
            ) for item in movie_norated_test
        ]
    )
    # pred user to list
    pred_user = list(pred_user_datas)
    # sorting user

    user_topn = pred_user.copy()
    # user_topn=sorted(user_topn,reverse=True)
    user_topn.sort(reverse=True)
    # sorting berdasarkan tetangga
    user_recomendations = []
    # banyak n
    temp = 0
    for i in user_topn:
        if temp < tetangga:
            # print(i)
            user_recomendations.append(movie_norated_test[pred_user.index(i)])
        else:
            break
        temp += 1

    # print("==================================================")
    # print("USER DONE")
    item_data_used_test = rating_matrix.loc[id_user].to_numpy()
    item_movie_norated_test = np.where(item_data_used_test == 0)[0]+1
    item_movie_norated_test.tolist()

    pred_item_datas = np.array(
        [
            # item,
            predict(
                rating_matrix.T,
                item_mean_user,
                item_mean_centered_user,
                item_similarity_user,
                user=id_user,
                item=item,
                jenis="item",
                tetangga=100
            ) for item in movie_norated_test
        ]
    )
    pred_item = list(pred_item_datas)
    item_topn = pred_item.copy()
    item_topn.sort(reverse=True)
    item_recomendations = []
    # banyak n
    temp = 0
    for i in item_topn:
        if temp < tetangga:
            item_recomendations.append(movie_norated_test[pred_item.index(i)])
        else:
            break
        temp += 1

    hybrid_toy_data = list(hybrid(pred_user_datas, pred_item_datas))
    hybrid_topn = hybrid_toy_data.copy()
    hybrid_topn.sort(reverse=True)

    recomendations = []
    # gt = ratings_test_k1[ratings_test_k1['user_id'] == id_user].loc[:,'item_id'].tolist()
    # topN = hybrid_toy_data[(-hybrid_toy_data[:, 1].astype(float)).argsort()][:,0]
    # evTopN = [[],[],[],[]]
    # for n in range(1, 101):
    #       p = precision(ground_truth=gt, topN=topN, n=n)
    #       r = recall(ground_truth=gt, topN=topN, n=n)
    #       f = f1Score(ground_truth=gt, topN=topN, n=n)
    #       n = ndcg(ground_truth=gt, topN=topN, n=n)
    #       evTopN[0].append(p)
    #       evTopN[1].append(r)
    #       evTopN[2].append(f)
    #       evTopN[3].append(n)

    temp = 0
    for i in hybrid_topn:
        if temp < tetangga:
            recomendations.append(movie_norated_test[hybrid_toy_data.index(i)])
        else:
            break
        temp += 1
    hasil_rekomendasi = []
    imdb_film = []
    for i in recomendations:
        hasil_rekomendasi.append(movie_data.loc[i][0])
        imdb_film.append(movie_data.loc[i][2])
    print("imdb done")
    count = 0
    for i in hasil_rekomendasi:
        if count < tetangga and count < 50:
            top_n.append(i)
        count += 1
    # precision=0.1
    # recall=0.2
    # f1=0.3
    banyak_data_rekomendasi = len(top_n)
    banyak_data_irisan = 0
    for i in top_n:
        if i in data_ground_truth:
            banyak_data_irisan += 1

    # EVALUASI MATRIX
    evTopN = [[], [], [], [], []]
    # for n in range(1, 101):
    # print("ini data ground truth",data_ground_truth)
    print("================================================================================================================")
    # print("ini data hasil Rekomendasi",hasil_rekomendasi)
    p = precision(ground_truth=data_ground_truth,
                  topN=hasil_rekomendasi, n=int(tetangga))
    print("Precision", p)
    r = recall(ground_truth=data_ground_truth,
               topN=hasil_rekomendasi, n=int(tetangga))
    f = f1Score(ground_truth=data_ground_truth,
                topN=hasil_rekomendasi, n=int(tetangga))
    d = dcg(ground_truth=data_ground_truth,
            topN=hasil_rekomendasi, n=int(tetangga))
    nd = ndcg(ground_truth=data_ground_truth,
              topN=hasil_rekomendasi, n=int(tetangga))
    evTopN[0].append(p)
    evTopN[1].append(r)
    evTopN[2].append(f)
    evTopN[3].append(d)
    evTopN[4].append(nd)
    # ev.append(evTopN)

    # tetangga=tetangga
    # get
    banyak_user = []
    for i in banyak_users:
        banyak_user.append(i)
    for i in range(1, 463):
        if i not in banyak_user:
            print(i)
    nama_user = "User "+str(id_user)
    return render_template("hasilrekomendasi.html", metod_user=metod_user, metod_item=metod_item, navnya=navnya, judulnya=judulnya, user=id_user, nama_user=nama_user, tetangga=tetangga,  # films=films,
                           banyak_data_train=banyak_data_train, data_train=data_train,
                           banyak_data_rekomendasi=banyak_data_rekomendasi,
                           banyak_data_irisan=banyak_data_irisan,  # data_irisan=data_irisan,
                           banyak_data_ground_truth=banyak_data_ground_truth, data_ground_truth=data_ground_truth,
                           hasil_rekomendasi=hasil_rekomendasi,
                           top_n=top_n, imdb_film=imdb_film,
                           # precision=precision,recall=recall,f1=f1
                           precision=evTopN[0][0], recall=evTopN[1][0], f1=evTopN[2][0], dcg=evTopN[3][0], ndcg=evTopN[4][0]
                           )


@app.route("/metrik_evaluasi")
def metrik_evaluasi_page():
    navnya = ["Home", "", "Metrik Evaluasi"]
    judulnya = "Hasil Rekomendasi"
    id_user = int(request.args.get('user'))
    tetangga = int(request.args.get("tetangga"))
    metod_user = (request.args.get('user-based'))
    metod_item = (request.args.get('item-based'))

    # User
    if metod_user == "PCC":
        metode_usernya = "Pearson Correlation Coefficient (PCC)"
        # User
        # with open(os.path.join(path, 'model, f'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        # mean_user, mean_center_user, similarity_user = joblib.load(os.path.join(path, 'model', f'user_k1_pcc.joblib'))
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    # elif metod_user=="ITR":
    #       metode_usernya = "Improved Triangle Similarity (ITR)"
    #       # User
    #       with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
    #                   mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        # mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        # mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        # similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    elif metod_user == "AMI":
        metode_usernya = "Adjusted Mutual Information (AMI)"
        # User
        # with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "ami", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
    else:
        metode_usernya = "Pearson Correlation Coefficient (PCC)"
        # User
        # with open(os.path.join(path, 'model, 'user_k11.pkl'), 'rb') as model_file:
        #             mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
        mean_user, mean_center_user, similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'user_k1.joblib'))
        mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
        mean_center_user = pd.DataFrame(
            mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
        similarity_user = pd.DataFrame(
            similarity_user, index=rating_matrix.index, columns=rating_matrix.index)

    if metod_item == "PCC":
        metode_itemnya = "Pearson Correlation Coefficient (PCC)"
        # item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'item_k1.joblib'))
        # mean_item_df_k1, mean_centered_item_df_k1, similarity_item_df_k1 = joblib.load(os.path.join(path, 'model', 'item_k1_itr.joblib'))
        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    # elif metod_item=="ITR":
    #       metode_itemnya = "Improved Triangle Similarity (ITR)"
    #       # item
    #       rating_matrix_T = rating_matrix.copy().T
    #       with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
    #                   item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
    #       # mean_item_df_k1, mean_centered_item_df_k1, similarity_item_df_k1 = joblib.load(os.path.join(path, 'model', 'item_k1_itr.joblib'))
    #       item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
    #       item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
    #       item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    elif metod_item == "AMI":
        metode_itemnya = "Adjusted Mutual Information (AMI)"
        # item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        # item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(os.path.join(path, 'model', 'item_k1_ami.joblib'))
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', 'ami', 'item_k1_ami.joblib'))

        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    else:
        metode_itemnya = "Pearson Correlation Coefficient (PCC)"
        # Item
        rating_matrix_T = rating_matrix.copy().T
        # with open(os.path.join(path, 'model, 'item_k11.pkl'), 'rb') as model_file:
        #             item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
        item_mean_user, item_mean_center_user, item_similarity_user = joblib.load(
            os.path.join(path, 'model', "pcc", 'item_k1.joblib'))
        item_mean_user = pd.DataFrame(
            item_mean_user, index=rating_matrix_T.index)
        item_mean_centered_user = pd.DataFrame(
            item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
        item_similarity_user = pd.DataFrame(
            item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)
    # print(id_user.type())
    # print(tetangga.type())
    # print("==========================="*2)
    # data_ground_truth = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5"]
    index_data_ground_truth = []
    # print("ini untuk i")
    asem = 0
    for i in range(1, len(rating_matrix_test.loc[id_user]+1)):
        if rating_matrix_test.loc[id_user][i] != 0.0:
            index_data_ground_truth.append(i)
    data_ground_truth = []
    for i in index_data_ground_truth:
        data_ground_truth.append(movie_data.loc[i][0])
    banyak_data_ground_truth = len(data_ground_truth)
    index_data_train = []
    # for i in range(len(rating_matrix[id_user])):
    for i in range(1, len(rating_matrix.loc[id_user]+1)):
        if rating_matrix.loc[id_user, i] != 0.0:
            index_data_train.append(i)
    data_train = []
    for i in index_data_train:
        data_train.append(movie_data.loc[i][0])
    # index_data_train=rating_matrix.iloc[id_user,:]
    banyak_data_train = len(data_train)

    if id_user not in banyak_users:
        # @app.route("/")
        navnya = ["Home", "Rekomendasi Film", "Tentang Aplikasi"]
        judulnya = "Rekomendasi System"
        nama_user = "Selamat datang di"
        films = ["ini film 1", "ini film 2", "ini film 3", "ini film 4", "ini film 5",
                 "ini film 6", "ini film 7", "ini film 8", "ini film 9", "ini film 10",]
        banyak_user = []
        for i in banyak_users:
            banyak_user.append(i)
        banyak_n = []
        for i in range(1, 51):
            banyak_n.append(i)
        pesan_error = "aktif"
        return render_template("index.html", navnya=navnya, judulnya=judulnya, nama_user=nama_user, banyak_user=banyak_user, banyak_n=banyak_n, pesan_error=pesan_error, id_user=id_user)

    top_n = []
    data_used_test = rating_matrix.loc[id_user].to_numpy()
    movie_norated_test = np.where(data_used_test == 0)[0]+1
    movie_norated_test.tolist()
    pred_user_datas = np.array(
        [
            # user,
            predict(
                rating_matrix,
                mean_user,
                mean_center_user,
                similarity_user,
                user=id_user,
                item=item,
                jenis="user",
                tetangga=10
            ) for item in movie_norated_test
        ]
    )
    # pred user to list
    pred_user = list(pred_user_datas)
    # sorting user

    user_topn = pred_user.copy()
    # user_topn=sorted(user_topn,reverse=True)
    user_topn.sort(reverse=True)
    # sorting berdasarkan tetangga
    user_recomendations = []
    # banyak n
    temp = 0
    for i in user_topn:
        if temp < tetangga:
            # print(i)
            user_recomendations.append(movie_norated_test[pred_user.index(i)])
        else:
            break
        temp += 1

    # print("==================================================")
    # print("USER DONE")
    item_data_used_test = rating_matrix.loc[id_user].to_numpy()
    item_movie_norated_test = np.where(item_data_used_test == 0)[0]+1
    item_movie_norated_test.tolist()

    pred_item_datas = np.array(
        [
            # item,
            predict(
                rating_matrix.T,
                item_mean_user,
                item_mean_centered_user,
                item_similarity_user,
                user=id_user,
                item=item,
                jenis="item",
                tetangga=100
            ) for item in movie_norated_test
        ]
    )
    pred_item = list(pred_item_datas)
    item_topn = pred_item.copy()
    item_topn.sort(reverse=True)
    item_recomendations = []
    # banyak n
    temp = 0
    for i in item_topn:
        if temp < tetangga:
            item_recomendations.append(movie_norated_test[pred_item.index(i)])
        else:
            break
        temp += 1

    hybrid_toy_data = list(hybrid(pred_user_datas, pred_item_datas))
    hybrid_topn = hybrid_toy_data.copy()
    hybrid_topn.sort(reverse=True)

    recomendations = []
    # gt = ratings_test_k1[ratings_test_k1['user_id'] == id_user].loc[:,'item_id'].tolist()
    # topN = hybrid_toy_data[(-hybrid_toy_data[:, 1].astype(float)).argsort()][:,0]
    # evTopN = [[],[],[],[]]
    # for n in range(1, 101):
    #       p = precision(ground_truth=gt, topN=topN, n=n)
    #       r = recall(ground_truth=gt, topN=topN, n=n)
    #       f = f1Score(ground_truth=gt, topN=topN, n=n)
    #       n = ndcg(ground_truth=gt, topN=topN, n=n)
    #       evTopN[0].append(p)
    #       evTopN[1].append(r)
    #       evTopN[2].append(f)
    #       evTopN[3].append(n)

    temp = 0
    for i in hybrid_topn:
        if temp < tetangga:
            recomendations.append(movie_norated_test[hybrid_toy_data.index(i)])
        else:
            break
        temp += 1
    hasil_rekomendasi = []
    imdb_film = []
    for i in recomendations:
        hasil_rekomendasi.append(movie_data.loc[i][0])
        imdb_film.append(movie_data.loc[i][2])
    print("imdb done")
    count = 0
    for i in hasil_rekomendasi:
        if count < tetangga and count < 50:
            top_n.append(i)
        count += 1
    # precision=0.1
    # recall=0.2
    # f1=0.3
    banyak_data_rekomendasi = len(top_n)
    banyak_data_irisan = 0
    for i in top_n:
        if i in data_ground_truth:
            banyak_data_irisan += 1

    # EVALUASI MATRIX
    evTopN = [[], [], [], [], []]
    # for n in range(1, 101):
    # print("ini data ground truth",data_ground_truth)
    print("================================================================================================================")
    # print("ini data hasil Rekomendasi",hasil_rekomendasi)
    p = precision(ground_truth=data_ground_truth,
                  topN=hasil_rekomendasi, n=int(tetangga))
    print("Precision", p)
    r = recall(ground_truth=data_ground_truth,
               topN=hasil_rekomendasi, n=int(tetangga))
    f = f1Score(ground_truth=data_ground_truth,
                topN=hasil_rekomendasi, n=int(tetangga))
    d = dcg(ground_truth=data_ground_truth,
            topN=hasil_rekomendasi, n=int(tetangga))
    icg = idcg(n=int(tetangga))
    print("icg", icg)
    nd = ndcg(ground_truth=data_ground_truth,
              topN=hasil_rekomendasi, n=int(tetangga))
    evTopN[0].append(p)
    evTopN[1].append(r)
    evTopN[2].append(f)
    evTopN[3].append(d)
    evTopN[4].append(nd)
    # ev.append(evTopN)

    # tetangga=tetangga
    # get
    banyak_user = []
    for i in banyak_users:
        banyak_user.append(i)
    for i in range(1, 463):
        if i not in banyak_user:
            print(i)
    nama_user = "User "+str(id_user)
    evTopN[0][0] = round(evTopN[0][0], 4)
    evTopN[1][0] = round(evTopN[1][0], 4)
    evTopN[2][0] = round(evTopN[2][0], 4)
    evTopN[3][0] = round(evTopN[3][0], 4)
    evTopN[4][0] = round(evTopN[4][0], 4)
    # EvTopN =[ round(elem, 3) for elem in my_list ]
    print('evTopN', round(evTopN[4][0], 4))
    return render_template("metrik_evaluasi.html", navnya=navnya, judulnya=judulnya, user=id_user, nama_user=nama_user, tetangga=tetangga,  # films=films,
                           banyak_data_train=banyak_data_train, data_train=data_train,
                           banyak_data_rekomendasi=banyak_data_rekomendasi,
                           banyak_data_irisan=banyak_data_irisan,  # data_irisan=data_irisan,
                           banyak_data_ground_truth=banyak_data_ground_truth, data_ground_truth=data_ground_truth,
                           hasil_rekomendasi=hasil_rekomendasi,
                           top_n=top_n, imdb_film=imdb_film,
                           metod_user=metod_user, metod_item=metod_item,
                           # precision=precision,recall=recall,f1=f1
                           precision=evTopN[0][0], recall=evTopN[1][0], f1=evTopN[2][0], dcg=evTopN[3][0], ndcg=evTopN[4][0]
                           # precision=precision,recall=evTopN[1][0],f1=evTopN[2][0],dcg=evTopN[3][0],ndcg=evTopN[4][0]
                           )


if __name__ == "__main__":
    app.run(debug=True)
