import numpy as np


def calculate_mean(data):
    """Calculate mean rating of user-item matrix

    Parameters
    ----------
    data: numpy.ndarray
        The user-item matrix to calculate which matrix's size is user (n) times items (m)

    Returns
    -------
    user_mean: numpy.ndarray
        mean rating user-item matrix which matrix size is user/items times 1
    """

    # sum all rating based on user/item row then divide by number of rating that is not zero
    user_mean = (data.sum(axis=1))/(np.count_nonzero(data, axis=1))
    user_mean[np.isnan(user_mean)] = 0.0
    return user_mean


def calculate_mean_centered(data, mean):
    """Calculate mean centered rating of user-item matrix

    Parameters
    ----------
    data: numpy.ndarray
        The user-item matrix to calculate which matrix's size is user (n) times items (m)
    mean: numpy.ndarray
        The mean rating of user-item matrix

    Returns
    -------
    mat_mean_centered: numpy.ndarray
        mean centered rating user-item matrix which matrix size is same as data parameter
    """

    mat_mean_centered = []
    # iterate by rows
    for i in range(len(data)):
        row = []
        # iterate columns
        for j in range(len(data[i])):
            row.append(data[i][j] - mean[i] if data[i][j] != 0 else 0)
        mat_mean_centered.append(row)

    return np.array(mat_mean_centered)


def predict(datas, mean, mean_centered, similarity, user=3, item=2, tetangga=2, jenis='user'):
    """Calculate prediction of user target, item target and how many prediction based on number of neighbor

    Parameters
    ----------
    data: numpy.ndarray
        The user-item matrix to calculate which matrix's size is user (n) times items (m)
    mean: numpy.ndarray
      The mean rating of user-item matrix
    mean_centered: numpy.ndarray
      The mean centered rating of user-item matrix
    similarity: numpy.ndarray
      The simmalirity (user or item) matrix
    user: int
      User target, default=3
    item: int
      Item target, default=2
    tetangga: int
      Amoun of neighbors, default=2
    jenis: str
      User or Item based model in CF technique that will be used, default=user

    Returns
    -------
    hasil: numpy.ndarray which has same size as 'tetangga' parameter
        simmilarity of user-item matrix
    """

    # determine based model wheter user-based or item-based
    # take user/item rating, mean centered, and simillarity to calculate
    if jenis == "user":
        dt = datas.loc[:, item].to_numpy()
        meanC = mean_centered.loc[:, item].to_numpy()
        simi = similarity.loc[user, :].to_numpy()
    elif jenis == "item":
        dt = datas.loc[:, user].to_numpy()
        meanC = mean_centered.loc[:, user].to_numpy()
        simi = similarity.loc[item, :].to_numpy()

    # user/item index that has rated
    idx_dt = np.where(dt != 0)

    # filter user/item rating, mean centered, and simillarity value that is not zero
    nilai_mean_c = np.array(meanC)[idx_dt]
    nilai_similarity = simi[idx_dt]

    # take user/item similarity index as neighbors and sort it
    idx_sim = (-nilai_similarity).argsort()[:tetangga]

    # see equation 5 & 6 (prediction formula) in paper
    # numerator
    a = np.sum(nilai_mean_c[idx_sim] * nilai_similarity[idx_sim])

    # denomerator
    b = np.abs(nilai_similarity[idx_sim]).sum()

    # check denominator is not zero and add μ (mean rating)
    if b != 0:
        if jenis == "user":
            hasil = mean.loc[user] + (a/b)
        else:
            hasil = mean.loc[item] + (a/b)
    else:
        if jenis == "user":
            hasil = mean.loc[user] + 0
        else:
            hasil = mean.loc[item] + 0

    return [item, hasil]


def hybrid(predict_user, predict_item, r1=0.7):
    """Calculate prediction of user-item matrix from hybridization of collaborative learning with Liniear Regression

    Parameters
    ----------
    data: numpy.ndarray
        The user-item matrix to calculate which matrix's size is user (n) times items (m)
    predict_user: numpy.ndarray
        The prediction ratings of user
    predict_item: numpy.ndarray
        The prediction ratings of item
    r1: int
        degree fusion is used as the weights of the prediction function (see Equations 12 and 13) in paper

    Returns
    -------
    result: numpy.ndarray
        unknown list of prediction with hybrid method
    """

    # degree of fusion will be splitted in to two parameter
    # the one (Γ1) is used for user-based model
    # the others (Γ2 = 1 - Γ1) is used for item-based model
    r = np.array([r1, 1-r1])

    # weighting all the users and items corresponding to the Topk UCF and TopkICF models
    # see equation 13 (hybrid formula) in paper
    r_caping = np.column_stack((predict_user, predict_item))
    result = np.sum((r*r_caping), axis=1)

    return result


def sistemRekomendasi(rating_matrix, mean_user_df, mean_centered_user_df, similarity_user_df, mean_item_df, mean_centered_item_df, similarity_item_df, user, tetanggaU, tetanggaI, r1):
    # prediksi UCF dan ICF
    prediksiUCF, prediksiICF = [], []
    for item in (np.where((rating_matrix.loc[1, :] == 0))[0]+1).tolist():
        prediksiUCF.append(predict(rating_matrix, mean_user_df, mean_centered_user_df,
                           similarity_user_df, user=user, item=item, tetangga=tetanggaU, jenis='user'))
        prediksiICF.append(predict(rating_matrix.T, mean_item_df, mean_centered_item_df,
                           similarity_item_df, user=user, item=item, tetangga=tetanggaI, jenis='item'))
    prediksiUCF, prediksiICF = np.array(
        prediksiUCF, dtype=object), np.asarray(prediksiICF, dtype=object)

    # Container untuk hybrid
    result_hybrid = prediksiUCF.copy()

    # hasil prediksi diambil index 1 dan diganti type float
    ucf = prediksiUCF[:, 1]
    icf = prediksiICF[:, 1]

    # hybrid
    uicf = hybrid(ucf, icf, r1)

    # replace container hybrid
    result_hybrid[:, 1] = uicf

    return result_hybrid
