import numpy as np


def calculate_mean(data):
    user_mean = (data.sum(axis=1))/(np.count_nonzero(data, axis=1))
    user_mean[np.isnan(user_mean)] = 0.0
    return user_mean


def calculate_mean_centered(data, mean):

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

  hasil = 0
  try:
    # determine based model wheter user-based or item-based
    # take user/item rating, mean centered, and simillarity to calculate
    if jenis == "user":
        dt = datas.loc[:, item].to_numpy()
        meanC = mean_centered.loc[:, item].to_numpy()
        simi = similarity.loc[user, :].to_numpy()
    elif jenis == "item":
        try:
            dt = datas.loc[:, user].to_numpy()
            meanC = mean_centered.loc[:, user].to_numpy()
            simi = similarity.loc[item, :].to_numpy()
        except KeyError:
            simi = np.zeros(similarity.shape[1])
            print(f"User {user} has yet rated Item {item}")

    # user/item index that is yet rated
    idx_dt = np.where(dt != 0)

    # filter user/item rating, mean centered, and simillarity value that is not zero
    nilai_mean_c = np.array(meanC)[idx_dt]
    nilai_similarity = simi[idx_dt]

    # take user/item simillarity index as neighbors and sort it
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
          if a == 0 or b == 0:
            hasil = 0
      else:
          hasil = mean.loc[item] + (a/b)
          if a == 0 or b == 0:
            hasil = 0

    else:
      if jenis == "user":
          hasil = mean.loc[user] + 0

      else:
          hasil = mean.loc[item] + 0

  except KeyError:
    if jenis == "user":
        print(f"Item {item} has never rated by all users")
        hasil = mean.loc[user] + 0
    else:
        print(f"User {user} has yet rated Item {item}")
        hasil = mean.loc[item] + 0

  return hasil


def hybrid(predict_user, predict_item, r1=0.7):

    # degree of fusion will be splitted in to two parameter
    # the one (Γ1) is used for user-based model
    # the others (Γ2 = 1 - Γ1) is used for item-based model
    r = np.array([r1, 1-r1])

    # weighting all the users and items corresponding to the Topk UCF and TopkICF models
    # see equation 13 (hybrid formula) in paper
    r_caping = np.column_stack((predict_user, predict_item))
    result = np.sum((r*r_caping), axis=1)

    return result
