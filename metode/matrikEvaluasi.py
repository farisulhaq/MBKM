import numpy as np


def precision(ground_truth, topN, n=1):
    return (len(np.intersect1d(topN[:n], ground_truth)) / n)


def recall(ground_truth, topN, n=1):
    return len(np.intersect1d(topN[:n], ground_truth)) / len(set(ground_truth))


def f1Score(ground_truth, topN, n=1):
    p = precision(ground_truth, topN, n)
    r = recall(ground_truth, topN, n)

    return ((2 * p * r) / (p + r)) if (p > 0 and r > 0) else 0


def idcg(n):
    return np.sum((1 / np.log2(1 + np.array(list(range(1, n+1))))))


def dcg(ground_truth, topN, n):
    a = np.array([(1 / np.log2(1 + x)) for x in range(1, n+1)])
    b = np.array([np.sum(np.where(tp == np.array(ground_truth), 1, 0))
                 for tp in topN[:n]])
    return np.sum(a*b)


def ndcg(ground_truth, topN, n):
    return (dcg(ground_truth, topN, n) / idcg(n))


def evaluasi(ground_truth, topN, n=1):
    """Calculate and show MSE, RMSE, & MAE from predicted rating with hybrid method

    Parameters
    ----------
    y_actual: numpy.ndarray
        The user-item test data
    y_predicted: numpy.ndarray
      The user-item rating that have been predicted with hybrid method

    Returns
    -------
    precision: float
        mean squared error of hybrid method
    recall: float
        root mean squared error of hybrid method
    f1-score: float
        mean absolute error of hybrid method
    """

    return [precision(ground_truth, topN, n=1), recall(ground_truth, topN, n=1), f1Score(ground_truth, topN, n=1), ndcg(ground_truth, topN, n=1)]
