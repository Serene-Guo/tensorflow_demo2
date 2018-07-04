from sklearn.metrics import roc_auc_score
import numpy as np
import multiprocessing as mp

def calc_group_scores(X, coef, groups):
    scores = []
    for g in groups:
        scores.append(np.asarray((X[:,g].dot(coef[:, g].T))))
    
    scores = np.hstack(scores)
    return scores

def get_permuation_importance(y, X, coef, groups, n_samples=20, seed=42, n_jobs=1):
    if len(coef.shape) == 1:
        coef = coef.reshape((1,-1))
    scores = calc_group_scores(X, coef, groups)
    total_score = scores.sum(axis=1)
    remaining_scores = total_score[:,np.newaxis] - scores

    rng = np.random.RandomState(seed)

    N = scores.shape[0]
    indices = rng.choice(N, (n_samples, N), replace=True)

    def calc_metrics(I, score_func=roc_auc_score):
        y_ = y[I]
        total_score_ = total_score[I]
        scores_ = scores[I]
        remaining_scores_ = remaining_scores[I]
        global_metric = score_func(y_, total_score_)
        single_metrics = np.array([score_func(y_, s) for s in scores_.T])
        remaining_metrics = np.array([score_func(y_, s) for s in remaining_scores_.T])
        return np.concatenate((single_metrics, global_metric-remaining_metrics))
    if n_jobs==1:
        metrics = np.array(list(map(calc_metrics, indices)))
    else:
        pool = mp.Pool(n_jobs)
        metrics = np.array(list(pool.map(calc_metrics, indices)))
    return metrics
