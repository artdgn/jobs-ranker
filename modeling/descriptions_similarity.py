from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import logger


def optional_import(package_name):
    import importlib
    try:
        return importlib.import_module(package_name)
    except ImportError as e:
        logger.error(f'pip install {package_name} module to use called functionality')
        logger.exception()
        raise e


def dedup_by_descriptions_similarity(strings, keep='first'):

    dup_i, dup_j = duplicates_by_tfidf_cosine(strings)
    # dup_i, dup_j = duplicates_by_edit_distance(strings)

    dup_dict_inds = defaultdict(set)
    for i, j in zip(dup_i, dup_j):
        dup_dict_inds[i].add(j)
        dup_dict_inds[j].add(i)

    if keep == 'first':
        take_ind = 0
    elif keep == 'last':
        take_ind = -1
    else:
        raise ValueError(f'unsupported "keep" value: {keep}')

    keep_inds = set()
    for i in range(len(strings)):
        keep_inds.add(sorted([i] + list(dup_dict_inds[i]))[take_ind])
    return sorted(list(keep_inds)), dup_dict_inds


def duplicates_by_tfidf_cosine(strings, dup_threshold=0.4, connect_cooccurring=True):
    tf = TfidfVectorizer(ngram_range=(1,3), max_df=50, stop_words='english')
    vecs = tf.fit_transform(strings)
    simil_mat = cosine_similarity(vecs)
    simil_mat -= np.eye(*simil_mat.shape)
    if connect_cooccurring:
        simil_mat = simil_mat.T @ simil_mat  # connect similarities
        # dup_threshold *= dup_threshold  # square threshold (because multiplying similarity scores)
    dup_i, dup_j = np.where(simil_mat > dup_threshold)
    return dup_i, dup_j
    # pd.Series(simil_mat.ravel()).hist(bins=200)


def duplicates_by_edit_distance(strings, n_first_words=100, dup_threshold=50):

    editdistance = optional_import('editdistance')

    vec = pd.Series(strings).str.split().values

    dist_func = lambda u, v: \
        editdistance.eval(u[0][:n_first_words], v[0][:n_first_words]) \
            if isinstance(u[0], list) and isinstance(v[0], list) else np.nan

    ## normalized version
    # dist_func = lambda u, v: \
    #     editdistance.eval(u[0][:int(len(u[0]) * len_ratio)],
    #                       v[0][:int(len(v[0]) * len_ratio)]) \
    #     / np.sqrt(len(u[0]) * len(v[0])) \
    #         if isinstance(u[0], list) and isinstance(v[0], list) else np.nan

    dist_mat = squareform(pdist(vec.reshape(-1, 1), dist_func))

    dup_i, dup_j = np.where((dist_mat + np.eye(*dist_mat.shape) * dup_threshold) < dup_threshold)
    return dup_i, dup_j

    # import matplotlib.pyplot as plt
    # plt.imshow(-dist_mat, cmap='hot')
    # show_dist = lambda d, i: print( \
    #   vec[np.where(dist_mat == d)[0][i]], \
    #   '\n', vec[np.where(dist_mat == d)[1][i]])
    # pd.Series(dist_mat.ravel()).hist(bins=50)
