from collections import defaultdict

import editdistance
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist


def duplicates_by_edit_distance(strings, n_first_words=100, dup_threshold=50):

    vec = pd.Series(strings).str.split().values

    dist_func = lambda u, v: \
        editdistance.eval(u[0][:n_first_words], v[0][:n_first_words]) \
            if isinstance(u[0], list) and isinstance(v[0], list) else np.nan

    dist_mat = squareform(pdist(vec.reshape(-1, 1), dist_func))

    return np.where((dist_mat + np.eye(*dist_mat.shape) * dup_threshold) < dup_threshold)

    # import matplotlib.pyplot as plt
    # plt.imshow(-dist_mat, cmap='hot')
    # show_dist = lambda d, i: print( \
    #   vec[np.where(dist_mat == d)[0][i]], \
    #   '\n', vec[np.where(dist_mat == d)[1][i]])
    # pd.Series(dist_mat.ravel()).hist(bins=50)


def dedup_by_descriptions_similarity(strings, keep='first'):

    dup_i, dup_j = duplicates_by_edit_distance(strings)

    dup_dict = defaultdict(set)
    for i, j in zip(dup_i, dup_j):
        dup_dict[i].add(j)
        dup_dict[j].add(i)

    if keep == 'first':
        take_ind = 0
    elif keep == 'last':
        take_ind = -1
    else:
        raise ValueError(f'unsupported "keep" value: {keep}')

    keep_inds = set()
    for i in range(len(strings)):
        keep_inds.add(sorted([i] + list(dup_dict[i]))[take_ind])
    return sorted(list(keep_inds)), dup_dict