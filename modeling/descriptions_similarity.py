from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import logger
import common


def dedup_by_descriptions_similarity(strings, keep='first'):

    dup_i, dup_j = duplicates_by_tfidf_cosine(strings)

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


def duplicates_by_tfidf_cosine(strings):
    max_docs_cutoff = max(common.MLParams.dedup_tfidf_max_df_cutoff,
                          int(len(strings) * common.MLParams.dedup_tfidf_max_df_ratio))
    tf = TfidfVectorizer(
        ngram_range=common.MLParams.dedup_tfidf_ngram_range,
        max_df=max_docs_cutoff,
        stop_words='english')
    vecs = tf.fit_transform(strings)
    simil_mat = cosine_similarity(vecs)
    simil_mat -= np.eye(*simil_mat.shape)
    dup_i, dup_j = np.where(simil_mat > common.MLParams.dedup_simil_thershold)
    return dup_i, dup_j


def inspect_simil_threshold(strings, simil_mat, threshold):
    # pd.Series(simil_mat.ravel()).hist(bins=200)
    dup_i, dup_j = np.where((simil_mat > threshold) &
                            (simil_mat < threshold + 0.1))
    i = np.random.randint(0, len(dup_i))
    print_side_by_side(strings[dup_i[i]], strings[dup_j[i]])


def print_side_by_side(a, b, size=100):

    def seek(s, ind):
        for offset in range(1, size):
            if ind + offset >= len(s) or s[ind + offset] == '\n':
                return s[ind:ind + offset], ind + offset + 1
        else:
            return s[ind:ind + size], ind + size

    i, j = 0, 0
    while i < len(a) and j < len(b):
        buff_a, i = seek(a, i)
        buff_b, j = seek(b, j)
        print(buff_a.ljust(size) + '  |  ' + buff_b)
