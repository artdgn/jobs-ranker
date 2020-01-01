import itertools
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jobs_ranker.config import common
from jobs_ranker.utils.instrumentation import log_time_and_shape


@log_time_and_shape
def calc_duplicates(strings, keep='first'):

    if keep not in ['first', 'last']:
        raise ValueError(f'keep value can only be "first" or "last"')

    strings = np.array(strings).astype(str)
    dup_i, dup_j = duplicates_by_tfidf_cosine(strings)

    dup_dict_inds = defaultdict(set)
    for i, j in zip(dup_i, dup_j):
        dup_dict_inds[i].add(j)
        dup_dict_inds[j].add(i)

    keep_inds = set()
    keep_i = 0 if keep == 'first' else -1
    for i in range(len(strings)):
        keep_inds.add(sorted([i] + list(dup_dict_inds[i]))[keep_i])

    return sorted(list(keep_inds)), dup_dict_inds


@log_time_and_shape
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
    threshold = common.MLParams.dedup_simil_thershold
    dup_i, dup_j = np.where(simil_mat > threshold)
    return dup_i, dup_j


def inspect_simil_threshold(strings, simil_mat, threshold):
    # pd.Series(simil_mat.ravel()).hist(bins=200)
    dup_i, dup_j = np.where((simil_mat > threshold) &
                            (simil_mat < threshold + 0.1))
    i = np.random.randint(0, len(dup_i))
    print_side_by_side(strings[dup_i[i]], strings[dup_j[i]])


def print_side_by_side(a, b, size=100):
    def buffer_gen(s):
        ind = 0
        while ind < len(s):
            offset = 0
            while offset < size and ind + offset < len(s) \
                    and s[ind + offset] != '\n':
                offset += 1
            yield s[ind:ind + offset]
            while ind + offset < len(s) and s[ind + offset] in ['\n', '']:
                offset += 1
            ind += offset

    for a_buff, b_buff in itertools.zip_longest(
            buffer_gen(a), buffer_gen(b), fillvalue=''):
        print(a_buff.ljust(size) + '  |  ' + b_buff)
