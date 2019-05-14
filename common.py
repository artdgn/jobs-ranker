import os
import datetime

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

LOG_DIR = os.path.join(ROOT_DIR, 'data/logs')

SCRAPY_LOG_DIR = os.path.join(ROOT_DIR, 'data/scrapy_logs')

CRAWLS_DIR = os.path.join(ROOT_DIR, 'data/crawls')

CRAWLS_JOB_DIR = os.path.join(ROOT_DIR, 'data/crawls_temp')

LABELED_ROOT_DIR = os.path.join(ROOT_DIR, 'data/labeled')

[os.makedirs(CRAWLS_JOB_DIR, exist_ok=True) for dir in
 [LOG_DIR, SCRAPY_LOG_DIR, CRAWLS_DIR, CRAWLS_JOB_DIR, LABELED_ROOT_DIR]]


CURRENT_TIMESTAMP = datetime.datetime.now().isoformat()
CURRENT_DATE = datetime.datetime.now().date().isoformat()


LOG_FILEPATH = os.path.join(LOG_DIR, f'log_{CURRENT_TIMESTAMP}.txt')


class MLParams:

   min_model_score = 0.1
   min_training_samples = 10
   test_ratio = 0.3

   rf_n_estimators = 100
   rf_tfidf_ngram_range = (1, 3)
   rf_tfidf_min_df = 3

   dedup_tfidf_ngram_range = (1, 3)
   dedup_tfidf_max_df_cutoff = 50
   dedup_tfidf_max_df_ratio = 0.02
   dedup_simil_thershold = 0.5


class InfoParams:
    top_n_feat = 20

