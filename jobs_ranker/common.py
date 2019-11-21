import datetime
import os

ROOT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'data')

LOG_DIR = os.path.join(DATA_DIR, 'logs')

SCRAPY_LOG_DIR = os.path.join(DATA_DIR, 'scrapy_logs')

CRAWLS_DIR = os.path.join(DATA_DIR, 'crawls')

CRAWLS_JOB_DIR = os.path.join(DATA_DIR, 'crawls_temp')

LABELED_ROOT_DIR = os.path.join(DATA_DIR, 'labeled')

TASKS_CONFIGS_DIR = os.path.join(DATA_DIR, 'tasks')

[os.makedirs(path, exist_ok=True) for path in
 [DATA_DIR, LOG_DIR, SCRAPY_LOG_DIR,
  CRAWLS_DIR, CRAWLS_JOB_DIR, LABELED_ROOT_DIR,
  TASKS_CONFIGS_DIR]]


def current_timestamp():
    return datetime.datetime.now().isoformat()


def current_date():
    return datetime.datetime.now().date().isoformat()


class MLParams:
    min_training_samples = 10
    test_ratio = 0.3

    rf_n_estimators = 100
    lgbm_n_estimators = 300
    lgbm_learning_rate = 0.005
    rf_tfidf_ngram_range = (1, 3)
    rf_tfidf_min_df = 3

    dedup_tfidf_ngram_range = (1, 3)
    dedup_tfidf_max_df_cutoff = 50
    dedup_tfidf_max_df_ratio = 0.02
    dedup_simil_thershold = 0.5
    dedup_keep = 'last'


class InfoParams:
    top_n_feat = 20


HEADERS = {
    'authority': 'www.google.com',
    'scheme': 'https',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image'
              '/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'accept-language': 'en-GB,en;q=0.9,en-US;q=0.8',
    'dnt': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML,'
                  ' like Gecko) Chrome/74.0.3729.157 Safari/537.36',
}
