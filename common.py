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

