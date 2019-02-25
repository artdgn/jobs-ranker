import os
import datetime

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

LOG_DIR = os.path.join(ROOT_DIR, 'data/logs')
os.makedirs(LOG_DIR, exist_ok=True)
SCRAPY_LOG_DIR = os.path.join(ROOT_DIR, 'data/scrapy_logs')
os.makedirs(SCRAPY_LOG_DIR, exist_ok=True)
CRAWLS_DIR = os.path.join(ROOT_DIR, 'data/crawls')
os.makedirs(CRAWLS_DIR, exist_ok=True)
CRAWLS_JOB_DIR = os.path.join(ROOT_DIR, 'data/crawls_temp')
os.makedirs(CRAWLS_JOB_DIR, exist_ok=True)

CURRENT_TIMESTAMP = datetime.datetime.now().isoformat()
CURRENT_DATE = datetime.datetime.now().date().isoformat()

LOG_FILEPATH = os.path.join(LOG_DIR, f'log_{CURRENT_TIMESTAMP}.txt')

