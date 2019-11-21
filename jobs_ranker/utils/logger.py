import sys
import logging
import os

from jobs_ranker import common

log_path = os.path.join(common.LOG_DIR, f'log-{common.current_timestamp()}.log')

logFormatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt='%H:%M:%S')
logger = logging.getLogger()

# level
logger.setLevel(logging.INFO)

# console
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# log file
fileHanlder = logging.FileHandler(log_path)
fileHanlder.setFormatter(logFormatter)
logger.addHandler(fileHanlder)



