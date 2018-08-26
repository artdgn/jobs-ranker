import pandas as pd
from joblist.joblist_processing import JobsListLabeler

scraped_jobs_old_csv = 'data/jora-6.csv'
scraped_jobs_csv = 'data/jora-7.csv'
labeled_jobs_csv = 'data/labeled_jobs.csv'
keywords_json = 'data/keywords.json'

jobs = JobsListLabeler(
    scraped=scraped_jobs_csv, keywords=keywords_json,
    labeled=labeled_jobs_csv, older_scraped=[scraped_jobs_old_csv])

jobs.label_jobs()







