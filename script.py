import os
import pandas as pd
from joblist_processing import JobsList
from labeled_jobs import LabeledJobs

scraped_jobs_csv = './crawler/spiders/jora-6.csv'
labeled_jobs_csv = 'labeled_jobs.csv'

scraped = JobsList().read_csv(scraped_jobs_csv).process_df()
df = scraped.df

labeled = LabeledJobs(labeled_jobs_csv)
for ind, row in df.drop('description', axis=1).iterrows():
    if not labeled.labeled(row.url):
        resp = input(str(row) + '\n' + 'y/n/label/stop?')
        if resp=='stop':
            break
        labeled.label(row.url, resp)
