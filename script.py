import pandas as pd
from joblist_processing import JobsList


filename = 'jora-5.csv'

jobs = JobsList().read_csv(filename).process_df()
df = jobs.df

print(df.head(10))




