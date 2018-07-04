import pandas as pd
import re

filename = 'jora-5.csv'
df = pd.read_csv(filename)

df['description'] = df['description'].str.lower()

def process_row(row):
    # salary
    sal_str = str(row['salary'])
    sal_nums = re.findall('[0-9]+', sal_str.replace(',', ''))
    sal_mult = ('year' in sal_str) * 1 + ('day' in sal_str) * 200 + ('hour' in sal_str) * 1600
    if len(sal_nums) == 2:
        row['salary_low'] = float(sal_nums[0]) * sal_mult
        row['salary_high'] = float(sal_nums[1]) * sal_mult

    # date
    date_str = str(row['date'])
    date_nums = re.findall('[0-9]+', date_str)
    date_mult = ('day' in date_str) * 1 + ('month' in date_str) * 30 + ('hour' in date_str) * 0.04
    if len(date_nums) == 1:
        row['days_age'] = int(int(date_nums[0]) * date_mult)
    return row

negatives = ['financ', 'banking', 'gambl', 'insurance', 'fintech']

positives = ['senior', 'deep learning', 'nlp', 'cnn']

df['pos_count'] = df['description'].str.contains()