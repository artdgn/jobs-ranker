import pandas as pd
import re


class JobsList:
    desc_neg_keywords = ['financ', 'banking', 'gambl', 'insurance', 'fintech', 'consult',
                         'marketi', 'professional services', '.net', 'react', 'frontend',
                         'scala', 'spark', 'government', 'citizen', 'tableau', 'money',
                         'insight', 'java', 'advert']
    desc_pos_keywords = ['senior', 'deep learning', 'nlp', 'cnn', 'machine learning', ' ml ',
                         'recognition', 'neural', 'keras', 'tensorflow', 'ruby']

    title_neg_keywords = ['consult', '.net', 'react', 'frontend', 'javascript',
                          'devops', 'sales', 'analyst', 'data engineer', 'business',
                          'fullstack', 'architect', 'scala', 'spark', 'ruby']
    title_pos_keywords = ['senior', 'machine learning', 'ml', 'data']


    def __init__(self):
        self._pandas_console_options()

    def read_csv(self, filename):
        self.df = pd.read_csv(filename)
        drop_cols = [col for col in self.df.columns if col.startswith('download_')] + \
                    ['depth']
        self.df.drop(drop_cols, axis=1, inplace=True)
        return self

    @staticmethod
    def _pandas_console_options():
        pd.set_option('display.max_colwidth', 300)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

    @staticmethod
    def _extract_numeric_fields_on_row(row):
        row['description'] = str(row['description']).lower().replace('\n', ' ').replace('\t', ' ')

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

    def extract_numeric_fields(self):
        self.df = self.df.apply(self._extract_numeric_fields_on_row, axis=1)
        self.df.drop(['salary', 'date'], axis=1, inplace=True)

    def process_df(self):
        self.extract_numeric_fields()
        self.add_keyword_score()
        return self

    def add_keyword_score(self):

        def named_regex(word_list, name):
            return '(?P<%s>' % name + '|'.join(word_list) + ')'

        def keyword_density_func(keyword_list, group_name):
            regex = named_regex(keyword_list, group_name)
            return lambda s: len(re.findall(regex, s.lower())) / len(s)

        self.df['desc_neg_count'] = self.df.description.apply(keyword_density_func(self.desc_neg_keywords, 'desc_neg'))
        self.df['desc_pos_count'] = self.df.description.apply(keyword_density_func(self.desc_pos_keywords, 'desc_pos'))

        self.df['title_neg_count'] = self.df.title.apply(keyword_density_func(self.title_neg_keywords, 'title_neg'))
        self.df['title_pos_count'] = self.df.title.apply(keyword_density_func(self.title_pos_keywords, 'title_pos'))

        self.df['comb_score'] = \
            1 / self.df.desc_pos_count.rank(ascending=False) - 1 / self.df.desc_neg_count.rank(ascending=False) + \
            1 / self.df.title_pos_count.rank(ascending=False) - 1 / self.df.title_neg_count.rank(ascending=False)

        self.df.sort_values('comb_score', ascending=False, inplace=True)

#########
'''
def reg_test(df, x_col, y_col, result_col):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    df_reg = df[~df[y_col].isnull()][[x_col, y_col]].copy()
    # fit for eval
    train, test = train_test_split(df_reg, test_size=0.3)
    reg = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=None)),
        # ('regressor', Lasso())])
        ('regressor', RandomForestRegressor())])
    reg.fit(train[x_col], train[y_col])
    # score
    print(r2_score(test[y_col], reg.predict(test[x_col])))
    # refit
    reg.fit(df_reg[x_col], df_reg[y_col])
    # predict
    df[result_col] = reg.predict(df[x_col])
    return df

df_guess = reg_test(df_post, x_col='title', y_col='salary_high', result_col='salary_guess')
'''