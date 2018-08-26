from joblist.joblist_processing import JobsListLabeler

scraped_jobs_csv = 'crawler/spiders/jora-7.csv'
labeled_jobs_csv = 'data/labeled_jobs.csv'
keywords_json = 'data/keywords.json'

jobs = JobsListLabeler(scraped=scraped_jobs_csv, keywords=keywords_json, labeled=labeled_jobs_csv)
jobs.label_jobs()


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