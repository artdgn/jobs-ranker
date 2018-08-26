import os
import pandas as pd

class LabeledJobs:
    url_col = 'url'
    label_col = 'label'

    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(filename):
            self.df = pd.read_csv(filename)
        else:
            self.df = pd.DataFrame({self.url_col: [], self.label_col: []})

    def save(self):
        self.df.to_csv(self.filename, index=None)

    def load(self):
        self.df = pd.read_csv(self.filename)

    def labeled(self, url):
        return url in self.df[self.url_col].values.tolist()

    def label(self, url, label):
        if not self.labeled(url):
            self.df = self.df.append(
                pd.DataFrame({self.url_col: [url], self.label_col: [label]}))
        else:
            self.df.loc[self.df[self.url_col] == url, self.label_col] = label
        self.save()

    def __repr__(self):
        return self.df.__repr__()