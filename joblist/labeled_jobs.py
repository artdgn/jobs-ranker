import os
import pandas as pd

class LabeledJobs:
    url_col = 'url'
    label_col = 'label'

    def __init__(self, filename):
        self.filename = filename
        self.dup_dict = {}
        if os.path.exists(filename):
            self.df = pd.read_csv(filename)
        else:
            self.df = pd.DataFrame({self.url_col: [], self.label_col: []})

    def save(self):
        self.df.to_csv(self.filename, index=None)

    def load(self):
        self.df = pd.read_csv(self.filename).\
            drop_duplicates(subset=[self.url_col], keep='last')

    def labeled(self, url):
        return url in self.df[self.url_col].values

    def label(self, url, label):
        if url in self.dup_dict:
            urls = self.dup_dict[url]
        else:
            urls = [url]

        if not any(self.labeled(u) for u in urls):
            self.df = self.df.append(
                pd.DataFrame({self.url_col: [url], self.label_col: [label]}))
        else:
            self.df.loc[self.df[self.url_col].isin(urls), self.label_col] = label
        self.save()

    def __repr__(self):
        return self.df.__repr__()

    def _set_dup_dict(self, ind_dup_dict, urls):
        self.dup_dict = {urls[i]: urls[sorted([i] + list(dups))]
                         for i, dups in ind_dup_dict.items()}

    def dedup(self, dup_dict, all_urls, keep='first'):

        self._set_dup_dict(ind_dup_dict=dup_dict, urls=all_urls)

        for url in all_urls:
            if len(self.dup_dict[url]):
                dup_urls = self.dup_dict[url]
                labeled = [self.labeled(u) for u in dup_urls]
                if any(labeled):
                    if keep == 'first':
                        label = self.df.loc[self.df[self.url_col].isin(dup_urls)][self.label_col].iloc[0]
                        self.label(dup_urls[0], label)
                        self.df = self.df.loc[~self.df[self.url_col].isin(dup_urls[1:])]
                    # elif keep == 'last':
                    #     label = self.df.loc[self.df[self.url_col].isin(dup_urls)][self.label_col].iloc[-1]
                    #     self.label(dup_urls[-1], label)
                    #     self.df = self.df.loc[~self.df[self.url_col].isin(dup_urls[:-1])]
                    else:
                        raise ValueError(f'unsupported "keep" value: {keep}')
        self.save()
