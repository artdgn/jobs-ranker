import os
import pandas as pd

from common import LABELED_ROOT_DIR


class LabeledJobs:
    url_col = 'url'
    label_col = 'label'
    pos_label = 'y'
    neg_label = 'n'

    def __init__(self, task_name, dup_dict=None):
        self.filename = self._task_name_to_filename(task_name)
        self.dup_dict = dup_dict if dup_dict is not None else {}
        self.df = self.load(self.filename)

    def _task_name_to_filename(self, task_name):
        return os.path.join(LABELED_ROOT_DIR, f'{task_name}.csv')

    def save(self):
        self.df.to_csv(self.filename, index=None)

    def load(self, filename):
        if os.path.exists(filename):
            return pd.read_csv(filename). \
                drop_duplicates(subset=[self.url_col], keep='last')
        else:
            return pd.DataFrame({self.url_col: [], self.label_col: []})

    def _urls_with_dups(self, url):
        return self.dup_dict[url] if url in self.dup_dict else [url]

    def _labeled_url(self, url):
        return url in self.df[self.url_col].values

    def labeled(self, url):
        return any(self._labeled_url(u) for u in self._urls_with_dups(url))

    def label(self, url, label):
        if not self.labeled(url):
            # add this url
            self.df = self.df.append(
                pd.DataFrame({self.url_col: [url], self.label_col: [label]}))
            self.save()

    def __repr__(self):
        total = len(self.df)
        neg = (self.df.loc[:, self.label_col] == self.neg_label).sum()
        pos = (self.df.loc[:, self.label_col] == self.pos_label).sum()
        return (f'LabeledJobs: {total} labeled ({neg / total:.1%} neg, '
                f'{pos / total:.1%} pos, '
                f'{(total - pos - neg) / total:.1%} partial relevance)')

    def export_df(self, keep='first'):

        df = self.df.copy()

        for url in df[self.url_col]:
            if url in df[self.url_col] and url in self.dup_dict and len(self.dup_dict[url]):
                dup_urls = self.dup_dict[url]
                if self.labeled(url):
                    if keep == 'first':
                        label = df.loc[df[self.url_col].isin(dup_urls)][self.label_col].iloc[0]
                        urls = self._urls_with_dups(url)
                        df.loc[df[self.url_col].isin(urls), self.label_col] = label
                        df = df.loc[~df[self.url_col].isin(dup_urls[1:])]
                    # elif keep == 'last':
                    #     label = df.loc[df[self.url_col].isin(dup_urls)][self.label_col].iloc[-1]
                    #     self.label(dup_urls[-1], label)
                    #     df = df.loc[~df[self.url_col].isin(dup_urls[:-1])]
                    else:
                        raise ValueError(f'unsupported "keep" value: {keep}')

        return df
