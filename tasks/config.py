import os

import common


class TaskConfig(dict):

    @property
    def name(self):
        return self['name']

    @property
    def search_url(self):
        return self['search_url']

    @property
    def crawls_dir(self):
        path = os.path.join(common.CRAWLS_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def scrapy_log_dir(self):
        dir = os.path.join(common.SCRAPY_LOG_DIR, self.name)
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def crawl_job_dir(self):
        path = os.path.join(common.CRAWLS_JOB_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        return path
