import os
import json

import common

TASKS_DIR = os.path.realpath(os.path.dirname(__file__))


def tasks_in_scope():
    return [f.split('.json')[0]
            for f in os.listdir(TASKS_DIR) if '.json' in f]


def get_task_config(task_name: str):
    path = task_name
    if os.path.exists(path):
        pass
    else:
        if os.path.sep not in path:
            path = os.path.join(TASKS_DIR, task_name)
        if not path.endswith('.json'):
            path += '.json'

    with open(path, 'rt') as f:
        task = TaskConfig()
        data = json.load(f)
        data['name'] = os.path.splitext(os.path.split(path)[1])[0]
        task.update(data)
        return task


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
