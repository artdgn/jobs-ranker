import os
import json

from utils.logger import logger

import common

TASKS_DIR = os.path.realpath(os.path.dirname(__file__))


def get_task_config(task_name: str):

    path = task_name
    if os.path.sep not in path:
        path = os.path.join(TASKS_DIR, task_name)
    if not path.endswith('.json'):
        path += '.json'

    try:
        with open(path, 'rt') as f:
            task = TaskConfig()
            data = json.load(f)
            data['name'] = os.path.splitext(os.path.split(path)[1])[0]
            task.update(data)
            return task

    except FileNotFoundError as e:
        found_tasks = [f.split('.json')[0]
                       for f in os.listdir(TASKS_DIR) if '.json' in f]

        logger.error(f'task json file for {task_name} not found. '
                     f'found task files for: {found_tasks}')
        raise e


class TaskConfig(dict):

    @property
    def name(self):
        return self['name']

    @property
    def search_url(self):
        return self['search_url']

    @property
    def crawls_dir(self):
        dir =  os.path.join(common.CRAWLS_DIR, self.name)
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def scrapy_log_dir(self):
        dir = os.path.join(common.SCRAPY_LOG_DIR, self.name)
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def crawl_job_dir(self):
        dir = os.path.join(common.CRAWLS_JOB_DIR, self.name)
        os.makedirs(dir, exist_ok=True)
        return dir
