import os
import json

import sys

from utils.logger import logger

import common

TASKS_DIR = os.path.realpath(os.path.dirname(__file__))


def task_dir_tasks():
    return [f.split('.json')[0]
     for f in os.listdir(TASKS_DIR) if '.json' in f]

def get_task_config(task_name: str = ''):

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
        return get_task_config(user_input_task_config())


def user_input_task_config():

    tasks = task_dir_tasks()
    tasks.append('.. cancel and exit')

    numbered_tasks_list = "\n".join(
        [f"\t{i}: {s}" for i, s in zip(range(len(tasks)), tasks)])
    prompt = \
        f'Found these tasks in the ./tasks/ folder:\n{numbered_tasks_list}\n' \
        f'Choose an option number ur provide exact path to another task: '

    resp = input(prompt)

    # parse input
    try:
        option_number = int(resp)
        if option_number == len(tasks) - 1:
            sys.exit()
        elif 0 <= option_number < len(tasks) - 1:
            return tasks[option_number]
    except ValueError:
        pass

    return resp


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
