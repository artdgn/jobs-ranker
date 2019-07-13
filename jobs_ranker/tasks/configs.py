import json
import os

from jobs_ranker import common


class TaskConfig(dict):

    @property
    def name(self):
        return self['name']

    @property
    def search_urls(self):
        return self['search_urls']

    @property
    def crawls_dir(self):
        path = os.path.join(common.CRAWLS_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def scrapy_log_dir(self):
        path = os.path.join(common.SCRAPY_LOG_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def crawl_job_dir(self):
        path = os.path.join(common.CRAWLS_JOB_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        return path


class TasksConfigsDao:
    TASKS_DIRS = [os.path.realpath(os.path.dirname(__file__)),
                  os.path.join(common.DATA_DIR, 'tasks')]

    @classmethod
    def tasks_in_scope(cls):
        tasks = []
        for path in cls.TASKS_DIRS:
            tasks.extend([f.split('.json')[0]
                          for f in os.listdir(path) if '.json' in f])
        return tasks

    @classmethod
    def load_task_config(cls, task_name: str):
        task_file = task_name
        if not task_file.endswith('.json'):  # append json
            task_file += '.json'

        for folder in cls.TASKS_DIRS:
            full_path = os.path.join(folder, task_file)
            if os.path.exists(full_path):
                break
        else:
            raise FileNotFoundError(f"couldn't find task '{task_name}' "
                                    f"in {cls.TASKS_DIRS}")

        with open(full_path, 'rt') as f:
            task = TaskConfig()
            data = json.load(f)
            data['name'] = task_name
            task.update(data)
            return task
