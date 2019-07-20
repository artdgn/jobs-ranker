import json
import os

from jobs_ranker import common


class TaskConfig(dict):

    @classmethod
    def from_dict(cls, name, path, **kwargs):
        return cls(_name=name, _path=path, **kwargs)

    @property
    def name(self):
        return self['_name']

    @property
    def path(self):
        return self['_path']

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

    def __str__(self):
        copy = self.copy()
        copy.pop('_name')
        copy.pop('_path')
        return json.dumps(copy, indent=2)


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
            return TaskConfig.from_dict(name=task_name,
                                        path=full_path,
                                        **json.load(f))

    @classmethod
    def save_task_config(cls, config: TaskConfig):
        with open(config.path, 'wt') as f:
            f.write(str(config))

    @classmethod
    def validate_config_json(cls, text):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError('not a valid JSON') from e
        config = TaskConfig(**data)
        for field in [
            'search_urls',
            'description_negative',
            'description_positive',
            'title_negative',
            'title_positive',
        ]:
            if field not in config:
                raise ValueError(f'field "{field}" is missing')
        if not config.search_urls:
            raise ValueError('"search_urls" key is empty')
        return config

    @classmethod
    def update_config(cls, task_name, text):
        orig_config = cls.load_task_config(task_name=task_name)
        updated_config = cls.validate_config_json(text)
        orig_config.update(updated_config)
        cls.save_task_config(config=orig_config)
