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
    def past_scrapes_relevance_days(self):
        return int(self.get('past_scrapes_relevance_days', 0) or 0)

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

    def data_dict(self):
        copy = self.copy()
        copy.pop('_name')
        copy.pop('_path')
        return copy

    def __str__(self):
        return json.dumps(self.data_dict(), indent=2)


class TasksConfigsDao:
    _TASK_DIR_CODE = os.path.realpath(os.path.dirname(__file__))
    TASKS_DIRS = [_TASK_DIR_CODE, common.TASKS_CONFIGS_DIR]

    @classmethod
    def all_names(cls):
        tasks = []
        for path in cls.TASKS_DIRS:
            tasks.extend([f.split('.json')[0]
                          for f in os.listdir(path) if '.json' in f])
        return tasks

    @classmethod
    def load_config(cls, task_name: str):
        task_file = f'{task_name}.json'
        for folder in cls.TASKS_DIRS:
            full_path = os.path.join(folder, task_file)
            if os.path.exists(full_path):
                break
        else:
            raise FileNotFoundError(f"couldn't find file '{task_file}' "
                                    f"in {cls.TASKS_DIRS}")

        with open(full_path, 'rt') as f:
            return TaskConfig.from_dict(name=task_name,
                                        path=full_path,
                                        **json.load(f))

    @classmethod
    def _save(cls, config: TaskConfig):
        with open(config.path, 'wt') as f:
            json.dump(config.data_dict(), f)

    @classmethod
    def _validate_data_json(cls, text):
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
    def update(cls, task_name, text):
        orig_config = cls.load_config(task_name=task_name)
        updated_config = cls._validate_data_json(text)
        orig_config.update(updated_config)
        cls._save(config=orig_config)

    @classmethod
    def _validate_new_name(cls, name):
        name = name.lower()
        if not name:
            raise ValueError('task name is empty')
        elif ' ' in name:
            raise ValueError('task name cannot contain spaces')
        elif name in cls.all_names():
            raise ValueError(f'task name "{name}" is already taken')
        else:
            return name

    @classmethod
    def new_task(cls, name, copy_from=None):
        name = cls._validate_new_name(name)

        if copy_from is None:
            first_name = cls.all_names()[0]
            copy_from = cls.load_config(task_name=str(first_name))

        if not isinstance(copy_from, TaskConfig):
            raise ValueError(f'copy_from must be of type '
                             f'"{TaskConfig.__name__}"')

        config = TaskConfig.from_dict(
            name=name,
            path=os.path.join(common.TASKS_CONFIGS_DIR, f'{name}.json'),
            **copy_from.data_dict())
        cls._save(config)

