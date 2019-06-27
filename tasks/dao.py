import json
import os

from tasks.config import TaskConfig


class TasksDao:
    TASKS_DIR = os.path.realpath(os.path.dirname(__file__))

    @classmethod
    def tasks_in_scope(cls):
        return [f.split('.json')[0]
                for f in os.listdir(cls.TASKS_DIR) if '.json' in f]


    @classmethod
    def get_task_config(cls, task_name: str):
        path = task_name
        if os.path.exists(path):
            pass
        else:
            if os.path.sep not in path:
                path = os.path.join(cls.TASKS_DIR, task_name)
            if not path.endswith('.json'):
                path += '.json'

        with open(path, 'rt') as f:
            task = TaskConfig()
            data = json.load(f)
            data['name'] = os.path.splitext(os.path.split(path)[1])[0]
            task.update(data)
            return task