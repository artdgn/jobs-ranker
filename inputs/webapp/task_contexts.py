import collections

import flask

from joblist.ranking import JobsRanker
from tasks.dao import TasksConfigsDao


class TaskContext:

    def __init__(self, task_name):
        self.task_name = task_name
        self._cur_url = None
        self._ranker = None
        self._skipped = set()

    def get_task_config(self):
        try:
            return TasksConfigsDao.get_task_config(self.task_name)
        except FileNotFoundError:
            flask.abort(404, f'task "{self.task_name}" not found')

    def get_ranker(self) -> JobsRanker:
        task_config = self.get_task_config()
        if self._ranker is None:
            self._ranker = JobsRanker(
                task_config=task_config,
                dedup_new=True,
                skipped_as_negatives=False)
            self.move_to_next_url()
        return self._ranker

    def load_ranker(self):
        ranker = self.get_ranker()
        if not ranker.loaded and not ranker.busy:
            ranker.load_and_process_data(background=True)
            flask.flash(f'loading data for task "{self.task_name}"')

    def get_url(self):
        ranker = self.get_ranker()
        if not self._cur_url:
            url = ranker.next_unlabeled()
            while url is not None and url in self._skipped:
                url = ranker.next_unlabeled()
            self._cur_url = url
        return self._cur_url

    def move_to_next_url(self):
        self._cur_url = None

    def reload_ranker(self):
        self.get_ranker().load_and_process_data(background=True)
        self.move_to_next_url()

    def recalc(self):
        self.get_ranker().rerank_jobs(background=True)
        self.move_to_next_url()

    def skip(self, url):
        self._skipped.add(url)
        self.move_to_next_url()


class TasksContexts(collections.defaultdict):
    def __missing__(self, key):
        self[key] = TaskContext(key)
        return self[key]

    # hint for code completion / lynting
    def __getitem__(self, item) -> TaskContext:
        return super().__getitem__(item)
