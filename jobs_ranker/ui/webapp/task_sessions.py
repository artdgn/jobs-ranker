import collections
import os
import functools
from multiprocessing import Process

import flask

from jobs_ranker.scraping.crawling import CrawlsFilesDao, JoraCrawlProcess
from jobs_ranker.joblist import ranking
from jobs_ranker.tasks.configs import TasksConfigsDao


def raise_404_on_filenotfound(func):
    @functools.wraps(func)
    def internal(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except FileNotFoundError as e:
            flask.abort(404, str(e))

    return internal


class TaskSession:

    def __init__(self, task_name):
        self.task_name = task_name
        self._cur_urls = set()
        self._ranker = None
        self._skipped = set()
        self.should_notify_scores = True
        self._crawler = None
        self._crawl_subproc = None
        self.recent_edit_attempt = None

    def get_config(self):
        try:
            return TasksConfigsDao.load_config(self.task_name)
        except FileNotFoundError:
            flask.abort(404, f'task "{self.task_name}" not found')

    @property
    def ranker(self) -> ranking.RankerAPI:
        if self._ranker is None:
            self._ranker = ranking.get_ranker(
                task_config=self.get_config(),
                dedup_new=True,
                skipped_as_negatives=False)
            self.reset_session_state()
        return self._ranker

    @raise_404_on_filenotfound
    def load_ranker(self):
        if not self.ranker.loaded and not self.ranker.busy:
            self.ranker.load_and_process_data(background=True)
            flask.flash(f'loading data for task "{self.task_name}"', 'info')

    def get_url(self):
        if not self._cur_urls:
            url = self.ranker.next_unlabeled()
            while url is not None and url in self._skipped:
                url = self.ranker.next_unlabeled()
            self._cur_urls.add(url)
        return next(iter(self._cur_urls))

    def add_label(self, url, label):
        self.ranker.labeler.add_label(url, label)
        self._cur_urls.discard(url)

    def reset_session_state(self):
        self._cur_urls = set()
        self._skipped = set()
        self.should_notify_scores = True

    @raise_404_on_filenotfound
    def reload_ranker(self):
        self.ranker.load_and_process_data(background=True)
        self.reset_session_state()

    def recalc(self):
        self.ranker.rerank_jobs(background=True)
        self.reset_session_state()

    def skip(self, url):
        self._skipped.add(url)
        self._cur_urls.discard(url)

    def _start_crawl(self):
        self._crawler.start()
        self._crawler.join()

    def start_crawl(self):
        self._crawler = JoraCrawlProcess(task_config=self.get_config())
        self._crawl_subproc = Process(target=self._start_crawl)
        self._crawl_subproc.start()

    @property
    def crawling(self):
        return (self._crawl_subproc is not None and
                self._crawl_subproc.is_alive())

    def days_since_last_crawl(self):
        return CrawlsFilesDao.days_since_last_crawl(self.get_config())

    def jobs_in_latest_crawl(self):
        if (self._crawler.crawl_output_path and
                os.path.exists(self._crawler.crawl_output_path)):
            return CrawlsFilesDao.rows_in_file(self._crawler.crawl_output_path)
        else:
            return 0

    def all_crawls_lengths(self):
        return CrawlsFilesDao.all_crawls_lengths(self.get_config())

    def expected_jobs_per_crawl(self):
        return (len(self.get_config().search_urls) *
                JoraCrawlProcess.expected_jobs_per_search)

    def ranker_outdated(self):
        if (self._crawler and
                not self.crawling and
                (self.ranker.recent_crawl_source !=
                 self._crawler.crawl_output_path)):
            return True
        else:
            return False

    def update_config(self, text):
        try:
            TasksConfigsDao.update(task_name=self.task_name, text=text)
            self.recent_edit_attempt = None
        except ValueError as e:
            self.recent_edit_attempt = text
            raise e


class TasksSessions(collections.defaultdict):
    def __missing__(self, key):
        self[key] = TaskSession(key)
        return self[key]

    # hint for code completion / lynting
    def __getitem__(self, item) -> TaskSession:
        return super().__getitem__(item)
