import collections
from multiprocessing import Process

import flask

from jobs_rank.crawler.scraping import CrawlsFilesDao, CrawlProcess
from jobs_rank.joblist.ranking import JobsRanker
from jobs_rank.tasks.dao import TasksConfigsDao


class TaskContext:

    def __init__(self, task_name):
        self.task_name = task_name
        self._cur_urls = set()
        self._ranker = None
        self._skipped = set()
        self._scraper = None
        self._scrape_subproc = None

    def get_config(self):
        try:
            return TasksConfigsDao.get_task_config(self.task_name)
        except FileNotFoundError:
            flask.abort(404, f'task "{self.task_name}" not found')

    def get_ranker(self) -> JobsRanker:
        if self._ranker is None:
            self._ranker = JobsRanker(
                task_config=self.get_config(),
                dedup_new=True,
                skipped_as_negatives=False)
            self.reset_urls()
        return self._ranker

    def load_ranker(self):
        ranker = self.get_ranker()
        if not ranker.loaded and not ranker.busy:
            ranker.load_and_process_data(background=True)
            flask.flash(f'loading data for task "{self.task_name}"')

    def get_url(self):
        ranker = self.get_ranker()
        if not self._cur_urls:
            url = ranker.next_unlabeled()
            while url is not None and url in self._skipped:
                url = ranker.next_unlabeled()
            self._cur_urls.add(url)
        return next(iter(self._cur_urls))

    def add_label(self, url, label):
        self.get_ranker().add_label(url, label)
        self._cur_urls.discard(url)

    def reset_urls(self):
        self._cur_urls = set()

    def reload_ranker(self):
        self.get_ranker().load_and_process_data(background=True)
        self.reset_urls()

    def recalc(self):
        self.get_ranker().rerank_jobs(background=True)
        self.reset_urls()

    def skip(self, url):
        self._skipped.add(url)
        self._cur_urls.discard(url)

    def _start_scrape(self):
        self._scraper.start_scraping()
        self._scraper.join()

    def start_scrape(self):
        self._scraper = CrawlProcess(task_config=self.get_config(),
                                     http_cache=True)
        self._scrape_subproc = Process(target=self._start_scrape)
        self._scrape_subproc.start()

    @property
    def scraping(self):
        return (self._scrape_subproc is not None and
                self._scrape_subproc.is_alive())

    def days_since_last_crawl(self):
        return CrawlsFilesDao.days_since_last_crawl(self.get_config())


class TasksContexts(collections.defaultdict):
    def __missing__(self, key):
        self[key] = TaskContext(key)
        return self[key]

    # hint for code completion / lynting
    def __getitem__(self, item) -> TaskContext:
        return super().__getitem__(item)
