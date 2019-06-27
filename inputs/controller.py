import sys

from inputs.text import LabelFrontendAPI
from joblist.ranking import RankerAPI
from tasks.dao import TasksDao


class LabelController:

    _SKIP = 'skip'
    _STOP = 'stop'
    _RECALC = 'recalc'
    _CONTROL_TOKENS = [_SKIP, _STOP, _RECALC]
    _END_MESSAGE = ('No more new unlabeled jobs. '
                    'Try turning dedup off to go over duplicates. '
                    'run with --help flag for more info')

    def __init__(self,
                 jobs_ranker: RankerAPI,
                 frontend: LabelFrontendAPI):
        self.jobs_ranker = jobs_ranker
        self.frontend = frontend
        self.frontend.y_tok = self.jobs_ranker.pos_label
        self.frontend.n_tok = self.jobs_ranker.neg_label
        self.frontend.skip_tok = self._SKIP
        self.frontend.stop_tok = self._STOP
        self.frontend.recalc_tok = self._RECALC

    def label_jobs(self, recalc_everytime=False):

        urls_stack = self.jobs_ranker.get_sorted_urls_stack()
        skipped = set()
        while len(urls_stack):

            url = urls_stack.pop()

            if (not self.jobs_ranker.is_labeled(url)) and not (url in skipped):

                row = self.jobs_ranker.displayable_job_by_url(url)

                resp = self.frontend.label_data(row)

                while not (resp in self._CONTROL_TOKENS or
                           self.jobs_ranker.is_valid_label_input(resp)):
                    resp = self.frontend.label_data()

                if resp == self._STOP:
                    break

                if resp == self._SKIP:
                    skipped.add(url)
                    continue

                if resp == self._RECALC:
                    self.jobs_ranker.rank_jobs()
                    urls_stack = self.jobs_ranker.get_sorted_urls_stack()
                    continue

                # not any of the control_tokens
                self.jobs_ranker.add_label(row.url, resp)

                if recalc_everytime:
                    self.jobs_ranker.rank_jobs()
                    urls_stack = self.jobs_ranker.get_sorted_urls_stack()

        if not len(urls_stack):
            self.frontend.end_labeling_message(self._END_MESSAGE)


class TaskChoiceController:

    def __init__(self, tasks_dao: TasksDao, frontend):
        self.tasks_dao = tasks_dao
        self.frontend = frontend

    def load_or_choose_task(self, task_name):
        try:
            return self.tasks_dao.get_task_config(task_name)

        except FileNotFoundError:
            pass

        tasks = self.tasks_dao.tasks_in_scope()

        tasks_folder = self.tasks_dao.TASKS_DIR

        tasks.append('.. cancel and exit')

        message = f'Found these tasks in the {tasks_folder} folder:'

        instructions = f'Choose an option number or provide exact path to your task: '

        resp = self.frontend.choose_from_task_list(
            tasks=tasks,
            message=message,
            instructions=instructions)

        # parse inputs
        try:
            option_number = int(resp)
            if option_number == len(tasks) - 1:
                sys.exit()
            elif 0 <= option_number < len(tasks) - 1:
                task_name = tasks[option_number]
        except ValueError:
            task_name = resp

        return self.load_or_choose_task(task_name)