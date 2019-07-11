import sys

from joblist.ranking import JobsRanker
from tasks import TasksConfigsDao
from utils.logger import logger


class Labeler:
    _SKIP = 'skip'
    _STOP = 'stop'
    _RECALC = 'recalc'
    _CONTROL_TOKENS = [_SKIP, _STOP, _RECALC]
    _END_MESSAGE = ('No more new unlabeled jobs. '
                    'Try turning dedup off to go over duplicates. '
                    'run with --help flag for more info')

    def __init__(self, jobs_ranker: JobsRanker):
        self.jobs_ranker = jobs_ranker
        self.y_tok = self.jobs_ranker.pos_label
        self.n_tok = self.jobs_ranker.neg_label
        self.skipped = set()

    def _prompt(self):
        return (
            "Rate the job relevance on a scale of 0.0 to 1.0, "
            f"or use '{self.y_tok}' for yes or '{self.n_tok}' for no.\n"
            f"Input ( {self.y_tok} / {self.n_tok} / number / "
            f"{self._STOP} / {self._RECALC} / {self._SKIP} ): ")

    def label_data(self, data=None):
        if data is not None:
            print(str(data))
        return input(self._prompt())

    def end_labeling_message(self, message):
        logger.info(message)

    def label_jobs_loop(self, recalc_everytime=False):
        for url in iter(self.jobs_ranker.next_unlabeled, None):

            row = self.jobs_ranker.url_data(url)

            resp = self.label_data(row)

            while not (resp in self._CONTROL_TOKENS or
                       self.jobs_ranker.is_valid_label(str(resp))):
                resp = self.label_data()

            if resp == self._STOP:
                break

            if resp == self._SKIP:
                self.skipped.add(url)
                continue

            if resp == self._RECALC:
                self.jobs_ranker.rerank_jobs()
                continue

            # not any of the control_tokens
            self.jobs_ranker.add_label(url, resp)

            if recalc_everytime:
                self.jobs_ranker.rerank_jobs()

        if self.jobs_ranker.next_unlabeled() is None:
            self.end_labeling_message(self._END_MESSAGE)


class TaskChooser:

    def __init__(self, tasks_dao: TasksConfigsDao):
        self.tasks_dao = tasks_dao

    def choose_from_task_list(self, tasks, message, instructions):
        numbered_tasks_list = "\n".join(
            [f"\t{i}: {s}" for i, s in zip(range(len(tasks)), tasks)])

        prompt = f'{message}\n{numbered_tasks_list}\n{instructions}'

        return input(prompt)

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

        resp = self.choose_from_task_list(
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
