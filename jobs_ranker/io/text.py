import sys

from jobs_ranker.joblist.ranking import RankerAPI
from jobs_ranker.tasks.configs import TasksConfigsDao
from jobs_ranker.utils.logger import logger


class LabelingLoop:
    _SKIP = 'skip'
    _STOP = 'stop'
    _RECALC = 'recalc'
    _CONTROL_TOKENS = [_SKIP, _STOP, _RECALC]
    _END_MESSAGE = ('No more new unlabeled jobs. '
                    'Try turning dedup off to go over duplicates. '
                    'run with --help flag for more info')

    def __init__(self, ranker: RankerAPI):
        self.ranker = ranker
        self.skipped = set()

    def _prompt(self):
        y_tok = self.ranker.labeler.pos_label
        n_tok = self.ranker.labeler.neg_label
        return (
            "Rate the job relevance on a scale of 0.0 to 1.0, "
            f"or use '{y_tok}' for yes or '{n_tok}' for no.\n"
            f"Input ( {y_tok} / {n_tok} / number / "
            f"{self._STOP} / {self._RECALC} / {self._SKIP} ): ")

    def label_input(self, data=None):
        if data is not None:
            print(str(data))
        return input(self._prompt())

    @staticmethod
    def end_labeling_message(message):
        logger.info(message)

    def run_loop(self, recalc_everytime=False):
        for url in iter(self.ranker.next_unlabeled, None):

            row = self.ranker.url_data(url)

            resp = self.label_input(row)

            while not (resp in self._CONTROL_TOKENS or
                       self.ranker.labeler.is_valid_label(str(resp))):
                resp = self.label_input()

            if resp == self._STOP:
                break

            if resp == self._SKIP:
                self.skipped.add(url)
                continue

            if resp == self._RECALC:
                self.ranker.rerank_jobs()
                continue

            # not any of the control_tokens
            self.ranker.labeler.add_label(url, resp)

            if recalc_everytime:
                self.ranker.rerank_jobs()

        if self.ranker.next_unlabeled() is None:
            self.end_labeling_message(self._END_MESSAGE)


class TaskChooser:

    def __init__(self, tasks_dao: TasksConfigsDao):
        self.tasks_dao = tasks_dao

    @staticmethod
    def choose_from_task_list(tasks, message, instructions):
        numbered_tasks_list = "\n".join(
            [f"\t{i}: {s}" for i, s in zip(range(len(tasks)), tasks)])

        prompt = f'{message}\n{numbered_tasks_list}\n{instructions}'

        return input(prompt)

    def load_or_choose_task(self, task_name):
        try:
            return self.tasks_dao.load_task_config(task_name)

        except FileNotFoundError:
            pass

        tasks = self.tasks_dao.tasks_in_scope()

        tasks_folders = self.tasks_dao.TASKS_DIRS

        tasks.append('.. cancel and exit')

        message = f'Found these tasks in the {tasks_folders} folder:'

        instructions = f'Choose an option number: '

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
