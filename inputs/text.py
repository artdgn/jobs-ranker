import abc

from utils.logger import logger


class LabelFrontendAPI(abc.ABC):
    y_tok = 'y'
    n_tok = 'n'
    skip_tok = 'skip'
    stop_tok = 'stop'
    recalc_tok = 'recalc'

    @abc.abstractmethod
    def label_data(self, data=None):
        return ''

    @abc.abstractmethod
    def end_labeling_message(self, message):
        pass


class TextLabelFrontend(LabelFrontendAPI):

    def _prompt(self):
        return (
            "Rate the job relevance on a scale of 0.0 to 1.0, "
            f"or use '{self.y_tok}' for yes or '{self.n_tok}' for no.\n"
            f"Input ( {self.y_tok} / {self.n_tok} / number / "
            f"{self.stop_tok} / {self.recalc_tok} / {self.skip_tok} ): ")

    def label_data(self, data=None):
        if data is not None:
            print(str(data))
        return input(self._prompt())

    def end_labeling_message(self, message):
        logger.info(message)


class TaskChoiceFrontendAPI(abc.ABC):

    @abc.abstractmethod
    def choose_from_task_list(self, tasks, message, instructions):
        return ''


class TaskChoiceFrontend(TaskChoiceFrontendAPI):

    def choose_from_task_list(self, tasks, message, instructions):
        numbered_tasks_list = "\n".join(
            [f"\t{i}: {s}" for i, s in zip(range(len(tasks)), tasks)])

        prompt = f'{message}\n{numbered_tasks_list}\n{instructions}'

        return input(prompt)


