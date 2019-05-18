import sys

from joblist.ranking import JobsRanker
from tasks.config import tasks_in_scope, TASKS_DIR, get_task_config
from utils.logger import logger


_SKIP = 'skip'
_STOP = 'stop'
_RECALC = 'recalc'
_CONTROL_TOKENS = [_SKIP, _STOP, _RECALC]


def label_jobs(joblist: JobsRanker, recalc_everytime=False):

    y = joblist.pos_label
    n = joblist.neg_label
    prompt = ("Rate the job relevance on a scale of 0.0 to 1.0, "
              f"or use '{y}' for yes or '{n}' for no.\n"
              f"Input ( {y} / {n} / number / {_STOP} / {_RECALC} / {_SKIP} ): ")

    urls_stack = joblist.get_sorted_urls_stack()
    skipped = set()
    while len(urls_stack):

        url = urls_stack.pop()

        if (not joblist.is_labeled(url)) and not (url in skipped):

            row = joblist.displayable_job_by_url(url)

            print(str(row))

            resp = input(prompt)

            while not (resp in _CONTROL_TOKENS or
                       joblist.is_valid_label_input(resp)):
                resp = input(prompt)

            if resp == _STOP:
                break

            if resp == _SKIP:
                skipped.add(url)
                continue

            if resp == _RECALC:
                joblist.rank_jobs()
                urls_stack = joblist.get_sorted_urls_stack()
                continue

            # not any of the control_tokens
            joblist.add_label(row.url, resp)

            if recalc_everytime:
                joblist.rank_jobs()
                urls_stack = joblist.get_sorted_urls_stack()

    if not len(urls_stack):
        logger.info('No more new unlabeled jobs. '
                    'Try turning dedup off to go over duplicates. '
                    'run with --help flag for more info')


def load_or_choose_task(task_name):

    try:
        return get_task_config(task_name)

    except FileNotFoundError:
        pass

    tasks = tasks_in_scope()

    tasks_folder = TASKS_DIR

    tasks.append('.. cancel and exit')

    numbered_tasks_list = "\n".join(
        [f"\t{i}: {s}" for i, s in zip(range(len(tasks)), tasks)])

    prompt = (f'Found these tasks in the {tasks_folder} folder:'
              f'\n{numbered_tasks_list}\n'
              f'Choose an option number or provide exact path to your task: ')

    resp = input(prompt)

    # parse input
    try:
        option_number = int(resp)
        if option_number == len(tasks) - 1:
            sys.exit()
        elif 0 <= option_number < len(tasks) - 1:
            task_name = tasks[option_number]
    except ValueError:
        task_name = resp

    return load_or_choose_task(task_name)