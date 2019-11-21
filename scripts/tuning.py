from jobs_ranker.io import text
from jobs_ranker.joblist.ranking import JobsRanker
from jobs_ranker.tasks.configs import TasksConfigsDao

task_chooser = text.TaskChooser(tasks_dao=TasksConfigsDao())

task_config = task_chooser.load_or_choose_task(task_name='product-manager')

ranker = JobsRanker(task_config=task_config)
ranker.load_and_process_data(background=False)