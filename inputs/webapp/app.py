import json

import flask

from joblist.ranking import JobsRanker
from tasks.dao import TasksDao
from crawler.scraping import start_scraping

app = flask.Flask(__name__)

tasks_dao = TasksDao()
rankers = {}


@app.route('/')
def instructions():
    return flask.redirect(flask.url_for('tasks_list'))


@app.route('/tasks/')
def tasks_list():
    tasks = tasks_dao.tasks_in_scope()
    task_urls = [{'name': t, 'url': flask.url_for('task_description', task_name=t)}
                 for t in tasks]
    return flask.render_template('tasks_list.html', task_urls=task_urls)


@app.errorhandler(404)
def not_found(message):
    return flask.render_template(
        'error.html',
        message=message,
        back_url=flask.url_for('instructions'),
        back_text='Go home..'), 404


def get_task_config(task_name):
    try:
        return tasks_dao.get_task_config(task_name)
    except FileNotFoundError:
        flask.abort(404, f'task "{task_name}" not found')


@app.route('/<task_name>/')
def task_description(task_name):
    task_config = get_task_config(task_name)
    return flask.render_template(
        'task_page.html',
        task_name=task_name,
        scrape_url=flask.url_for('scrape_task', task_name=task_name),
        label_url=flask.url_for('label_task', task_name=task_name),
        task_data=json.dumps(task_config, indent=4))


def get_ranker(task_name) -> JobsRanker:
    task_config = get_task_config(task_name)
    ranker = rankers.get(task_name)
    if ranker is None:
        ranker = JobsRanker(
            task_config=task_config,
            dedup_new=True,
            skipped_as_negatives=False)
        rankers[task_name] = ranker
    return ranker


@app.route('/<task_name>/label/')
def label_task(task_name):
    ranker = get_ranker(task_name)

    if not ranker.loaded:
        if not ranker.busy:
            ranker.load_and_process_data(background=True)
        return flask.render_template(
            'waiting.html',
            message='Waiting for labeler to crunch all the data',
            seconds=5)
    else:


        return f'labeling for {task_name}'


@app.route('/<task_name>/scrape/')
def scrape_task(task_name):
    task_config = get_task_config(task_name)

    # proc, scrape_log_path = start_scraping(task_config=task_config,
    #                                        http_cache=True,
    #                                        blocking=False)
    """
    add /start/ url to start
    start scraping and stream log file 
    keep scraping processes in some global structure (filesystem file with pid?) 
    active - show, else show "no active scrape" - /start
    https://stackoverflow.com/questions/36384286/how-to-integrate-flask-scrapy
    https://github.com/scrapinghub/scrapyrt
    """
    return f'scraping for {task_name}'


def run_app(debug=False):
    app.run(host='0.0.0.0', debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
