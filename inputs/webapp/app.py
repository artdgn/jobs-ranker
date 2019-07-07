import collections
import json

import flask
import requests

from common import HEADERS
from joblist.ranking import JobsRanker
from tasks.dao import TasksDao
from crawler.scraping import start_scraping
from utils.logger import logger

app = flask.Flask(__name__)

tasks_dao = TasksDao()
rankers = {}
skipped = collections.defaultdict(set)
url_deques = collections.defaultdict(collections.deque)


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
        back_text='Go back to start..'), 404


def get_task_config(task_name):
    try:
        return tasks_dao.get_task_config(task_name)
    except FileNotFoundError:
        flask.abort(404, f'task "{task_name}" not found')

def load_ranker(task_name):
    ranker = get_ranker(task_name)

    if not ranker.loaded and not ranker.busy:
        ranker.load_and_process_data(background=True)


@app.route('/<task_name>/')
def task_description(task_name):
    task_config = get_task_config(task_name)
    load_ranker(task_name)

    return flask.render_template(
        'task_page.html',
        task_name=task_name,
        back_url=flask.url_for('tasks_list'),
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


def get_url(task_name):
    ranker = get_ranker(task_name)
    if not url_deques[task_name]:
        url = ranker.next_unlabeled()
        while url is not None and url in skipped[task_name]:
            url = ranker.next_unlabeled()
        url_deques[task_name].append(url)
    else:
        url = url_deques[task_name][0]
    return url


@app.route('/<task_name>/label/')
def label_task(task_name):
    ranker = get_ranker(task_name)
    load_ranker(task_name)

    if ranker.busy:
        return flask.render_template(
            'waiting.html',
            message='Waiting for labeler to crunch all the data',
            seconds=5)
    else:
        url = get_url(task_name)

        if url is None:
            return flask.render_template(
                'error.html',
                message=(f'No more new unlabeled jobs for task "{task_name}", '
                         f'try dedup off, or scrape new jobs'),
                back_url=flask.url_for('instructions'),
                back_text='Go back to start..')
        else:
            # go label it
            return flask.redirect(flask.url_for(
                'label_url',
                task_name=task_name,
                url=url))


@app.route('/<task_name>/label/<path:url>/', methods=['GET', 'POST'])
def label_url(task_name, url):
    ranker = get_ranker(task_name)
    data = ranker.url_data(url)

    if flask.request.method == 'GET':
        return flask.render_template(
            'job_page.html',
            job_url=url,
            url_data=data,
            skip_url=flask.url_for('skip_url', task_name=task_name, url=url),
            recalc_url=flask.url_for('recalc', task_name=task_name),
            back_url=flask.url_for('task_description', task_name=task_name),
            iframe_url=flask.url_for('source_posting', url=url)
            )

    else:
        form = flask.request.form
        if form.get('no'):
            resp = ranker.neg_label
        elif form.get('yes'):
            resp = ranker.pos_label
        else:
            resp = form['label']

        if not ranker.is_valid_label(str(resp)):
            # bad input, render same page again
            return flask.redirect(flask.url_for(
                'label_url',
                task_name=task_name,
                url=url))
        else:
            ranker.add_label(url, resp)
            url_deques[task_name].remove(url)

        # label next
        return flask.redirect(flask.url_for(
            'label_task',
            task_name=task_name))


@app.route('/<task_name>/label/skip/<path:url>/')
def skip_url(task_name, url):
    logger.info(f'skipping: {url} for {task_name}')
    skipped[task_name].add(url)
    url_deques[task_name].remove(url)
    return flask.redirect(flask.url_for('label_task', task_name=task_name))


@app.route('/<task_name>/label/recalc/')
def recalc(task_name):
    logger.info(f'recalculating: {task_name}')
    ranker = get_ranker(task_name)
    ranker.rerank_jobs()
    return flask.redirect(flask.url_for('label_task', task_name=task_name))


@app.route('/source_posting/<path:url>')
def source_posting(url):
    return requests.get(url, headers=HEADERS).text

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
