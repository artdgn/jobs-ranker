import json

import flask
import requests

from common import HEADERS
from inputs.webapp.task_contexts import TasksContexts
from tasks.dao import TasksConfigsDao
from utils.logger import logger

app = flask.Flask(__name__)
app.secret_key = b'secret'

tasks = TasksContexts()


@app.route('/')
def instructions():
    flask.flash(f'redirected to tasks-list')
    return flask.redirect(flask.url_for('tasks_list'))


@app.route('/tasks/')
def tasks_list():
    tasks = TasksConfigsDao.tasks_in_scope()
    task_urls = [{'name': t,
                  'url': flask.url_for('task_description', task_name=t)}
                 for t in tasks]
    return flask.render_template('tasks_list.html', task_urls=task_urls)


@app.errorhandler(404)
def not_found(message):
    return flask.render_template(
        'error.html',
        message=message,
        back_url=flask.url_for('instructions'),
        back_text='Go back to start..'), 404


@app.route('/<task_name>/')
def task_description(task_name):
    task = tasks[task_name]
    task.load_ranker()

    return flask.render_template(
        'task_page.html',
        task_name=task_name,
        back_url=flask.url_for('tasks_list'),
        scrape_url=flask.url_for('scrape_task', task_name=task_name),
        label_url=flask.url_for('label_task', task_name=task_name),
        reload_url=flask.url_for('reload_ranker', task_name=task_name),
        task_data=json.dumps(task.get_task_config(), indent=4))


@app.route('/<task_name>/label/')
def label_task(task_name):
    task = tasks[task_name]
    ranker = task.get_ranker()
    task.load_ranker()

    if ranker.busy:
        return flask.render_template(
            'waiting.html',
            message='Waiting for labeler to crunch all the data',
            seconds=5)
    else:
        url = task.get_url()

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
    task = tasks[task_name]
    ranker = task.get_ranker()
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
            flask.flash(f'not a valid input: "{resp}", please relabel (or skip)')
            return flask.redirect(flask.url_for(
                'label_url',
                task_name=task_name,
                url=url))
        else:
            ranker.add_label(url, resp)
            task.move_to_next_url()
            flask.flash(f'labeled "{resp}" for ({url})')

        # label next
        return flask.redirect(flask.url_for(
            'label_task',
            task_name=task_name))


@app.route('/<task_name>/label/skip/<path:url>/')
def skip_url(task_name, url):
    task = tasks[task_name]
    task.skip(url)
    logger.info(f'skip: {url} for "{task_name}"')
    flask.flash(f'skipped url {url} for "{task_name}"')
    return flask.redirect(flask.url_for('label_task', task_name=task_name))


@app.route('/<task_name>/label/recalc/')
def recalc(task_name):
    task = tasks[task_name]
    task.recalc()
    logger.info(f'recalculating: {task_name}')
    flask.flash(f're-calculating rankings for task "{task_name}"')
    return flask.redirect(flask.url_for('label_task', task_name=task_name))


@app.route('/<task_name>/reload/')
def reload_ranker(task_name):
    task = tasks[task_name]
    task.reload_ranker()
    logger.info(f'reloading ranker: {task_name}')
    flask.flash(f're-loading data for task "{task_name}"')
    return flask.redirect(flask.url_for('label_task', task_name=task_name))


@app.route('/source_posting/<path:url>')
def source_posting(url):
    return requests.get(url, headers=HEADERS).text


@app.route('/<task_name>/scrape/')
def scrape_task(task_name):
    task = tasks[task_name]

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
