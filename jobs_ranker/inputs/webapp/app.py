import json

import flask
import requests

from jobs_rank.common import HEADERS
from jobs_rank.inputs.webapp.task_contexts import TasksContexts
from jobs_rank.tasks.dao import TasksConfigsDao
from jobs_rank.utils.logger import logger

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
        task_data=json.dumps(task.get_config(), indent=4))


@app.route('/<task_name>/label/')
def label_task(task_name):
    task = tasks[task_name]
    ranker = task.get_ranker()
    task.load_ranker()
    back_url = flask.url_for('task_description', task_name=task_name)

    if ranker.busy:
        return flask.render_template(
            'waiting.html',
            message='Waiting for labeler to crunch all the data',
            seconds=5,
            back_url=back_url,
            back_text=f'.. or go back: {back_url}')
    else:
        url = task.get_url()

        if url is None:
            return flask.render_template(
                'error.html',
                message=(f'No more new unlabeled jobs for task "{task_name}", '
                         f'try dedup off, or scrape new jobs'),
                back_url=back_url,
                back_text=f'Back: {back_url}')
        else:
            # go label it
            return flask.redirect(flask.url_for(
                'label_url',
                task_name=task_name,
                url=url))


@app.route('/<task_name>/label/<path:url>/', methods=['GET', 'POST'])
def label_url(task_name, url):
    task = tasks[task_name]
    task.load_ranker()
    ranker = task.get_ranker()
    back_url = flask.url_for('task_description', task_name=task_name)

    if ranker.busy:
        return flask.render_template(
            'waiting.html',
            message='Waiting for labeler to crunch all the data',
            seconds=5,
            back_url=back_url,
            back_text=f'.. or go back: {back_url}')

    data = ranker.url_data(url).drop('url')

    if flask.request.method == 'GET':
        return flask.render_template(
            'job_page.html',
            job_url=url,
            url_data=data,
            skip_url=flask.url_for('skip_url', task_name=task_name, url=url),
            recalc_url=flask.url_for('recalc', task_name=task_name),
            back_url=back_url,
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
            task.add_label(url, resp)
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
    return flask.redirect(flask.url_for('task_description', task_name=task_name))


@app.route('/source_posting/<path:url>')
def source_posting(url):
    return requests.get(url, headers=HEADERS).text


@app.route('/<task_name>/scrape/')
def scrape_task(task_name):
    task = tasks[task_name]
    back_url = flask.url_for('reload_ranker', task_name=task_name)

    days_since_last = task.days_since_last_crawl()

    if task.scraping:
        n_jobs = task.jobs_in_latest_crawl() or 0
        return flask.render_template(
            'waiting.html',
            message=(f'Waiting for scraper to finish '
                     f'scraping (crawled {n_jobs} jobs)'),
            seconds=30,
            back_url = back_url,
            back_text = f'Back and reload (will not cancel scrape): {back_url}')

    start_url = flask.url_for('scrape_start', task_name=task_name)

    return flask.render_template(
        'confirm.html',
        message=(f'Are you sure you want to start a scrape? '
                 f'latest crawl is from {days_since_last} day ago'),
        option_1_url=start_url,
        option_1_text=f'Start: {start_url}',
        option_2_url=back_url,
        option_2_text=f'Back and reload data: {back_url}',
    )

@app.route('/<task_name>/scrape/start')
def scrape_start(task_name):
    task = tasks[task_name]
    task.start_scrape()
    logger.info(f'started scraping for {task_name}')
    flask.flash(f'Started scraping for task "{task_name}"')
    return flask.redirect(flask.url_for('scrape_task', task_name=task_name))


def run_app(debug=False):
    app.run(host='0.0.0.0', debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
