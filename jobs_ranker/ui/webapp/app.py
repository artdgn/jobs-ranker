import flask

from jobs_ranker.tasks.configs import TasksConfigsDao
from jobs_ranker.ui.webapp.task_sessions import TasksSessions
from jobs_ranker.utils.logger import logger, log_path

app = flask.Flask(__name__)
app.secret_key = b'secret'

tasks = TasksSessions()


@app.route('/')
def home():
    flask.flash(f'redirected to tasks-list', 'info')
    return flask.redirect(flask.url_for('tasks_list'))


@app.route('/')
@app.route('/tasks/')
def tasks_list():
    return flask.render_template('tasks_list.html',
                                 task_names=TasksConfigsDao.all_names())


@app.errorhandler(404)
def not_found(message):
    return flask.render_template(
        'error.html',
        message=message,
        back_url=flask.url_for('home'),
        back_text='Go back to start..'), 404


@app.route('/<task_name>/')
def task_description(task_name):
    task = tasks[task_name]

    return flask.render_template('task_page.html',
                                 task_name=task_name,
                                 config_data=str(task.get_config()))


@app.route('/<task_name>/edit', methods=['GET', 'POST'])
def edit_task(task_name):
    task = tasks[task_name]
    if flask.request.method == 'GET':

        if task.recent_edit_attempt:
            flask.flash(f'continuing previous edit, '
                        f'press "reset" to discard', 'info')
            text_data = task.recent_edit_attempt
        else:
            text_data = str(task.get_config())

        return flask.render_template('edit_task.html',
                                     task_name=task_name,
                                     text_data=text_data)
    else:  # post
        form = flask.request.form

        if form.get('reset'):
            task.recent_edit_attempt = None
            flask.flash(f'resetting and discarding changes', 'warning')
            return flask.redirect(flask.url_for('edit_task', task_name=task_name))

        try:
            task.update_config(form.get('text'))
            flask.flash(f'Task edit succesful!', 'success')
            return flask.redirect(flask.url_for('reload_ranker', task_name=task_name))

        except ValueError as e:
            message = str(e)
            flask.flash(f'Task edit error: {message}', 'danger')
            return flask.redirect(flask.url_for('edit_task', task_name=task_name))


@app.route('/tasks/new_task/', methods=['GET', 'POST'])
def new_task():
    back_url = flask.url_for('tasks_list')
    if flask.request.method == 'GET':
        return flask.render_template('new_task.html', back_url=back_url)
    else:
        form = flask.request.form
        name = form.get('name')
        try:
            TasksConfigsDao.new_task(name)
            return flask.redirect(flask.url_for('edit_task', task_name=name))
        except ValueError as e:
            flask.flash(str(e), 'danger')
            return flask.redirect(flask.url_for('new_task'))


def _ranker_busy_page(task_name):
    refresh_delay = 5
    flask.flash(f'auto-refreshing every {refresh_delay} seconds', 'info')
    return flask.render_template(
        'waiting.html',
        title='labeling',
        message='Please wait: calculating ranking',
        seconds=refresh_delay,
        task_name=task_name)


@app.route('/<task_name>/label/')
def labeling(task_name):
    task = tasks[task_name]
    task.load_ranker()

    if task.ranker.busy:
        return _ranker_busy_page(task_name)

    url = task.get_url()

    if url is None:
        return flask.render_template(
            'error.html',
            message=(f'No more new unlabeled jobs for task "{task_name}", '
                     f'try with dedup off, or scraping new jobs'),
            back_url=flask.url_for('task_description', task_name=task_name),
            back_text=f'Back:')
    else:
        flask.flash(f'Ranking based on "{task.ranker.sort_col}" '
                    f'({task.ranker.ranking_scores})', 'info')
        # go label it
        return flask.redirect(
            flask.url_for('label_url_get', task_name=task_name, url=url))


@app.route('/<task_name>/label/<path:url>/', methods=['GET'])
def label_url_get(task_name, url):
    task = tasks[task_name]
    task.load_ranker()

    if task.ranker.busy:
        return _ranker_busy_page(task_name)

    if task.ranker_outdated():
        reload_url = flask.url_for('reload_ranker', task_name=task_name)
        logger.info(f'ranker outdated for "{task_name}" (new data scraped)')
        flask.flash(flask.Markup(
            f'New data scraped, "Reload" to update: '
            f'<a href="{reload_url}" class="alert-link">{reload_url}</a>'), 'success')

    url_attributes, raw_description = task.ranker.url_data(url)
    url_att_html = (url_attributes.drop(['url', 'title']).
                    to_frame().to_html(header=False, justify='right'))

    return flask.render_template(
        'job_page.html',
        task_name=task_name,
        job_url=url,
        job_title=url_attributes.get('title'),
        job_description=raw_description,
        url_data=url_att_html
    )


@app.route('/<task_name>/label/<path:url>/', methods=['POST'])
def label_url_post(task_name, url):
    task = tasks[task_name]
    task.load_ranker()

    form = flask.request.form
    if 'skip' in form:
        return flask.redirect(flask.url_for('skip_url', task_name=task_name, url=url))

    if 'recalc' in form:
        return flask.redirect(flask.url_for('recalc', task_name=task_name))

    if 'numeric' in form:
        resp = form['label']
    elif 'no' in form:
        resp = task.ranker.labeler.neg_label
    elif 'yes' in form:
        resp = task.ranker.labeler.pos_label
    elif 'somewhat' in form:
        resp = '0.3'
    else:
        resp = form['label']

    if not task.ranker.labeler.is_valid_label(str(resp)):
        # bad input, render same page again
        flask.flash(f'not a valid input: "{resp}", please relabel (or skip)', 'danger')
        return flask.redirect(flask.url_for('label_url_get', task_name=task_name, url=url))
    else:
        task.add_label(url, resp)
        flask.flash(flask.Markup(
            f"""labeled "{resp}" for <a href="{url}" 
            class="alert-link" target="_blank"=>{url}</a>"""),
            'success')

    # label next
    return flask.redirect(flask.url_for('labeling', task_name=task_name))


@app.route('/<task_name>/label/skip/<path:url>/')
def skip_url(task_name, url):
    task = tasks[task_name]
    task.skip(url)
    logger.info(f'skip: {url} for "{task_name}"')
    flask.flash(f'skipped url {url} for "{task_name}"', 'warning')
    return flask.redirect(flask.url_for('labeling', task_name=task_name))


@app.route('/<task_name>/label/recalc/')
def recalc(task_name):
    task = tasks[task_name]
    task.recalc()
    logger.info(f'recalculating: {task_name}')
    flask.flash(f're-calculating rankings for task "{task_name}"', 'info')
    return flask.redirect(flask.url_for('labeling', task_name=task_name))


@app.route('/<task_name>/reload/')
def reload_ranker(task_name):
    task = tasks[task_name]
    if task.crawling:
        logger.info(f'not reloading ranker data because '
                    f'scraping is in progress: {task_name}')
        flask.flash(f'Scraping in progress, not reloading data.', 'warning')
    if task.ranker.busy:
        logger.info(f'not reloading ranker data because '
                    f'ranker is busy (reloading or recalculating): {task_name}')
        flask.flash(f'Ranker is busy, not reloading data.', 'warning')
    else:
        task.reload_ranker()
        logger.info(f'reloading ranker: {task_name}')
        flask.flash(f're-loading data for task "{task_name}"', 'info')
    return flask.redirect(flask.url_for('task_description', task_name=task_name))


@app.route('/<task_name>/scrape/')
def scraping(task_name):
    task = tasks[task_name]

    try:
        days_since_last = task.days_since_last_crawl()
    except FileNotFoundError:
        days_since_last = None
        flask.flash(f'no crawl data found for task (is this a new task?)', 'warning')

    if task.crawling:
        n_jobs = task.jobs_in_latest_crawl() or 0
        expected = task.expected_jobs_per_crawl()
        refresh_delay = 30
        flask.flash(f'auto-refreshing every {refresh_delay} seconds', 'info')
        return flask.render_template('scraping_page.html',
                                     task_name=task_name,
                                     n_scraped=n_jobs,
                                     expected=expected,
                                     seconds=refresh_delay)

    return flask.render_template('scraping_page.html',
                                 task_name=task_name,
                                 days_since_last=days_since_last)


@app.route('/<task_name>/scrape/start')
def scrape_start(task_name):
    task = tasks[task_name]
    task.start_crawl()
    logger.info(f'started scraping for {task_name}')
    flask.flash(f'Started scraping for task "{task_name}"', 'success')
    return flask.redirect(flask.url_for('scraping', task_name=task_name))


@app.route('/<task_name>/labels_history')
def labels_history(task_name):
    task = tasks[task_name]
    task.ranker.labeler.load()
    table = task.ranker.labeler.export_html_table()
    return flask.render_template('rawtext_or_html.html', html=table)


@app.route('/<task_name>/scrapes_history')
def scrapes_history(task_name):
    task = tasks[task_name]
    table = task.all_crawls_lengths().to_html(
        index=False, classes="table-sm table-bordered table-hover table-striped")
    return flask.render_template('rawtext_or_html.html', html=table)


@app.route('/log')
@app.route('/logs')
def server_logs():
    with open(log_path) as log_file:
        return flask.render_template('rawtext_or_html.html', text=log_file.read())


def start_server(debug=False, port=None):
    app.run(host='0.0.0.0', debug=debug, port=port)


if __name__ == '__main__':
    start_server(debug=True, port=5001)
