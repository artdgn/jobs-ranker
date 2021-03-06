from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='jobs-ranker',
    version='0.1.0',
    description='Jobs-ads relevance learner, ranker, deduplicator, and scraper',
    long_description=open(os.path.join(here, 'README.md'), encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/artdgn/jobs_ranker',
    author='Arthur Deygin',
    author_email='arthurdgn@gmail.com',
    install_requires=open(os.path.join(here, 'requirements.txt')).read().splitlines(),
)
