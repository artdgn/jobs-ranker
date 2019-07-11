FROM python:3.6

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV APP_DIR=/job_scraping

ADD . ${APP_DIR}

WORKDIR ${APP_DIR}

ENTRYPOINT ["python", "scrape_and_label.py"]

CMD ["--help"]

# docker build --pull -t artdgn/jobs-recommender .
# docker push artdgn/jobs-recommender
# docker run --rm -it -v $(realpath ./data):/jobs-recommender/data artdgn/jobs-recommender --help