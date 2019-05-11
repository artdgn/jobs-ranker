FROM python:3.6

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV APP_DIR=/job_scraping

ADD . ${APP_DIR}

WORKDIR ${APP_DIR}

ENTRYPOINT ["python", "scrape_and_label.py"]

CMD ["--help"]

# docker build --pull -t artdgn/job_scraping .
# docker push artdgn/job_scraping
# docker run --rm -it -v $(realpath ./data):/job_scraping/data artdgn/job_scraping -t example-task