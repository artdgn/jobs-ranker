# What : 
Jobs-ads relevance learner, ranker, deduplicator, and scraper. Webapp or command line usage.  

# Why: 
Job-ads are long and confusing and contain too little signal to noise (at least in Australia): 
- the companies are "obfuscated".
- same job may be posted by multiple agencies / recruiters with minor differences, 
or will be reposted with minor changes as new after a while.
- the descriptions are low quality, bloated, and non distinct.

Automation and basic ML can help reduce the job-ad reading and mental fatigue to a minimum.       

# Quick start (docker image):
1. `mkdir ~/jobs_data` (to store data between uses) 
2. `docker run -v $(realpath ~/jobs_data):/data -p 5000:5000 artdgn/jobs-ranker python server.py`
3. Open http://localhost:5000/ in browser.
4. Create and edit a task, then scrape it, then view and label jobs.

# Installation: more details / other options:

### Docker (downloaded image):
1. Make a folder for persisting your data between the scraping runs. E.g. `~/jobs_data`
2. `docker pull artdgn/jobs-ranker`

### Docker local (to build the image yourself):
1. Clone repo.

### Local virtual environment (for development / tinkering):
1. Clone repo.
2. `make install` (Creates a virtual environment and installs dependencies in it) 

# Usage

## Webapp:
This will run a local website through which the application is used. 
1. Run: 
    - Docker: To run server in background `docker run -dit 
    -v $(realpath ~/jobs_data):/data -e TZ=$(cat /etc/timezone) --name jobs-ranker 
    --restart unless-stopped -p 5000:5000 artdgn/jobs-ranker python server.py`
    Check the `make docker-server` (webapp) or `make docker-bash` (CLI) commands in 
    the `Makefile` of this project for recommended arguments.
    - Local docker: `make docker-server`.
    - Local virtual environment: `make server`.  
This will launch local flask development(!) server on port 5000. 
The server is listening to the network in case you want 
to use it from other devices on your wifi network.
 
2. Go to http://localhost:5000/ (on the same machine) and navigate the links to:
    1. Create and define your task: edit the the search urls (jora only for now) and the positive and 
    negative keywords for the initial heuristic ranking.
    2. Navigate to your task and and start a scrape. It will take a while. 
    3. Navigate to "View jobs and Label" and label jobs until 
    the jobs are mostly irrelevant, teach the ranker what you like.  
    4. Wait for new jobs and repeat (scrape, label).

## CLI (console / text interface):
This will run a CLI (text interface) which will present the ranked jobs in an interactive manner.  
1. Create your "task" definition. Local: copy or rename `tasks/example-task.json`. 
Docker: put the file in `your-data-dir/tasks/` dir, e.g. `~/jobs_data/tasks/my_task.json`). 
Edit the fields:
    - Get your search-urls (by going to the job-ads site, 
    using the filters and copying the resulting URLs).
    - Edit your positive and negative keywords for the initial ranking.
2. Run local docker: `make docker-bash`. Or activate local virtual environment: `. venv/bin/activate`. 
Or run just the image: `docker run -it -v $(realpath ~/jobs_data):/app/data artdgn/jobs-ranker bash`
3. (Optional) Run `python console.py --help` to read about the possible parameters / options.
4. Run `python console.py --scrape` and choose or provide your task from step 1 at 
the first prompt. (Alternatively provide it as an argument using the `-t` option).
5. Wait for it to finish scraping and follow the prompts for labeling. 
    - You're shown basic attributes (title, age, salary etc..) and the URL, 
    click the URL (or right-click and open).
    - Label `n` for "irrelevant garbage", `y` for "give me more of those" and use numbers 
    between 0 and 1 for "it's kinda ok" (e.g `0.1`, `0.5` etc..). 
    - use `recalc` if you've edited keywords in your task file, 
    or you want to retrain the model on the new labels that you've just added.    
6. Stop when you've had enough for one day. You can resume labeling the same batch by 
running `python console.py -t your-task-name` (without the `--scrape` flag!). 
7. Once you're done (or remaining unlabeled jobs are mostly garbage) just stop.
That's it, now just wait for some time for new jobs to be added and run with the `--scrape` flag again 
to get a new batch. Your labeled jobs from previous batches will be used for ranking and deduplication.


### Undo?
- Accidental / wrong labels: to change any of the labels - 
just edit `data/labeled/your-task-name.csv`.
- Accidental scrape: if you've triggered a scrape and you didn't 
mean to, you won't be shown jobs from the previous scrape (which you might still 
want to view / label) unless they're also in the new one or the scrapes 
were done on the same day. If this happened, don't worry, 
you can stop the scrape and remove the last csv file with that day's date 
from `data/crawls/your-task-name/` -  it's as if it never happened.  
- Docker warning: if running in docker, be mindful 
that you'll need to `chown` the files or `sudo` edit them, because anything 
created inside docker will be owned by root.


# How this works:

## Scraping: 
- [Scrapy](https://scrapy.org/) is used for scraping some relevant jobs from au.jora.com 
(a job aggregator in Australia). Scraping machinery was forked initially from 
 [scrapy demo repo](https://github.com/scrapinghub/spidyquotes).
- Scraping and labeling is done periodically on request and old job-ads and labels are used
for ranking and deduplication of future scrapes.

## Initial ranking:
- First, the jobs are presented ordered by a simple heuristic using 
user defined positive and negative keywords (in title and description).
- When there are enough labels to train a model that is a better 
ranker than the heuristic - the model used for ranking.
- The ads are presented one-by-one and the user is asked to 
label the relevance of each one (or skip).

## Machine Learning:
- Relevance learning: TFIDF features, salary guess model output, keywords score -> Simple RF model.
- Deduplication: TFIDF features (for uncommon n-grams) -> Cosine similarty -> Hand-tuned threshold (may not be ideal for your dataset).
- Salary guess (used as a feature): TFIDF + user keywords score -> RF.

# Possible questions:
1. Q: Other job sites? 
A: You'll need to create a different scraper(spider) to suit that site. 
Refer to the [scrapy docs](https://docs.scrapy.org/en/latest/) 
and [selector gadget](https://selectorgadget.com/)
2. Q: Why not Deep Learning, where's ELMO / BERT / GPT2 / $LATEST_NLP_MIRACLE? A: My guess is that there is little 
to gain from sophisticated models because for the initial relevance screening, job-ads are 
often just a collection of keywords anyway (surrounded by unimportant sentences and paragraphs). 
And the actual "apply or not" decision requires much more thought and context to solve.       
3. Q: Your modeling approach / code is stupid and it sucks.. A: Any ideas, 
suggestions, collaborataion, etc are welcome (also, there's a todo.md file with possible ideas)  