# What : 
Jobs-ads relevance learner, ranker, deduplicator, and scraper.

# Why: 
Job-ads are long and confusing and contain too little signal to noise (at least in Australia): 
- the companies are "obfuscated".
- same job may be posted by multiple agencies / recruiters with minor differences
- jobs will be reposted as new after a while (after editting).
- the descriptions are long, annoying to read, and have no standards.
 
This creates a strong feeling of deja-vu, discomfort, and waste of time. 

Automation and basic ML can help reduce the job-ad reading to a minimum.       

# How:

## Scraping: 
- [Scrapy](https://scrapy.org/) is used for scraping some relevant jobs from au.jora.com 
(a job aggregator in Australia). Scraping machinery was forked initially from 
 [scrapy demo repo](https://github.com/scrapinghub/spidyquotes).
- Scraping and labeling is done periodically (on command) and old job-ads and labels are used
for ranking and deduplication of future scrapes.

## Labeling relevant jobs:
- The jobs are initially presented ordered by a simple heuristic using 
user defined positive and negative keywords (in title and description).
- Initially there are too few labels, so the jobs are ranked by the heuristic.
- When the model starts being somewhat accurate the jobs are ranked by the model.
- The ads are presented one-by-one with their basic features and their URL, 
and the user is asked to label the relevance.

## Machine Learning:
- Relevance learning: TFIDF features, salary guess model output, keywords score -> Simple RF model.
- Deduplication: TFIDF features (for uncommon n-grams) -> Cosine similarty -> Hand-tuned threshold (may not be ideal for your dataset).
- Salary guess (used as a feature): TFIDF + user keywords score -> RF.

# Installation:
## Local:
1. Clone repo.
2. Go to the repo folder and create your favorite kind of virtual environment 
(e.g. `python -m venv . && source ./bin/activate`)
3. `pip install -r requirements.txt`

## Docker:
1. Make a folder for persisting your data between the scraping runs. E.g. `~/jobs_data`
2. In the instructions below instead of running `python scrape_and_label.py ...` 
run `docker run --rm -it -v $(realpath ./jobs_data):/jobs_rank/data artdgn/jobs_ranker ...` 

# Running

## Workflow:
1. Create your "task" JSON definition. Use the format in `tasks/example-task.json` 
and edit the fields:
    - Get your search-url (by going to the job-ads site, using the filters and copying the URL).
    - Edit your positive and negative keywords for the initial ranking.   
2. (Optional) Run `python scrape_and_label.py --help` to read about the possible flags.
3. Run `python scrape_and_label.py --scrape` and choose or provide your task at 
the first prompt or run `python scrape_and_label.py -t your-task-name --scrape` 
where `your-task-name`:
    - is the name of the JSON file 
    - or a path to it if you didn't put it in the `tasks/` folder.
    - or if you're running in docker, put the file itself in `~/jobs_data` folder, 
and specify the path using `-t` flag or on task prompt as `./data/your-task-name.json` 
(i.e. `./data/` is the mounted volume in the docker container) 
4. Wait for it to finish scraping and follow the prompts for labeling. 
    - You're shown basic attirbutes (title, age, salary etc..) and the URL, 
    click the URL (or right-click and open).
    - Label `n` for "irrelevant garbage", `y` for "give me more of those" and use floats 
    between 0 and 1 for "it's kinda ok" (e.g `0.1`, `0.5` etc..). 
    - use `recalc` if you've edited keywords in your task file, 
    or you want to retrain the model on the new labels that you've just added.    
5. Stop when you've had enough joy for one day. You can resume labeling the same batch by 
running `python scrape_and_label.py -t your-task-name` (without the `--scrape` flag!). 

Once you're done (or remaining unlabeled jobs are complete garbage) just stop.

That's it, now just wait for some time for new jobs to be added and run with the `--scrape` flag again 
to get a new batch. Your labels from previous batches will be used for ranking and all 
the previous scrapes will be used for deduplication.

#### Undo?
- Accidental / wrong labels: there is no undo, so if you want to change any of the labels - 
just edit `data/labeled/your-task-name.csv`.
- Accidental scrape: if you've set the `-s / --scrape` flag and you didn't 
mean to, you won't be shown jobs from the previous scrape (which you might still 
want to label). If this happened, don't worry, you can stop the scrape and 
remove the last csv file with that day's date from `./data/crawls/your-task-name/` - 
it's as if it never happened. If you did multiple scrapes on the same day for the same task (how desparate are you?)
than you don't need to remove it, because all of it will be used to show new jobs.  
- Docker warning: if running in docker, be mindful 
that you'll need to `chown` the files or `sudo` edit them, because anything 
created inside docker will be owned by root.


# Possible questions:
1. Q: Other job sites? 
A: You'll need to create a different scraper(spider) object and tune the scraping to suit that site. 
Refer to the [scrapy docs](https://docs.scrapy.org/en/latest/) 
and [selector gadget](https://selectorgadget.com/)
2. Q: Why not Deep Learning, where's ELMO / BERT / GPT2 / $LATEST_NLP_MIRACLE? A: The main goal was to simplify job seeking 
for myself. Also my guess is that there is little 
to gain from sophsticated models because for the initial relevance task a job-ad is 
often just a collection of important keywords embedded in unimportant sentences and paragraphs. 
The actual "apply" decision task requires much more personal context, world-knowledge, 
and good old brain thinking for it be an easy task to solve with ML.       
3. Q: Your modeling approach / code is stupid and it sucks.. A: Any ideas, 
suggestions, collaborataion, etc are welcome (also, there's a todo.md file with possible ideas)  