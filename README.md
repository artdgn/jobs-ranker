# What : 
Scraper + Interactive "online learner" + De-duplicator for filtering job-ads.

# Why: 
Job ads are long and confusing and contain too little signal to noise (at least in Australia): 
- the companies are "obfuscated".
- same job may be posted by multiple agencies / recruiters with minor differences
- jobs will be reposted as new after a while (after editting).
- the descriptions are long, annoying to read, and have no standards.
 
This creates a strong feeling of deja-vu, discomfort, and waste of time and peace of mind. 

Automation and basic ML can help reduce the job-ad reading to a minimum.       

# How:

## Scraping: 
- [Scrapy](https://scrapy.org/) is used for scraping some relevant jobs from au.jora.com 
(a job aggregator in Australia) by using the site's filters (in the url). Scraping machinery was forked initially from 
 [scrapy demo repo](https://github.com/scrapinghub/spidyquotes).
- Scraping is done periodically (on command) and old job ads are kept 
for future deduplication and label learning.

## Labeling relevant jobs:
- The jobs are initially presented ordered by a simple heuristic using 
user defined positive and negative keywords (in title and description).
- Initially there are too few labels, so the jobs are ranked by the heuristic.
- When the model starts being somewhat accurate the jobs are ranked by the model.
- The ads are presented one-by-one with their basic features and their URL, 
and the user is asked to label the relevance.

## Machine Learning:
- Relevance learning: TFIDF features, salary guess model output, keywords score -> Simple RF model.
- Deduplication: TFIDF features (differently tuned) -> Cosine similarty -> 
    Cooccurrence propagation -> Hand-tuned threshold (may not be ideal for your dataset).
- Salary guess (used as a feature): TFIDF + user keywords score -> RF.

# Installtion:
## Local:
1. Clone repo.
2. Go to the repo folder and create your favorite kind of virtual environment 
(e.g. `python -m venv . && source ./bin/activate`)
3. `pip install -r requirements.txt`

## Docker:
1. Make a folder for persisting your data between the scraping runs. E.g. `~/jobs_data`
2. In the instructions below instead of running `python scrape_and_label.py ...` 
run `docker run --rm -it -v $(realpath ./data):/job_scraping/data artdgn/job_scraping ...` 

# Running

## Workflow:
1. Create your "task" JSON definition. Use the format in `tasks/example-task.json` 
and edit the fields:
    - Get your search-url (by going to the job ads site, using the filters and copying the URL).
    - Edit your positive and negative keywords for the initial ranking.   
3. (Optional) Run `python scrape_and_label.py` or `python scrape_and_label.py --help` to 
read about the flags and arguments.
4. Run `python scrape_and_label.py -t your-task-name --scrape` where `your-task-name`:
    - is the name of the JSON file 
    - or a path to it if you didn't put it in the `tasks/` folder.
    - or if you're running in docker, put the file itself in `~/jobs_data` folder, 
and specify the path in the docker container `.. -t ./data/your-task-name.json ..`  
5. Wait for it to finish scraping and follow the prompts for labeling. 
    - You get basic description and URL, click the URL (or right-click and open).
    - Label `n` for "irrelevant garbage", `y` for "give me more of those" and use floats 
    between 0 and 1 for "it's kinda ok" (e.g `0.1`, `0.5` etc..). Pick a scale and stick to it. 
    - use `recalc` if you've edited keywords in your task file (to take effect), 
    or you want to retrain the model on the new labels that you've just added.
    - there is no undo, so if you want to change any of the labels - 
    just edit `data/labeled/your-task-name.csv`. (if running in docker, be mindful 
    that you'll need to `chown` the files or `sudo` edit, because anything created in 
    docker will be owned by root)
6. Stop when you've had enough joy for one day. You can resume labeling the same batch by 
running `python scrape_and_label.py -t your-task-name` (without the `--scrape` flag)
7. Once you're done (or remaining unlabeled jobs are complete garbage) just stop.

That's it, now just wait for some time for new jobs to be added and run with the `--scrape` flag again 
to get a new batch. Your labels from previous batches with be used for ranking and all 
the previous scrapes will be used for deduplication.


# Possible questions:
1. Q: Other job sites? 
A: You'll need to create a different scraper object and tune the scraping to suit that site.
2. Q: Why not Deep Learning, where's ELMO and BERT? A: The main goal was to simplify job seeking 
for myself and not an NLP side project. Also my guess is that there is little 
to gain from sophsticated models because for the initial relevance task a job-ad is 
often just a collection of important keywords embedded in unimportant sentences and paragraphs. 
The actual "apply" decision task requires much more personal context, world-knowledge, 
and good old brain thinking for it be an easy task to solve with ML.       
