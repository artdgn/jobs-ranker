scraping parts forked and adjusted from http://quotes.toscrape.com ([github repo](https://github.com/scrapinghub/spidyquotes)).


#### Running

    python scrape_and_label.py --scrape --delay 10 --recalc


#### ideas
    summary / sentiment:
        https://elitedatascience.com/python-nlp-libraries
        https://www.reddit.com/r/datascience/comments/8qde2g/sentiment_analysis_in_python_any_pretrained_models/
        http://nlp.town/blog/off-the-shelf-sentiment-analysis/  
    model improvement:
        only user NER words
        w2v / send2vec
        sefl-supervision (title prediction) pretraining          
    use AWS free tier to host interactive / django frontend (https://aws.amazon.com/free/faqs/?ft=n)
        deploy in jupyter notebook on the cloud (storage in S3)
        web frontend (django)
    find a way to handle the imbalance (weights for regression?)    
    separate tuning and refitting
    other sites (indeed, linked in)
