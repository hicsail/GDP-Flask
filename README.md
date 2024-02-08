# GDP-Flask

This repo includes three different instances: `scraper`, `translator`, `classifier`. GitHub action usually fail right after push. Manually re-run the action will solve the problem.

## Scraper

It will scrape articles on Ministry of Commerce's website based on given search terms. Currently, the frequency of the scraper is every 15 minutes. It can be modified at `scraper/app.py` at `line 171`.

Based on PI's request, the frequency of the finalized version should be the first day of January and July. This line of code is currently commented out in the same file at `line 170`.

## Translator

It has both DeepL and Google Translate. The DeepL is mainly used for testing purposes to avoid spending quotas for finalized version. The finalized version should be using Google Translate.

## Classifier

It is only a script for pulling articles from the database, sending it to the LLM, and updating the database record. If there are articles that need to be verified, it will keep running. If no more articles to be verified, it is scheduled to look for articles in the database every hour.
