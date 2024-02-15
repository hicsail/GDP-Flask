from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

import os
import requests
import json

app = Flask(__name__)
scheduler = BackgroundScheduler()

@app.route('/health')
def health_check():
    return 'healthy'

def classify():
    scheduler.pause()

    while True:
        db_url = os.getenv("NOCO_DB_URL")
        headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
        params = {
            "where": "(status,eq,unverified)~and(isEnglish,eq,false)",
            "limit": 5,
        }
        
        llm_url = os.getenv("LLM_URL")
        form_data = {
            "variables": "headline,body",
            "string_prompt": "<<SYS>> \n You are an assistant tasked with classifying whether the given headline and body is related to China Development Bank or China Import Export Bank or Debt or Loan From China. \n <</SYS>> \n\n [INST] Generate a SHORT response if the given headline and article body is related to China. The output should be either Yes or No. \n\n Headline: \n\n {headline} \n\n Body: {body}\n\n [/INST]",
        }

        articles = requests.get(db_url, headers=headers, params=params)
        articles = articles.json()
        if articles.get("pageInfo").get("totalRows") == 0:
            print("[MOF Classifier] No articles to classify")
            scheduler.resume()
            break

        for article in articles.get("list"):
            attempts = 3
            while attempts > 0:
                try:
                    print("Article to classify: " + article["originalTitle"])
                    form_data["llm_request"] = json.dumps({
                        "headline": article["originalTitle"],
                        "body": article["originalContent"]
                    })

                    res = requests.post(llm_url, data=form_data)

                    if "yes" in res.json().get("Result").lower()[:5]:
                        article["status"] = "relevant"
                    else:
                        article["status"] = "irrelevant"

                    break
                except:
                    attempts -= 1
                    article["status"] = "undetermined"

            requests.patch(db_url, headers=headers, json=article)

if __name__ == '__main__':
    load_dotenv()
    scheduler.add_job(classify, "cron", hour="*", minute=5)
    scheduler.start()
    app.run(port=5003)