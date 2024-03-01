from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from datetime import datetime

import os
import requests
import json

app = Flask(__name__)
scheduler = BackgroundScheduler()

@app.route('/health')
def health_check():
    return 'healthy'

def classify():
    print("[MOF Classifier] Classifying started at " + datetime.now().isoformat() + "\n")
    scheduler.pause()

    while True:
        db_url = os.getenv("NOCO_DB_URL")
        headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
        params = {
            "where": "(status,eq,unverified)~and(isEnglish,eq,true)",
            "limit": 1,
        }

        llm_url = os.getenv("LLM_URL")
        form_data = {
            "variables": "headline,body",
            "string_prompt": '<<SYS>>\n You are an assistant tasked with classifying whether the given headline and body is related to financial activities involving Chinese financial institutions. Specifically, the content should be marked as relevant if it involves:\n 1. A Chinese lender, including the Import Export Bank or Development Bank.\n 2. A loan being signed, including agreements, ceremonies, or mentioning of loan amounts.\n 3. Information related to loans or debts.\n 4. Interest from other Chinese banks in financial activities.\n Generate a short response indicating whether the content meets any of the above criteria. Respond with "Yes" for relevance or "No" if not.<</SYS>> \n\n [INST]Assess the given headline and article body based on the specified criteria. Provide a concise response indicating relevance.\n\n Headline: {headline} \n\n Body: {body} \n\n [/INST]',
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
                    form_data["llm_request"] = json.dumps({
                        "headline": article["originalTitle"],
                        "body": article["originalContent"]
                    })

                    res = requests.post(llm_url, data=form_data, timeout=180)

                    if "yes" in res.json().get("Result").lower()[:5]:
                        article["status"] = "relevant"
                    else:
                        article["status"] = "irrelevant"

                    break
                except:
                    attempts -= 1
                    article["status"] = "undetermined"

            requests.patch(db_url, headers=headers, json=article)
            print("Article classified: " + article["originalTitle"])

if __name__ == '__main__':
    load_dotenv()
    scheduler.add_job(classify, "cron", hour="*", minute="*/5")
    scheduler.start()
    print("Classifier schedule started")
    app.run(port=5003)
