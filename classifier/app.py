from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from datetime import datetime
from ollama import Client, ChatResponse
import os
import requests
import json

app = Flask(__name__)
scheduler = BackgroundScheduler()

client = Client(
  host=os.getenv("LLM_URL"),
)

@app.route('/health')
def health_check():
    return 'healthy'

def classify():
    print("[MOF Classifier] Classifying started at " + datetime.now().isoformat() + "\n")

    while True:
        db_url = os.getenv("NOCO_DB_URL")
        headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
        params = {
            "where": "(status,eq,unverified)~and(isEnglish,eq,true)",
            "limit": 5,
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
            break

        for article in articles.get("list"):
            attempts = 2
            while attempts > 0:
                try:
                    originalEnglish = article["originalLanguage"] == "en"
                    print("[MOF Classifier] Classifying article: " + article["originalTitle"])
                    response: ChatResponse = client.chat(model='llama3.2', messages=[
                        {
                            'role': 'system',
                            'content': 'You are an assistant tasked with classifying whether the given headline and body is related to financial activities involving Chinese financial institutions. Specifically, the content should be marked as relevant if it involves:'
                                       '1. A Chinese lender, including the Import Export Bank or Development Bank.'
                                       '2. A loan being signed, including agreements, ceremonies, or mentioning of loan amounts.'
                                       '3. Information related to loans or debts.'
                                       '4. Interest from other Chinese banks in financial activities.'
                                       'Generate a short response indicating whether the content meets any of the above criteria. '
                                       'Respond with "Yes" for relevance or "No" if not.',
                        },
                        {
                            'role': 'user',
                            'content': f'Headline: {article["originalTitle"] if originalEnglish else article["translatedTitle"]}\n\nBody: {article["originalContent"] if originalEnglish else article["translatedContent"]}',
                        }
                    ])

                    print(response['message']['content'])

                    #
                    #if "yes" in res.json().get("Result").lower()[:3]:
                    #    article["status"] = "relevant"
                    #elif "no" in res.json().get("Result").lower()[:2]:
                    #    article["status"] = "irrelevant"
                    #else:
                    #    article["status"] = "undetermined"
#
                    break
                except e:
                    print(e)
                    attempts -= 1
                    print(f"[MOF Classifier] Request timeout. {attempts} attempt(s) left ...")
                    article["status"] = "undetermined"

            #requests.patch(db_url, headers=headers, json=article)
            #print("Article classified: " + article["originalTitle"])

if __name__ == '__main__':
    load_dotenv()
    scheduler.add_job(classify, "cron", hour="*", minute="*/1", max_instances=1)
    scheduler.start()
    print("Classifier schedule started")
    app.run(port=5003)
