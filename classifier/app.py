from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from datetime import datetime
from ollama import Client, ChatResponse
import os
import requests
from pydantic import BaseModel


class LLMOutput(BaseModel):
    overall_score: int
    overall_justification: str
    score_a: int
    score_b: int
    score_c: int
    score_d: int

app = Flask(__name__)
scheduler = BackgroundScheduler()

client = Client(
  host=os.getenv("LLM_URL"),
)

prompt = """Conduct an evaluation of the provided content using the following rubric.
Assess whether the headline and body indicate financial activities where a Chinese financial institution
is acting as the lender. Evaluate each factor separately, providing a score from 1-5 and a justification for it. Keep the justification concise, up to 25 words.

Scoring Rubric:

A. Recipient
Score 5: Clearly identifies a recipient of the loan, either by explicitly stating the loan is received or by indicating that a project is being undertaken in a specific country, implying it as the borrower/host.
Score 3: Mentions a country or entity in connection with financing but does not clearly establish it as the loan recipient.
Score 1: No recipient is mentioned, or there is no clear indication of who is receiving the loan.


B. Chinese Lender 
Score 5: The content explicitly names a Chinese financial institution, such as the Import-Export Bank of China (Exim bank) or the China Development Bank (CDB), as the lender in a loan or financial agreement.
Score 3: A Chinese lender or actor is mentioned in connection with financial activities, but its role as the lender in the transaction is unclear or speculative.
Score 1: No reference to a Chinese funder or the given project is associated with non-Chinese lenders such as the World Bank or another country’s finance institution.


C. Financial Instrument
Score 5: Clearly identifies the transaction as a loan, using terms like lending, borrowing, debt, or loan.
Score 3: Mentions financing or general support but does not explicitly confirm it as a loan. The description is vague regarding the financial instrument.
Score 1: Indicates the transaction is not a loan, instead describing it as an equity investment, grant, or donation.


D. Activity Precision
Score 5: Specifies a clear infrastructure project or a defined cooperation model where China and the recipient work together on development, such as infrastructure or a technical agreement. If the recipient is a bank, it is stated that the facility is for on-lending purposes.
Score 3: Refers to general development support without specifying a particular project or cooperation model. Or refers to a specific activity such as infrastructure development but does the connection with Chinese financing is not clear.
Score 1: Mentions China’s support but provides no details on the specific activity or project involved.
"""


@app.route('/health')
def health_check():
    return 'healthy'

def classify():
    print("[MOF Classifier] Classifying started at " + datetime.now().isoformat() + "\n")

    while True:
        db_url = os.getenv("NOCO_DB_URL")
        headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
        params = {
            "fields": "Id,originalTitle,translatedTitle,originalContent,translatedContent,originalOutlet,translatedOutlet,isEnglish,AIScore,originalLanguage",
            "where": "(AIScore,is,null)~and(isEnglish,eq,true)",
            "limit": 50
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
                    response: ChatResponse = client.chat(model='deepseek-r1:latest', messages=[
                        {
                            'role': 'system',
                            'content': prompt,
                        },
                        {
                            'role': 'user',
                            'content': f'Headline: {article["originalTitle"] if originalEnglish else article["translatedTitle"]}\n\nBody: {article["originalContent"] if originalEnglish else article["translatedContent"]}',
                        }
                    ],
                    format=LLMOutput.model_json_schema(),
                    stream=False)

                    try :
                        text = response['message']['content']
                        score_a = text.split('"score_a":')[1].split(',')[0]
                        score_b = text.split('"score_b":')[1].split(',')[0]
                        score_c = text.split('"score_c":')[1].split(',')[0]
                        score_d = text.split('"score_d":')[1].split('}')[0]
                        score = (int(score_a) + int(score_b) + int(score_c) + int(score_d)) / 4

                        print("[MOF Classifier] Score: " + str(score))
                        article["AIScore"] = score
                        requests.patch(db_url, headers=headers, json=article)
                        print("Article classified: " + article["originalTitle"])
                    except Exception as e:
                        print(e)
                        article["AIScore"] = -1

                    break
                except Exception as e:
                    print(e)
                    attempts -= 1
                    print(f"[MOF Classifier] Request timeout. {attempts} attempt(s) left ...")
                    article["AIScore"] = -1



if __name__ == '__main__':
    load_dotenv()
    scheduler.add_job(classify, "cron", hour="*", minute="*/1", max_instances=1)
    scheduler.start()
    print("Classifier schedule started")
    app.run(port=5003)
