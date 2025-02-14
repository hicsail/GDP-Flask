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
is acting as the lender. Evaluate each factor separately, make an overall assessment, and then provide a justification for the overall score.
Keep the justification concise, up to 25 words.

Be very strict in the overall assessment score—if any of the required factors is not met, the overall score should be 1.

Scoring Rubric:

A. Chinese Lender in a Loan for Development or Infrastructure Projects
Score 5: The content explicitly names a Chinese financial institution, such as the Import-Export Bank of China
or the China Development Bank, as the lender in a loan or financial agreement.

Score 3: A Chinese lender is mentioned in connection with financial activities,
but its role as the lender in the transaction is unclear or speculative.

Score 1: No reference to a Chinese lender, or only mentions 'foreign banks' or 'international financial institutions'
without specifying China.

B. Loan Agreements and Transactions
Score 5: The content describes a loan agreement or financial transaction where a Chinese lender is providing funds,
including details such as loan amounts, agreements, or signing ceremonies.

Score 3: The content discusses potential loans or ongoing negotiations involving a Chinese lender
but does not confirm a formal agreement.

Score 1: No mention of loan agreements, signing ceremonies, or related transactions involving a Chinese lender.

C. Information on Loans or Debt
Score 5: The content provides detailed information on outstanding loans, repayment terms, or debt obligations
linked to a Chinese lender.

Score 3: The content references loans or debt but lacks specific details about the involvement of a Chinese lender.

Score 1: No mention of loans, debt, or financial obligations involving China.

D. Chinese Financial Institutions Investing in or Extending Credit
Score 5: The content discusses Chinese financial institutions investing in or providing credit to a government,
organization, or entity as part of a financial agreement.

Score 3: There is mention of Chinese economic engagement, but it is unclear whether it directly involves lending
or financial transactions.

Score 1: No indication of Chinese financial institutions participating in lending or financial activities.
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
            "where": "(AIScore,is,null)~and(isEnglish,eq,true)",
            "limit": 50,
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
                    except Exception as e:
                        print(e)
                        article["AIScore"] = -1

                    break
                except Exception as e:
                    print(e)
                    attempts -= 1
                    print(f"[MOF Classifier] Request timeout. {attempts} attempt(s) left ...")
                    article["AIScore"] = -1

            requests.patch(db_url, headers=headers, json=article)
            print("Article classified: " + article["originalTitle"])

if __name__ == '__main__':
    load_dotenv()
    scheduler.add_job(classify, "cron", hour="*", minute="*/1", max_instances=1)
    scheduler.start()
    print("Classifier schedule started")
    app.run(port=5003)
