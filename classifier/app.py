import json

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from datetime import datetime
from ollama import Client, ChatResponse
import os
import requests
from pydantic import BaseModel
from bs4 import BeautifulSoup
import re

AI_SCORE = "AIScore4"
model = "gemma3:12b"

class LLMExtractionA(BaseModel):
    recipient: str

class LLMExtractionB(BaseModel):
    chinese_institution: str

class LLMExtractionC(BaseModel):
    financial_instrument: str

class LLMExtractionD(BaseModel):
    project_or_activity: str

app = Flask(__name__)
scheduler = BackgroundScheduler()

client = Client(
  host=os.getenv("LLM_URL"),
)

prompt = """
Prompt: First, extract the necessary information from the provided content (headline and article title) using the extraction rubric.  Second, conduct an evaluation of the provided headline and article title and the extracted content using the Score Generator rubric. Assess whether the headline and body indicate financial activities where a Chinese financial institution is acting as the lender. Evaluate each factor separately, providing a score from 1-5 and a justification for each. Keep the justification concise, up to 25 words
"""

prompt_a_extraction = """
A. Recipient: Identify and extract the name of the main country that is the main subject of the article (excluding China). If multiple countries/entities are mentioned, list up to the top three most relevant ones. China cannot be one of them.
"""

prompt_b_extraction = """
. Chinese Institution: Identify and extract any mention of a lender, funder or financial institution (e.g., Exim Bank of China, China Development Bank, Bank of China). If none are found, note this explicitly.
"""

prompt_c_extraction = """
C. Financial Instrument: Identify and extract specific terminology related to financial transactions. For example, “loan," "debt," "borrow,“, “financial support”, “equity investment”, “grant”, “donation”, “aid” and others. If multiple terms are mentioned, list up to the top three most relevant ones.
"""

prompt_d_extraction = """
D. Project or Activity: Identify and extract the name of any specific project (e.g., bridge, highway, power plant) or activity (e.g., mining, connectivity). If no specific project is mentioned, extract the industry or purpose (e.g., infrastructure, development, trade). If neither is mentioned, note the lack of details.
"""

prompt_a_score = """
A. Recipient
Score 5: The recipient (or recipient countries) extracted is discussed in the content is a country that is borrowing or hosting a project for development.
Score 3: The recipient (or recipients) extracted is a country but is not connected with financing or mentioned in the context of borrowing.
Score 1: No recipient country is mentioned.
"""

prompt_b_score = """
B. Chinese Lender 
Score 5: The identified institution is a Chinese development financial institution. There are two: the Import-Export Bank of China (Exim Bank or CEXIM, for short) or the China Development Bank (CDB, for short).
Score 3: The identified institution is a Chinese actor or financier but not one of the two development banks.
Score 1: No reference to a Chinese funder or the given project is associated with non-Chinese lenders such as the World Bank or another country’s finance institution.
"""

prompt_c_score = """
C. Financial Instrument
Score 5: Clearly identifies the transaction as a loan, using terms like lending, borrowing, debt, or loan.
Score 3: Mentions financing or general support but does not explicitly confirm it as a loan. The description is vague regarding the financial instrument.
Score 1: Indicates the transaction is not a loan, instead describing it as an equity investment, grant, or donation.
"""

prompt_d_score = """
D. Activity Precision
Score 5: The extracted project is a development project that has a specific name. 
Score 3: Refers to general development support or an industry without specifying a particular project or cooperation model. 
Score 1: There is no detail about a project.
"""

class LLMScore(BaseModel):
    score: int

class LLMOutput(BaseModel):
    justification: str


def getExtraction(prompt, extractionPrompt, content, OutputClass):
    response: ChatResponse = client.chat(model=model, messages=[
        {
            'role': 'system',
            'content': prompt,
        },
        {
            'role': 'system',
            'content': extractionPrompt,
        },
        {
            'role': 'user',
            'content': content,
        }
    ],
    format=OutputClass.model_json_schema(),
    stream=False)
    return response


@app.route('/health')
def health_check():
    return 'healthy'


def getText(url):
    print("[MOF Classifier] Loading URL: " + url)
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            return re.sub(r"\s+", " ", text).strip()
        else:
            return None
    except Exception as e:
        print(f"[MOF Classifier] Error loading URL: {e}")
        return None

def classify():
    print("[MOF Classifier] Classifying started at " + datetime.now().isoformat() + "\n")

    while True:
        db_url = os.getenv("NOCO_DB_URL")
        headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
        params = {
            "fields": "Id,originalTitle,translatedTitle,originalContent,translatedContent,originalOutlet,translatedOutlet,isEnglish,originalLanguage,articleUrl,webScrapedContent",
            "where": f"({AI_SCORE},is,null)~and(isEnglish,eq,true)~and(FinanceClassification,isnot,null)"
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
                    print("[MOF Classifier] Classifying article: " + article["originalTitle"])
                    llm_title = article["translatedTitle"] if article.get("translatedTitle") else article["originalTitle"]
                    llm_content = article["translatedContent"] if article.get("translatedContent") else article["originalContent"]
                    if len(llm_content) < 1000:
                        if( article["webScrapedContent"] == None):
                            article["webScrapedContent"] = getText(article["articleUrl"])
                    llm_content = article["webScrapedContent"] if article["webScrapedContent"] != None else llm_content
                    llm_prompt = f"Headline: {llm_title}\n\nBody: {llm_content}"
                    a_extraction_response = getExtraction(prompt, prompt_a_extraction, llm_prompt, LLMExtractionA)
                    b_extraction_response = getExtraction(prompt, prompt_b_extraction, llm_prompt, LLMExtractionB)
                    c_extraction_response = getExtraction(prompt, prompt_c_extraction, llm_prompt, LLMExtractionC)
                    d_extraction_response = getExtraction(prompt, prompt_d_extraction, llm_prompt, LLMExtractionD)
                    article['a'] = json.loads(str(a_extraction_response['message']['content']))['recipient']
                    article['b'] = json.loads(str(b_extraction_response['message']['content']))['chinese_institution']
                    article['c'] = json.loads(str(c_extraction_response['message']['content']))['financial_instrument']
                    article['d'] = json.loads(str(d_extraction_response['message']['content']))['project_or_activity']
                    print("[MOF Classifier] A extraction response: " + article['a'])
                    print("[MOF Classifier] B extraction response: " + article['b'])
                    print("[MOF Classifier] C extraction response: " + article['c'])
                    print("[MOF Classifier] D extraction response: " + article['d'])
                    a_score_prompt = f"A. Recipient\n{article['a']}"
                    a_score_response = getExtraction(prompt, prompt_a_score, a_score_prompt, LLMScore)
                    b_score_prompt = f"B. Chinese Lender\n{article['b']}"
                    b_score_response = getExtraction(prompt, prompt_b_score, b_score_prompt, LLMScore)
                    c_score_prompt = f"C. Financial Instrument\n{article['c']}"
                    c_score_response = getExtraction(prompt, prompt_c_score, c_score_prompt, LLMScore)
                    d_score_prompt = f"D. Activity Precision\n{article['d']}"
                    d_score_response = getExtraction(prompt, prompt_d_score, d_score_prompt, LLMScore)

                    article[f"{AI_SCORE}_a"] = int(json.loads(a_score_response['message']['content'])['score'])
                    article[f"{AI_SCORE}_b"] = int(json.loads(b_score_response['message']['content'])['score'])
                    article[f"{AI_SCORE}_c"] = int(json.loads(c_score_response['message']['content'])['score'])
                    article[f"{AI_SCORE}_d"] = int(json.loads(d_score_response['message']['content'])['score'])

                    print("[MOF Classifier] A score: " + str(article[f"{AI_SCORE}_a"]))
                    print("[MOF Classifier] B score: " + str(article[f"{AI_SCORE}_b"]))
                    print("[MOF Classifier] C score: " + str(article[f"{AI_SCORE}_c"]))
                    print("[MOF Classifier] D score: " + str(article[f"{AI_SCORE}_d"]))

                    if article[f"{AI_SCORE}_a"] > 5 or article[f"{AI_SCORE}_b"] > 5 or article[f"{AI_SCORE}_c"] > 5 or article[f"{AI_SCORE}_d"] > 5:
                        raise Exception("Invalid score")

                    article[AI_SCORE] = (article[f"{AI_SCORE}_a"] + article[f"{AI_SCORE}_b"] + article[f"{AI_SCORE}_c"] + article[f"{AI_SCORE}_d"]) / 4

                    justificationPrompt = (
                       f"A: {article['a']}: Score{article[f'{AI_SCORE}_a']}\n" +
                       f"B: {article['b']}: Score{article[f'{AI_SCORE}_b']}\n" +
                       f"C: {article['c']}: Score{article[f'{AI_SCORE}_c']}\n" +
                       f"D: {article['d']}: Score{article[f'{AI_SCORE}_d']}")

                    response = getExtraction(prompt, "Please provided justification", justificationPrompt, LLMOutput)
                    response = json.loads(response['message']['content'])
                    justification = response['justification']
                    article[f"{AI_SCORE}_Justification"] = justification
                    requests.patch(db_url, headers=headers, json=article)
                except Exception as e:
                    print(e)
                    attempts -= 1
                    print(f"[MOF Classifier] Request timeout. {attempts} attempt(s) left ...")
                    article[AI_SCORE] = -2
                break


if __name__ == '__main__':
    print(f"Updating {AI_SCORE}")
    load_dotenv()
    scheduler.add_job(classify, "cron", hour="*", minute="*/5", max_instances=4)
    scheduler.start()
    #classify()
    print("Classifier schedule started")
    app.run(port=5003)
