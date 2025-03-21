import json
import os
import re
from datetime import datetime
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask
from ollama import Client, ChatResponse
from pydantic import BaseModel
from enum import Enum
import re


AI_SCORE = "AIScore4"
model = "gemma3:12b"

country_list = [
    'Afghanistan',
    'Albania',
    'Algeria',
    'Andorra',
    'Angola',
    'Antigua and Barbuda',
    'Argentina',
    'Armenia',
    'Aruba',
    'Australia',
    'Austria',
    'Azerbaijan',
    'Bahamas',
    'Bahrain',
    'Bangladesh',
    'Barbados',
    'Belarus',
    'Belgium',
    'Belize',
    'Benin',
    'Bermuda',
    'Bhutan',
    'Bolivia',
    'Bosnia and Herzegovina',
    'Botswana',
    'Brazil',
    'British Virgin Islands',
    'Brunei',
    'Bulgaria',
    'Burkina Faso',
    'Burundi',
    'Cabo Verde',
    'Cambodia',
    'Cameroon',
    'Canada',
    'Cayman Islands',
    'Central African Republic',
    'Chad',
    'Chile',
    'China',
    'Colombia',
    'Comoros',
    'Congo, Democratic Republic of the',
    'Congo, Republic of the',
    'Costa Rica',
    'Côte d’Ivoire',
    'Croatia',
    'Cuba',
    'Curaçao',
    'Cyprus',
    'Czech Republic',
    'Denmark',
    'Djibouti',
    'Dominica',
    'Dominican Republic',
    'Ecuador',
    'Egypt',
    'El Salvador',
    'Equatorial Guinea',
    'Eritrea', ','
               'Estonia',
    'Eswatini',
    'Ethiopia',
    'Faroe Islands',
    'Fiji',
    'Finland',
    'France',
    'French Polynesia',
    'Gabon',
    'Gambia',
    'Georgia',
    'Germany',
    'Ghana',
    'Gibraltar',
    'Greece',
    'Greenland',
    'Grenada',
    'Guam',
    'Guatemala',
    'Guinea',
    'Guinea-Bissau',
    'Guyana',
    'Haiti',
    'Honduras',
    'Hong Kong SAR, China',
    'Hungary',
    'Iceland',
    'India',
    'Indonesia',
    'Iran, Islamic Republic',
    'Iraq',
    'Ireland',
    'Isle of Man',
    'Israel',
    'Italy',
    'Jamaica',
    'Japan',
    'Jordan',
    'Kazakhstan',
    'Kenya',
    'Kiribati',
    'Korea, Dem. People\'s Rep.',
    'Korea, Rep.',
    'Kosovo',
    'Kuwait',
    'Kyrgyz Republic',
    'Lao People\'s Democratic Republic',
    'Latvia',
    'Lebanon',
    'Lesotho',
    'Liberia',
    'Libya',
    'Liechtenstein',
    'Lithuania',
    'Luxembourg',
    'Macao SAR, China',
    'Madagascar',
    'Malawi',
    'Malaysia',
    'Maldives',
    'Mali',
    'Malta',
    'Marshall Islands',
    'Mauritania',
    'Mauritius',
    'Mexico',
    'Micronesia, Fed. Sts.',
    'Moldova',
    'Monaco',
    'Mongolia',
    'Montenegro',
    'Morocco',
    'Mozambique',
    'Myanmar',
    'Namibia',
    'Nauru',
    'Nepal',
    'Netherlands',
    'New Caledonia',
    'New Zealand',
    'Nicaragua',
    'Niger',
    'Nigeria',
    'North Macedonia',
    'Northern Mariana Islands',
    'Norway',
    'Oman',
    'Pakistan',
    'Palau',
    'Panama',
    'Papua New Guinea',
    'Paraguay',
    'Peru',
    'Philippines',
    'Poland',
    'Portugal',
    'Puerto Rico',
    'Qatar',
    'Regional',
    'Romania',
    'Russian Federation',
    'Rwanda',
    'Samoa',
    'Samoa',
    'San Marino',
    'São Tomé and Príncipe',
    'Saudi Arabia',
    'Senegal',
    'Serbia',
    'Seychelles',
    'Sierra Leone',
    'Singapore',
    'Sint Maarten (Dutch part)',
    'Slovak Republic',
    'Slovenia',
    'Solomon Islands',
    'Somalia',
    'South Africa',
    'South Sudan',
    'Spain',
    'Sri Lanka',
    'St. Kitts and Nevis',
    'St. Lucia',
    'St. Martin (French part)',
    'St. Vincent and the Grenadines',
    'Sudan',
    'Suriname',
    'Sweden',
    'Switzerland',
    'Syrian Arab Republic',
    'Taiwan, China',
    'Tajikistan',
    'Tanzania',
    'Thailand',
    'Timor-Leste',
    'Togo',
    'Tonga',
    'Trinidad and Tobago',
    'Tunisia',
    'Türkiye',
    'Turkmenistan',
    'Turks and Caicos Islands',
    'Tuvalu',
    'Uganda',
    'Western Sahara',
    'Ukraine',
    'United Arab Emirates',
    'United Kingdom',
    'United States',
    'Uruguay',
    'Uzbekistan',
    'Vanuatu',
    'Venezuela',
    'Vietnam',
    'Virgin Islands (U.S.)',
    'West Bank and Gaza',
    'Yemen',
    'Zambia',
    'Zimbabwe',
]
Country = Enum("Country", {c.upper(): c for c in country_list}, type=str)


class LLMExtractionA(BaseModel):
    recipient: Country

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
First, extract the necessary information from the provided content (headline and article title) using the extraction rubric.

A. Recipient: Identify and extract the name of the country that is the main subject of the article (excluding China) using the country list. If multiple countries/entities are mentioned, list up to the top three most relevant ones. China cannot be one of them. 
"""

prompt_b_extraction = """
First, extract the necessary information from the provided content (headline and article title) using the extraction rubric.

B. Chinese Institution: Identify and extract any mention of a lender, funder or financial institution (e.g., Exim Bank of China, China Development Bank, Bank of China). If none are found, note this explicitly.
"""

prompt_c_extraction = """
First, extract the necessary information from the provided content (headline and article title) using the extraction rubric.

C. Financial Instrument: Identify and extract specific terminology related to financial transactions. For example, “loan," "debt," "borrow,“, “financial support”, “equity investment”, “grant”, “donation”, “aid” and others. If multiple terms are mentioned, list up to the top three most relevant ones.
"""

prompt_d_extraction = """
First, extract the necessary information from the provided content (headline and article title) using the extraction rubric.


D. Project or Activity: Identify and extract the purpose of the loan. For example, the name of any specific project (e.g., bridge, highway, power plant, mine), if provided. If no specific project is mentioned, extract the industry or purpose (e.g., infrastructure, development, trade). If neither are provided, note the topic of the article.
"""

prompt_a_score = """
Second, conduct an evaluation of the provided headline and article title and the extracted content using the Score Generator rubric. Assess whether the headline and body indicate financial activities where a Chinese financial institution is acting as the lender. Evaluate each factor separately, providing a score from 1-5 and a justification for each. Keep the justification concise, up to 25 words

A. Recipient
Score 5: The recipient or recipients are countries in the following World Bank Income Group: ‘low income’, ‘lower middle income’, and ‘upper middle income’.
Score 3: The recipient (or recipients) extracted is a country in the ‘high income’ group.
Score 1: The recipient extracted is not a country or is China.

"""

prompt_b_score = """
Second, conduct an evaluation of the provided headline and article title and the extracted content using the Score Generator rubric. Assess whether the headline and body indicate financial activities where a Chinese financial institution is acting as the lender. Evaluate each factor separately, providing a score from 1-5 and a justification for each. Keep the justification concise, up to 25 words

B. Chinese Lender 
Score 5: The identified institution is a Chinese development financial institution. There are two: the Import-Export Bank of China (Exim Bank or CEXIM, for short) or the China Development Bank (CDB, for short).
Score 3: The identified institution is a Chinese actor or financier but not one of the two development banks.
Score 1: No reference to a Chinese funder or the given project is associated with non-Chinese lenders such as the World Bank or another country’s finance institution.
"""

prompt_c_score = """
Second, conduct an evaluation of the provided headline and article title and the extracted content using the Score Generator rubric. Assess whether the headline and body indicate financial activities where a Chinese financial institution is acting as the lender. Evaluate each factor separately, providing a score from 1-5 and a justification for each. Keep the justification concise, up to 25 words

C. Financial Instrument
Score 5: Clearly identifies the transaction as a loan, using terms like lending, borrowing, debt, or loan.
Score 3: Mentions financing or general support but does not explicitly confirm it as a loan. The description is vague regarding the financial instrument.
Score 1: Indicates the transaction is not a loan, or exclusively describes it as an equity investment, grant, or donation.
"""

prompt_d_score = """
Second, conduct an evaluation of the provided headline and article title and the extracted content using the Score Generator rubric. Assess whether the headline and body indicate financial activities where a Chinese financial institution is acting as the lender. Evaluate each factor separately, providing a score from 1-5 and a justification for each. Keep the justification concise, up to 25 words

D. Activity Precision
Score 5: The extracted entity is a project that has a specific name or location. 
Score 3: The extracted entity does not specify a particular project but rather mentions a vague purpose, such as ‘infrastructure development’, or broad industry like ‘telecommunications sector’. 
Score 1: The topic is not relevant to a project being financed.
"""

class LLMScore(BaseModel):
    score: int

class LLMOutput(BaseModel):
    justification: str

def split_into_chunks(text, max_chars=3000):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def condense_article(article, max_chunk_chars=3000):
    chunks = split_into_chunks(article, max_chunk_chars)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    return "\n".join(summaries)

def summarize_chunk(text):
    prompt = (f"Summarize this text, only remove words that are not useful when determining:\n"
              f"A. Recipient: Identify and extract the name of the country that is the main subject of the article\n"
              f"B. Chinese Institution: Identify and extract any mention of a lender, funder or financial institution\n"
              f"C. Financial Instrument: Identify and extract specific terminology related to financial transactions\n"
              f"D. Project or Activity: Identify and extract the purpose of the loan\n"
              f"\n{text}")
    response = client.chat(model=model, messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']


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


def classify(offset=0):
    print("[MOF Classifier] Classifying started at " + datetime.now().isoformat() + "\n")

    while True:
        db_url = os.getenv("NOCO_DB_URL")
        headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
        params = {
            "fields": "Id,originalTitle,translatedTitle,originalContent,translatedContent,originalOutlet,translatedOutlet,isEnglish,originalLanguage,articleUrl,webScrapedContent",
            "where": f"({AI_SCORE},is,null)~and(isEnglish,eq,true)~and(FinanceClassification,isnot,null)",
            "offset": offset,
            "limit": 10,
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

                    if (len(llm_prompt) > 12000):
                        o_len = len(llm_prompt)
                        print("[MOF Classifier] Article too long. Condensing ...")
                        llm_prompt = condense_article(llm_prompt)
                        n_len = len(llm_prompt)
                        print(f"[MOF Classifier] Condensed prompt by {o_len - n_len} characters")

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
    #scheduler.add_job(classify, "cron", hour="*", minute="*/1", max_instances=1)
    #scheduler.add_job(classify, "cron", hour="*", minute="*/1", max_instances=1, args=[10])
    #scheduler.add_job(classify, "cron", hour="*", minute="*/1", max_instances=1, args=[20])
    #scheduler.add_job(classify, "cron", hour="*", minute="*/1", max_instances=1, args=[30])
    #scheduler.start()
    classify()
    print("Classifier schedule started")
    app.run(port=5003)
