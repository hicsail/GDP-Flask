from flask import Flask
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from dotenv import load_dotenv

import pycountry
import requests
import re
import os

app = Flask(__name__)
scheduler = BackgroundScheduler()
terms = []

with open("terms.list", "r") as file:
    terms = file.readlines()

def get_target_url(country, type, keyword, page = 1):
    domain = "http://search.mofcom.gov.cn"
    path = "allSearch"
    countryVar = f"?siteId={country}"
    searchTypeVar = f"&keyWordType={type}"
    keyWordVar = f"&acSuggest={keyword}"
    pageVar = f"&page={page}"

    url = "/".join([domain, path, countryVar + searchTypeVar + keyWordVar]) + pageVar

    return url

@app.route("/health")
def health_check():
    return "healthy"

result_set = set()
def scrape_country(country, content_type, keywords):
    new_records = []

    # loop through all pages
    pageNum = 1
    while True:
        URL = get_target_url(country, content_type, keywords, pageNum)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        div = soup.find("div", class_="wms-con").find("div", class_="s-info-box")

        result = div.find_all("li")
        if len(result) == 0:    # no result with current search term
            break
        
        # loop through all articles in current page
        for i in result:
            contype = ""
            tm = ""

            # access article page
            link = i.find("a").get("href")
            article_page = requests.get(link)
            article = BeautifulSoup(article_page.content, "html.parser")

            # extract content type and publish date
            scripts = article.find_all("script")
            for script in scripts:
                if "contype" in script.text:
                    match = re.search(r'var contype = "(.*)";', script.text)
                    if match:
                        contype = match.group(1)

                    match = re.search(r'var tm = "(.*)";', script.text)
                    if match:
                        tm = match.group(1)

                    break

            # ignore policy articles
            if contype == "政策":
                continue

            date = datetime.strptime(tm, "%Y-%m-%d %H:%M:%S")
            title = article.find(id="artitle").text
            for script in article.find(id="zoom").find_all("script"):
                script.decompose()

            content = article.find(id="zoom").text

            # ignore duplicate articles
            if title in result_set:
                continue

            # ignore articles without keywords
            irrelevant = False
            for keyword in keywords.split("+"):
                if keyword.strip() not in content:
                    irrelevant = True
                    break

            if irrelevant:
                continue

            result_set.add(title)
            record = {
                "title": title,
                "content": content.strip(),
                "language": "zh",
                "source": "Ministry of Commerce of the People's Republic of China",
                "article_publish_date": date.isoformat(),
                "article_link": link,
                "country": pycountry.countries.get(alpha_2=country.upper()).name,
                "translated": False
            }

            new_records.append(record)

        pageNum += 1

    return new_records

def scrape():
    print("[MOF Scraper] Sraping started at " + datetime.now().isoformat() + "\n")
    ignore = ["CN", "HK", "MO", "TW"]
    for country in pycountry.countries:
        # if country.alpha_2 not in ignore:
        if country.alpha_2 in ["AO"]:   # for testing
            print("[MOF Scraper] =====================================")
            timestart = datetime.now()
            articles = []
            for term in terms:
                country_code = country.alpha_2.lower()
                articles.extend(scrape_country(country_code, "title", "+".join(term.split(" "))))
                articles.extend(scrape_country(country_code, "content", "+".join(term.split(" "))))

            url = os.getenv("NOCO_DB_URL")
            headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}

            for article in articles:
                params = {
                    "where": f"(title,eq,{article['title']})"
                }
                
                # check if article already exists in the database
                req = requests.get(url, headers=headers, params=params)
                if req.json().get("pageInfo").get("totalRows") == 0:
                    requests.post(url, headers=headers, json=article)

            timeend = datetime.now()

            print(f"\n[MOF Scraper] Scraped {len(articles)} articles from {country.name} in {timeend - timestart}")

            result_set.clear()
            

if __name__ == "__main__":
    load_dotenv()
    # initial scrape, this process will take longer
    print("[MOF Scraper] Start inital scraping")
    scrape()
    # scheduler.add_job(scrape, "cron", month="1,7", day="1", hour="0", minute="0")
    scheduler.add_job(scrape, "interval", minutes=15)   # for testing
    scheduler.start()
    app.run(port=5001)