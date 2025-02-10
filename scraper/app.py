from flask import Flask
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from dotenv import load_dotenv

import json
import pycountry
import pytz
import requests
import re
import os

app = Flask(__name__)
scheduler = BackgroundScheduler()
terms = []
regions = {}

beijing_tz = pytz.timezone("Asia/Shanghai")
est_tz = pytz.timezone("US/Eastern")

with open("terms.list", "r") as file:
    terms = file.readlines()

with open("region.json", "r") as file:
    regions = json.load(file)

def get_target_url(country, keyword, startTime = "", page = 1):
    domain = "http://search.mofcom.gov.cn"
    path = "allSearch"
    countryVar = f"?siteId={country}"
    searchTypeVar = f"&keyWordType=all"
    keyWordVar = f"&includeAll={keyword}"
    miscVar = "&random=&notInclude=&size=30&searchScope=is_all&hightSearchType=all&radio=publish_time_str"
    startTime = f"&startTime={startTime}&endTime=2024-07-01"
    pageVar = f"&page={page}"

    url = "/".join([domain, path, countryVar + searchTypeVar + keyWordVar + miscVar + startTime + pageVar]) 

    return url

@app.route("/health")
def health_check():
    return "healthy"

result_set = set()
def scrape_country(country, latest_date, keywords):
    new_records = []

    # loop through all pages
    pageNum = 1
    while True:
        URL = get_target_url(country, keywords, latest_date, pageNum)
        req_cnt = 0
        timeout = True
        page = None
        while req_cnt < 5:
            try:
                page = requests.get(URL, timeout=15)   # request timeout after 15 seconds
                timeout = False
                break
            except:
                req_cnt += 1
                print(f"[MOF Scraper] Request timeout for {URL}, retrying...")
        
        if timeout:
            print(f"[MOF Scraper] Request failed for {URL}, skipping...")
            break

        soup = BeautifulSoup(page.content, "html.parser")
        div = soup.find("div", class_="wms-con").find("div", class_="s-info-box")

        result = div.find_all("li")
        if len(result) == 0:    # no result with current search term
            break
        
        # loop through all articles in current page
        for i in result:
            contype = ""
            tm = ""
            original_source = ""
            est_time = None

            # access article page
            link = i.find("a").get("href")

            # check if article already exists in the database
            url = os.getenv("NOCO_DB_URL")
            headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
            params = {
                "where": f"(articleUrl,eq,{link})"
            }

            req = requests.get(url, headers=headers, params=params)
            if len(req.json()["list"]) > 0:
                print(f"[MOF Scraper] Article {link} already exists in the database, skipping...")
                continue

            # try to access article page
            req_cnt = 0
            timeout = True
            article_page = None
            while req_cnt < 3:
                try:
                    article_page = requests.get(link, timeout=15)   # request timeout after 15 seconds
                    timeout = False
                    break
                except:
                    req_cnt += 1
                    print(f"[MOF Scraper] Request timeout for {link}, retrying...")

            if timeout:
                print(f"[MOF Scraper] Request failed for {link}, skipping...")
                continue
            article = BeautifulSoup(article_page.content, "html.parser")

            # article does not exist
            if article.find(id="zoom") is None or article.find(id="artitle") is None:
                print(f"[MOF Scraper] Article does not exist for {link}")
                if "政策" in i.find("em", class_="tag").text:
                    continue

                title = "[DELETED] " + i.find("a").text.strip()
                content = i.find("div", class_="bd").text.strip()
                sub_content = i.find("div", class_="ft-col").find("p").text.strip()

                match = re.search(r"来源：(.+?) (\d{4}-\d{2}-\d{2})", sub_content)
                if match:
                    original_source = match.group(1)
                    date = datetime.strptime(match.group(2), "%Y-%m-%d")
                    localized_beijing_time = beijing_tz.localize(date)
                    est_time = localized_beijing_time.astimezone(est_tz)
                else:
                    print(f"[MOF Scraper] Failed to parse date for {link}, set to default date")
                    est_time = datetime.min

            else:
                # extract content type and publish date
                # scripts = article.find_all("script")
                # for script in scripts:
                #     if "contype" in script.text:
                #         match = re.search(r'var contype = [\'"](.*)[\'"];', script.text)
                #         if match:
                #             contype = match.group(1)

                #         match = re.search(r'var tm = [\'"](.*)[\'"];', script.text)
                #         if match:
                #             tm = match.group(1)

                #         match = re.search(r'var source = [\'"](.*)[\'"];', script.text)
                #         if match:
                #             original_source = match.group(1)

                #         break

                top_info = article.find("section", class_="article-tool")

                contype = ""
                if "分类" in top_info.get_text(strip=True):
                    category_span = top_info.find_all('span', class_='m-ar-none')
                    contype = category_span[1].get_text(strip=True).replace("分类：", "")
                else:
                    print(f"[MOF Scraper] Failed to get content type for {link}")

                original_source = ""
                if "来源" in top_info.get_text(strip=True):
                    source_text = top_info.find('p').get_text(strip=True)
                    original_source = source_text.split("来源：")[1].split("类型：")[0].strip()
                else:
                    print(f"[MOF Scraper] Failed to get source for {link}")

                tm = top_info.find_all('p')[1].get_text(strip=True)

                # ignore policy articles
                if contype == "政策":
                    continue

                title = article.find(id="artitle").text.strip()
                for script in article.find(id="zoom").find_all("script"):
                    script.decompose()

                content = article.find(id="zoom").text.strip()

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

                try:
                    date = datetime.strptime(tm, "%Y-%m-%d %H:%M")
                    localized_beijing_time = beijing_tz.localize(date)
                    est_time = localized_beijing_time.astimezone(est_tz)
                except:
                    print(f"[MOF Scraper] Failed to parse date for {link}, set to default date")
                    est_time = datetime.min

            record = {
                "originalTitle": title.replace("\n", ""),
                "originalContent": content,
                "originalLanguage": "zh",
                "source": "Ministry of Commerce of the People's Republic of China",
                "originalOutlet": original_source,
                "articlePublishDateEst": est_time.strftime("%Y-%m-%d %H:%M"),
                "articleUrl": link,
                "country": pycountry.countries.get(alpha_2=country.upper()).name,
                "region": regions[country.upper()],
                "isEnglish": False,
                "keywords": ",".join(keywords.split("+"))
            }

            new_records.append(record)

        pageNum += 1

    return new_records

def scrape():
    print("[MOF Scraper] Sraping started at " + datetime.now().isoformat() + "\n")
    ignore = ["CN", "HK", "MO", "TW"]   # ignore Mainland China, Hong Kong, Macau, and Taiwan
    for country in pycountry.countries:
        if country.alpha_2 not in ignore:
            url = os.getenv("NOCO_DB_URL")
            headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}

            # get latest date of article in the database
            date = ""
            date_params = {
                "fields": "articlePublishDateEst",
                "sort": "-articlePublishDateEst",
                "where": f"(country,eq,{country.name})~and(articlePublishDateEst,lte,exactDate,2024-07-01)",
                "limit": 1
            }
            date_req = requests.get(url, headers=headers, params=date_params)
            try:
                if len(date_req.json().get("list")) > 0:
                    date_est = date_req.json().get("list")[0].get("articlePublishDateEst")
                    date_obj = datetime.fromisoformat(date_est)
                    beijing_time = date_obj.astimezone(beijing_tz)
                    date = beijing_time.strftime("%Y-%m-%d")
            except Exception as e:
                print(f"[MOF Scraper] Failed to get latest date for {country.name}")
                print(e)
                continue

            print("[MOF Scraper] =====================================")
            print(f"[MOF Scraper] Scraping {country.name} from {date} CST...")
            timestart = datetime.now()
            articles = []
            for term in terms:
                country_code = country.alpha_2.lower()
                articles.extend(scrape_country(country_code, date, "+".join(term.split(" "))))

            for article in articles:
                params = {
                    "where": f"(originalTitle,eq,{article['originalTitle']})"
                }
                
                # check if article already exists in the database
                req = requests.get(url, headers=headers, params=params)
                try:
                    if len(req.json()["list"]) == 0:
                        requests.post(url, headers=headers, json=article)
                except Exception as e:
                    print(f"[MOF Scraper] Failed to post article {article['originalTitle']} to the database")
                    print(e)


            timeend = datetime.now()

            print(f"\n[MOF Scraper] Scraped {len(articles)} articles from {country.name} in {timeend - timestart}")

            result_set.clear()
            

if __name__ == "__main__":
    load_dotenv()
    # initial scrape, this process will take longer
    print("[MOF Scraper] Start inital scraping")
    scrape()
    # scheduler.add_job(scrape, "cron", month="1,7", day="1", hour="0", minute="0")
    # scheduler.start()
    app.run(port=5001)