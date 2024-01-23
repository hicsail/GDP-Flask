from flask import Flask
from bs4 import BeautifulSoup
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from translator import Translator

app = Flask(__name__)
scheduler = BackgroundScheduler()
translator = Translator()

def get_target_url(country, type, keyword, page = 1):
    domain = "http://search.mofcom.gov.cn"
    path = "allSearch"
    countryVar = f"?siteId={country}"
    searchTypeVar = f"&keyWordType={type}"
    keyWordVar = f"&acSuggest={keyword}"
    pageVar = f"&page={page}"

    url = "/".join([domain, path, countryVar + searchTypeVar + keyWordVar]) + pageVar

    return url

def scrape():
    URL = get_target_url("vn", "title", "贷款")
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    div = soup.find("div", class_="wms-con").find("div", class_="s-info-box")

    if div:
        result = div.find_all("li")

        cnt = 0
        for i in result:
            if cnt > 3:
                break

            link = i.find("a").get("href")
            articlePage = requests.get(link)
            article = BeautifulSoup(articlePage.content, "html.parser")
            title = article.find(id="artitle").text
            content = article.find(id="zoom").text
            print(title)
            print(content)
            if cnt == 1:
                print(translator.translate_text_deepl(content))
            print(link)
            print()

            cnt += 1
    else:
        print("No result found")


if __name__ == "__main__":
    scheduler.add_job(scrape, "interval", minutes=1)
    scheduler.start()
    app.run()