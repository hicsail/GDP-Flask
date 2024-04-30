from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from dotenv import load_dotenv

import os
import translator
import requests

app = Flask(__name__)
scheduler = BackgroundScheduler()

@app.route("/health")
def health_check():
    return "healthy"

def translate():
    try:
        print("[MOF Translator] Translating started at " + datetime.now().isoformat() + "\n")
        url = os.getenv("NOCO_DB_URL")
        headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
        params = {
            "where": "(isEnglish,eq,false)",
            "limit": 10, # translate 10 records at a time
        }
        res = requests.get(url, headers=headers, params=params)
        if res.json().get("pageInfo").get("totalRows") == 0:
            print("[MOF Translator] No records to translate\n")
            return

        translator_instance = translator.Translator()
        target_field = {
            "originalTitle": "translatedTitle",
            "originalContent": "translatedContent",
            "originalOutlet": "translatedOutlet"
        }

        for record in res.json().get("list"):
            for key in record.keys():
                if key in target_field.keys():
                    if not record.get(key):
                        continue
                    sample_text = record.get(key).strip()[:100]
                    language = translator_instance.detect_lang_google(sample_text)
                    if language != "en" and language != "und":
                        # record[target_field[key]] = translator_instance.translate_text_deepl(record.get(key)).text
                        record[target_field[key]] = translator_instance.translate_text_google(record.get(key))

                        if not record.get(key) and record[target_field[key]] == None:
                            raise Exception("Translation failed")

            record["isEnglish"] = True
            requests.patch(url, headers=headers, json=record)
            print(f"[MOF Translator] Translated record: {record.get('originalTitle')}\n to {record.get('translatedTitle')}\n")
    except Exception as e:
        print(f"[MOF Translator] Error: {e}")


if __name__ == "__main__":
    load_dotenv()
    scheduler.add_job(translate, "cron", hour="*", minute="*/5")
    scheduler.start()
    print("[MOF Translator] Start translating")
    app.run(port=5002)