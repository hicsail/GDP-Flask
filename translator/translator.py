import deepl
import requests
import os

class Translator:
    def __init__(self):
        self.deepl_key = os.getenv("DEEPL_API_KEY")
        self.deepl_translator = deepl.Translator(self.deepl_key)

        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.google_url = "https://translation.googleapis.com/language/translate/v2"

    def translate_text_deepl(self, text, target_lang="EN-US"):
        return self.deepl_translator.translate_text(text, target_lang=target_lang)
    
    def translate_text_google(self, text, target_lang="en"):
        param = { "target": target_lang, "key": self.google_key }
        data = { "q": text }
        res = requests.post(self.google_url, params=param, data=data)

        if res.status_code != 200:
            return None
        
        return res.json()["data"]["translations"][0]["translatedText"]
    
    def detect_lang_google(self, text):
        param = { "key": self.google_key }
        data = { "q": text}
        res = requests.post(self.google_url + "/detect", params=param, data=data)

        if res.status_code != 200:
            return None
        
        return res.json()["data"]["detections"][0][0]["language"]