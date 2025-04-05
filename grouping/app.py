import os
import requests
import dedupe
import pandas as pd
import json

CACHE_PATH = "/app/output/fetched_articles.json"


def fetch_all_articles(page_size=100, max_records=50000):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
    offset = 0
    fetched = 0

    while True:
        params = {
            "fields": "Id,originalTitle,articleUrl,a,b,d",
            "where": "(a,neq,null)~and(a,neq,'')",
            "offset": offset,
            "limit": page_size,
            "sort": "-articlePublishDateEst",
        }

        response = requests.get(db_url, headers=headers, params=params)
        if response.status_code == 422:
            print(f"Skipping offset {offset}: HTTP 422 - Invalid request")
            break
        if response.status_code != 200:
            print(f"Skipping offset {offset}: HTTP {response.status_code}")
            offset += page_size
            continue

        try:
            json_data = response.json()
            batch = json_data.get("list", [])
            if not isinstance(batch, list):
                print(f"Unexpected response format at offset {offset}: {json_data}")
                break
        except Exception as e:
            print(f"Error decoding JSON at offset {offset}: {e}")
            break

        if not batch:
            break  # No more records

        for article in batch:
            yield article

        offset += page_size
        fetched += len(batch)

        if fetched >= max_records:
            break

def save_articles_to_disk(articles, path="/app/output/fetched_articles.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(articles)} articles to {path}")

def load_articles_from_disk(path="/app/output/fetched_articles.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_article_for_dedupe(article):
    return {
        "Id": article.get("Id", ""),
        "title": article.get("originalTitle", ""),
        "url": article.get("articleUrl", ""),
        "A": article.get("a", ""),
        "B": article.get("b", ""),
        "D": article.get("d", ""),
    }


def deduplicate_articles(records):
    settings_path = '/app/output/dedupe_learned_settings'

    if os.path.exists(settings_path):
        print("📦 Loading trained settings...")
        with open(settings_path, 'rb') as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        print("🧠 Training new model...")
        fields = [
            dedupe.variables.String("title"),
            dedupe.variables.String("url"),
            dedupe.variables.String("A"),
            dedupe.variables.String("B"),
            dedupe.variables.String("D"),
        ]
        deduper = dedupe.Dedupe(fields)
        deduper.prepare_training(records, sample_size=1000)
        dedupe.console_label(deduper)
        deduper.train()
        print("💾 Saving trained settings...")
        with open(settings_path, 'wb') as f:
            deduper.write_settings(f)

    print("🧮 Clustering with auto threshold...")
    clustered_dupes = deduper.partition(records)

    return clustered_dupes


def save_results(clustered_dupes, records, output_path="deduped_results.csv"):
    rows = []
    for cluster_id, (record_ids, confidence) in enumerate(clustered_dupes):
        for record_id in record_ids:
            record = records[record_id].copy()
            record["cluster_id"] = cluster_id
            record["confidence"] = confidence
            rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved clustered results to {output_path}")

def update_article(article_id, cluster_id):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}

    data = {
        "id": article_id,
        "cluster_id": cluster_id,
    }

    response = requests.patch(db_url, headers=headers, json=data)
    if response.status_code not in (200, 204):
        print(f"⚠️ Failed to update article {article_id}: {response.status_code} - {response.text}")

def update_article(article_id, cluster_id):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {
        "xc-token": os.getenv("NOCO_XC_TOKEN"),
        "Content-Type": "application/json",
    }

    data = {
        "id": article_id,
        "cluster_id": cluster_id,
    }

    response = requests.patch(db_url, headers=headers, json=data)
    if response.status_code not in (200, 204):
        print(f"⚠️ Failed to update article {article_id}: {response.status_code} - {response.text}")
    else:
        print(f"✅ Updated article {article_id} (cluster {cluster_id})")



def main():
    if  os.path.exists(CACHE_PATH):
        print("📁 Loading articles from disk...")
        raw_articles = load_articles_from_disk(CACHE_PATH)
    else:
        print("🔄 Fetching articles...")
        raw_articles = list(fetch_all_articles())
        save_articles_to_disk(raw_articles, CACHE_PATH)

    print(f"✅ Got {len(raw_articles)} articles")

    # Format first, then filter
    formatted_records = {
        str(article["Id"]): format_article_for_dedupe(article)
        for article in raw_articles
    }

    # Filter records: must have A and at least one other field
    def is_valid(record):
        if not record.get("A"):
            return False
        return any(record.get(f) for f in ["title", "url", "B", "D"])

    records = {
        rid: record for rid, record in formatted_records.items()
        if is_valid(record)
    }

    print(f"✅ Filtered to {len(records)} valid records (A + other fields present).")


    print("🧠 Running deduplication...")
    clustered_dupes = deduplicate_articles(records)

    print("💾 Saving results...")
    save_results(clustered_dupes, records, output_path="/app/output/deduped_results.csv")

    print("🔄 Updating articles in NOCO...")
    for cluster_id, (record_ids, confidence) in enumerate(clustered_dupes):
        for record_id in record_ids:
            update_article(str(record_id), cluster_id)

    print("✅ Finished updating articles in NOCO.")
if __name__ == '__main__':
    main()
