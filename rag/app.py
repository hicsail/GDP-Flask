import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import json
import os
import requests
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


CACHE_PATH = "/app/output/fetched_articles.json"
CSV_PATH = "/app/database.csv"

def fetch_all_articles(page_size=100, max_records=50000):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
    offset = 0
    fetched = 0

    while True:
        params = {
            "fields": "Id,originalTitle,articleUrl,a,b,c,d,webScrapedContent,originalContent,translatedTitle,translatedContent,cluster_id,source",
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

def extract_article_text(article):
    return article.get("translatedContent") or article.get("originalContent") or ""

def deduplicate_articles_by_cluster(articles):
    seen_clusters = set()
    filtered_articles = []

    for article in articles:
        cluster_id = article.get("cluster_id")
        if cluster_id is None:
            filtered_articles.append(article)  # No cluster, keep it
        elif cluster_id not in seen_clusters:
            seen_clusters.add(cluster_id)
            filtered_articles.append(article)  # First time seeing this cluster

    return filtered_articles

def load_chroma_collection():
    chroma_client = chromadb.Client()
    return chroma_client.get_or_create_collection(name="articles")

def embed_articles(articles, model):
    texts = [extract_article_text(a) for a in articles]
    embeddings = model.encode(texts, normalize_embeddings=True)
    ids = [str(a["Id"]) for a in articles]
    return ids, texts, embeddings.tolist()

def insert_into_chroma(articles, model, batch_size=5000):
    collection = load_chroma_collection()
    ids, texts, embeddings = embed_articles(articles, model)

    # Clean metadata: remove 'Id', 'cluster_id', and any None values
    cleaned_metadatas = [
        {
            k: (v if v is not None else "unknown")
            for k, v in a.items()
            if k not in {"Id", "cluster_id"}
        }
        for a in articles
    ]

    total = len(ids)
    for start in range(0, total, batch_size):
        end = start + batch_size
        batch_ids = ids[start:end]
        batch_texts = texts[start:end]
        batch_embeddings = embeddings[start:end]
        batch_metadatas = cleaned_metadatas[start:end]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

        print(f"✅ Inserted batch {start}–{end} / {total}")

    print(f"🎉 Finished inserting {total} articles into ChromaDB")

def fetch_all_embeddings_from_chroma(collection):
    results = collection.get(include=["embeddings", "metadatas", "documents"])
    return results["ids"], results["embeddings"], results["metadatas"]


def cluster_embeddings(embeddings, metadatas, eps=0.25, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(np.array(embeddings))

    clustered = {}
    for label, meta in zip(labels, metadatas):
        if label == -1:
            continue  # Noise
        clustered.setdefault(label, []).append(meta)

    return clustered

def load_and_embed_csv(csv_path, model):
    df = pd.read_csv(csv_path)

    # Choose fields to include in embedding
    def combine_fields(row):
        return " | ".join([
            str(row.get("Project Name", "")),
            str(row.get("Narrative", "")),
            str(row.get("Loan Type", "")),
            str(row.get("Sector", "")),
            str(row.get("Country", "")),
            str(row.get("Reported Amount in millions", "")) + " " + str(row.get("Currency", ""))
        ])

    combined_texts = df.apply(combine_fields, axis=1).tolist()
    embeddings = model.encode(combined_texts, normalize_embeddings=True)
    return df, embeddings

def get_cluster_centroids(clustered_articles, model):
    centroids = {}
    for cluster_id, articles in clustered_articles.items():
        texts = [
            a.get("translatedContent") or a.get("originalContent") or ""
            for a in articles
        ]
        if texts:
            vecs = model.encode(texts, normalize_embeddings=True)
            centroids[cluster_id] = np.mean(vecs, axis=0)
    return centroids

def match_clusters_to_csv(centroids, csv_embeddings, csv_df, threshold=0.75):
    matches = {}

    for cluster_id, centroid in centroids.items():
        sims = cosine_similarity([centroid], csv_embeddings)[0]
        max_score = np.max(sims)
        max_index = np.argmax(sims)

        if max_score >= threshold:
            matches[cluster_id] = {
                "match_score": max_score,
                "csv_row": csv_df.iloc[max_index].to_dict()
            }
        else:
            matches[cluster_id] = None  # No match

    return matches

def match_top_3_clusters_to_csv(centroids, csv_embeddings, csv_df):
    matches = {}

    for cluster_id, centroid in centroids.items():
        sims = cosine_similarity([centroid], csv_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:3]  # Top 3 scores

        top_matches = []
        for idx in top_indices:
            row = csv_df.iloc[idx].to_dict()
            top_matches.append({
                "score": float(sims[idx]),  # Convert from np.float32 for JSON-safe output
                "row": row
            })

        matches[cluster_id] = top_matches

    return matches

def main():
    if  os.path.exists(CACHE_PATH):
        print("📁 Loading articles from disk...")
        raw_articles = load_articles_from_disk(CACHE_PATH)
    else:
        print("🔄 Fetching articles...")
        raw_articles = list(fetch_all_articles())
        save_articles_to_disk(raw_articles, CACHE_PATH)

    print(f"✅ Got {len(raw_articles)} articles")

    # Deduplicate by cluster_id
    filtered_articles = deduplicate_articles_by_cluster(raw_articles)
    print(f"🧹 Filtered to {len(filtered_articles)} articles (1 per cluster)")

    # Group by similarity
    print("🔄 Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("🔄 Loading articles into ChromaDB...")
    insert_into_chroma(filtered_articles, model)
    print("🧠 Running clustering...")

    collection = load_chroma_collection()
    ids, embeddings, metadatas = fetch_all_embeddings_from_chroma(collection)
    clustered = cluster_embeddings(embeddings, metadatas)

    csv_df, csv_embeddings = load_and_embed_csv(CSV_PATH, model)
    centroids = get_cluster_centroids(clustered, model)
    matches = match_top_3_clusters_to_csv(centroids, csv_embeddings, csv_df)

    for cluster_id, top_matches in matches.items():
        print(f"\n📚 Cluster {cluster_id} — Top 3 Matches:")

        for match in top_matches:
            row = match["row"]
            print(f"  ✅ {row.get('Project Name', 'Unknown')} ({row.get('Country', 'Unknown')}) — Score: {match['score']:.2f}")

        print("  📰 Articles in this cluster:")
        for article in clustered[cluster_id]:
            title = article.get('translatedTitle') or article.get('originalTitle') or 'Untitled'
            source = article.get('source', 'unknown')
            print(f"  - {title} (from {source})")



if __name__ == "__main__":
    main()