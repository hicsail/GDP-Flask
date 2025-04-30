from http.client import responses

import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import json
import os
import requests
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

CACHE_PATH = "/app/output/fetched_articles.json"
CHROMA_CACHE_PATH = "/app/output/chroma_cache.json"
CSV_PATH = "/app/database.csv"

loanIds = [
    "EG.056", "EG.058", "DJ.007", "UG.044", "UG.041", "DJ.003", "DJ.017", "NG.034", "GA.006",
    "GA.037", "GA.038", "GA.015", "GA.045", "ID.O.1", "GA.004", "GA.042", "LA.W.1", "ER.016",
    "KE.101", "ER.005", "ER.008", "ER.011", "ER.010", "ER.004", "EG.057", "ER.019", "ER.022",
    "ET.017", "ET.006", "ET.033", "CG.009.16", "CG.009.14", "CG.009.19", "CG.009.20", "CD.009.21",
    "ZM.O.01", "NG.C.01", "PAK.P. 1", "SAU.1", "AO.144", "ET.055", "AO.009.75", "AO.009.71",
    "AO.009.82", "CF.001", "CF.014", "BR.E.001", "NG.017", "NG.016", "EG.059", "EG.060",
    "GH.015.02", "GH.019", "GH.005", "CM.062", "CM.058", "CM.046", "CM.054", "CM.068", "CM.013",
    "CM.016", "CM.005", "CM.014", "CM.017", "CM.018", "CM.084", "KE.104", "JM.S.1", "EG.003",
    "BR.E.002", "BR.E.003", "AO.001.", "GH.001.A", "GH.001.B", "KE.105", "KE.106", "KE.071",
    "KE.058", "KE.010", "KE.107", "MA.027", "TN.030", "GA.003", "GA.044", "CM.010", "CM.028",
    "CM.008", "CM.039", "UG.018", "UG.034", "AO.148", "EG.059", "BR.015"
]

def fetch_all_articles(page_size=100, max_records=50000):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
    offset = 0
    fetched = 0
    while True:
        params = {
            "fields": "Id,originalTitle,articleUrl,a,b,c,d,webScrapedContent,originalContent,translatedTitle,translatedContent,cluster_id,source,Loans",
            "offset": offset,
            "limit": page_size,
            "where": "(FinanceClassification,isnot,null)",
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
            break
        for article in batch:
            yield article
        offset += page_size
        fetched += len(batch)
        if fetched >= max_records:
            break

def save_articles_to_disk(articles, path=CACHE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(articles)} articles to {path}")

def load_articles_from_disk(path=CACHE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def insert_into_chroma_if_needed(articles, model, cache_path=CHROMA_CACHE_PATH):
    if os.path.exists(cache_path):
        print("📁 Skipping ChromaDB insert — already cached")
        return
    insert_into_chroma(articles, model)
    with open(cache_path, "w") as f:
        f.write("inserted")

def weighted_text(article):
    return " ".join([
        (article.get("translatedTitle") or "") * 3,
        (article.get("webScrapedContent") or "") * 2,
        (article.get("translatedContent") or ""),
        (article.get("originalContent") or ""),
        (article.get("source") or "")
    ])

def deduplicate_articles_by_cluster(articles):
    seen_clusters = set()
    filtered_articles = []
    for article in articles:
        cluster_id = article.get("cluster_id")
        if cluster_id is None or cluster_id not in seen_clusters:
            seen_clusters.add(cluster_id)
            filtered_articles.append(article)
    return filtered_articles

def load_chroma_collection():
    chroma_client = chromadb.PersistentClient('/data')
    return chroma_client.get_or_create_collection("articles", metadata={"hnsw:space": "cosine"})

def embed_articles(articles, model):
    texts = [weighted_text(a) for a in articles]
    embeddings = model.encode(texts, normalize_embeddings=True)
    ids = [str(a["Id"]) for a in articles]
    return ids, texts, embeddings.tolist()

def insert_into_chroma(articles, model, batch_size=5000):
    collection = load_chroma_collection()
    ids, texts, embeddings = embed_articles(articles, model)
    cleaned_metadatas = [
        {k: (v if v is not None else "unknown") for k, v in a.items() if k not in {"Id", "cluster_id"}}
        for a in articles
    ]
    total = len(ids)
    for start in range(0, total, batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=cleaned_metadatas[start:end]
        )
        print(f"✅ Inserted batch {start}–{end} / {total}")
    print(f"🎉 Finished inserting {total} articles into ChromaDB")

def build_project_text(row):
    return " | ".join([
        str(row.get("Project Name", "")),
        str(row.get("Loan Sign Year", "")),
        str(row.get("Loan Type", "")),
        str(row.get("Borrowing Entity", "")),
        str(row.get("Country", "")),
        str(row.get("Region (UN)", "")),
        str(row.get("Borrowing Entity", "")),
        str(row.get("Reported Amount in millions", "")),
        str(row.get("Sector", "")),
        str(row.get("Sub-sector", "")),
        str(row.get("Lender", ""))
    ])

FIELD_WEIGHTS = {
    "translatedTitle": 5,
    "webScrapedContent": 3,
    "translatedContent": 2,
    "originalContent": 1,
    "source": 1
}

def weighted_text(article):
    return " ".join([
        (article.get(field) or "") * weight
        for field, weight in FIELD_WEIGHTS.items()
    ])

def rerank(project_text, candidates):
    pairs = [(project_text, c["document"]) for c in candidates]
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.predict(pairs)
    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)
        # Add fuzzy match as backup if semantic score is low
        candidates[i]["fuzzy_title_score"] = fuzz.token_set_ratio(project_text, candidates[i]["article"].get("translatedTitle", "")) / 100.0
    return sorted(candidates, key=lambda x: (x["rerank_score"], x["fuzzy_title_score"]), reverse=True)

def main():
    if os.path.exists(CACHE_PATH):
        print("📁 Loading articles from disk...")
        raw_articles = load_articles_from_disk(CACHE_PATH)
    else:
        print("🔄 Fetching articles...")
        raw_articles = list(fetch_all_articles())
        save_articles_to_disk(raw_articles, CACHE_PATH)
    print(f"✅ Got {len(raw_articles)} articles")
    filtered_articles = deduplicate_articles_by_cluster(raw_articles)
    print(f"🧹 Filtered to {len(filtered_articles)} articles (1 per cluster)")
    print("🔄 Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    insert_into_chroma_if_needed(filtered_articles, model)
    csv_df = pd.read_csv(CSV_PATH)
    collection = load_chroma_collection()
    print(f"📦 Chroma Collection contains {collection.count()} documents")
    for idx, row in csv_df.iterrows():
        loanId = row.get("BU ID")
        if not loanId or loanId not in loanIds:
            continue
        project_text = build_project_text(row)
        embedding = model.encode([project_text], normalize_embeddings=True)
        results = collection.query(
            query_embeddings=embedding,
            n_results=30,
            include=["metadatas", "documents", "distances"]
        )
        if not results["documents"][0]:
            continue
        candidates = []
        for meta, doc, dist in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
            score = 1 - dist
            candidates.append({"score": score, "article": meta, "document": doc})
        reranked = rerank(project_text, candidates)
        reranked = [r for r in reranked if r["rerank_score"] >= 0.5 or r["fuzzy_title_score"] > 0.85]
        if not reranked:
            continue
        print(f"\n🔍 Project {loanId}: {row.get('Project Name', 'Unnamed')}")
        for result in reranked[:3]:
            title = result["article"].get("translatedTitle") or result["article"].get("originalTitle") or "Untitled"
            source = result["article"].get("source") or "unknown"
            print(f"  ✅ {title} (from {source}) — Cosine: {result['score']:.2f}, Rerank: {result['rerank_score']:.2f}, Fuzzy: {result['fuzzy_title_score']:.2f}")

if __name__ == "__main__":
    main()
