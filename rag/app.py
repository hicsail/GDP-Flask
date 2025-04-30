from http.client import responses

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
CHROMA_CACHE_PATH = "/app/output/chroma_cache.json"
CSV_PATH = "/app/database.csv"

loanIds = [
    "EG.056",
    "EG.058",
    "DJ.007",
    "UG.044",
    "UG.041",
    "DJ.003",
    "DJ.017",
    "NG.034",
    "GA.006",
    "GA.037",
    "GA.038",
    "GA.015",
    "GA.045",
    "ID.O.1",
    "GA.004",
    "GA.042",
    "LA.W.1",
    "ER.016",
    "KE.101",
    "ER.005",
    "ER.008",
    "ER.011",
    "ER.010",
    "ER.004",
    "EG.057",
    "ER.019",
    "ER.022",
    "ET.017",
    "ET.006",
    "ET.033",
    "CG.009.16",
    "CG.009.14",
    "CG.009.19",
    "CG.009.20",
    "CD.009.21",
    "ZM.O.01",
    "NG.C.01",
    "PAK.P. 1",
    "SAU.1",
    "AO.144",
    "ET.055",
    "AO.009.75",
    "AO.009.71",
    "AO.009.82",
    "CF.001",
    "CF.014",
    "BR.E.001",
    "NG.017",
    "NG.016",
    "EG.059",
    "EG.060",
    "GH.015.02",
    "GH.019",
    "GH.005",
    "CM.062",
    "CM.058",
    "CM.046",
    "CM.054",
    "CM.068",
    "CM.013",
    "CM.016",
    "CM.005",
    "CM.014",
    "CM.017",
    "CM.018",
    "CM.084",
    "KE.104",
    "JM.S.1",
    "EG.003",
    "BR.E.002",
    "BR.E.003",
    "AO.001.",
    "GH.001.A",
    "GH.001.B",
    "KE.105",
    "KE.106",
    "KE.071",
    "KE.058",
    "KE.010",
    "KE.107",
    "MA.027",
    "TN.030",
    "GA.003",
    "GA.044",
    "CM.010",
    "CM.028",
    "CM.008",
    "CM.039",
    "UG.018",
    "UG.034",
    "AO.148",
    "EG.059",
    "BR.015",
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
            "where": "(FinanceClassification,isnot,null)",  # This ensures there is at least one Loan
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

def insert_into_chroma_if_needed(articles, model, cache_path=CHROMA_CACHE_PATH):
    if os.path.exists(cache_path):
        print("📁 Skipping ChromaDB insert — already cached")
        return

    insert_into_chroma(articles, model)

    with open(cache_path, "w") as f:
        f.write("inserted")


def extract_article_text(article):
    parts = [
        article.get("originalTitle", ""),
        article.get("articleUrl", ""),
        article.get("a", ""),
        article.get("b", ""),
        article.get("c", ""),
        article.get("d", ""),
        article.get("a", ""),
        article.get("b", ""),
        article.get("c", ""),
        article.get("d", ""),
        article.get("a", ""),
        article.get("b", ""),
        article.get("c", ""),
        article.get("d", ""),
        article.get("a", ""),
        article.get("b", ""),
        article.get("c", ""),
        article.get("d", ""),
        article.get("webScrapedContent", ""),
        article.get("originalContent", ""),
        article.get("translatedTitle", ""),
        article.get("translatedContent", ""),
        article.get("source", "")
    ]
    return " ".join(p for p in parts if p)


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
    chroma_client = chromadb.PersistentClient('/data')
    return chroma_client.get_or_create_collection("articles", metadata={"hnsw:space": "cosine"})

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

    LENDER_MAP = {
        "CDB": "China Development Bank",
        "CHEXIM": "China Export-Import Bank",
    }

    def expand_lenders(lender_str):
        return ", ".join(LENDER_MAP.get(l.strip(), l.strip()) for l in lender_str.split(","))

    # Choose fields to include in embedding
    def combine_fields(row):
        return " | ".join([
            str(row.get("Project Name", "")),
            str(row.get("Loan Sign Year", "")),
            str(row.get("Project Status", "")),
            str(row.get("Loan Type", "")),
            str(row.get("Borrowing Entity", "")),
            expand_lenders(str(row.get("Lender", "")))
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

def getSourceText(row, column="Source 1"):
    url = row.get(column)
    if url:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            else:
                return ""
        except requests.RequestException as e:
            return ""
    return ""

LENDER_MAP = {
    "CDB": "China Development Bank",
    "CHEXIM": "China Export-Import Bank",
}

def expand_lenders(lender_str):
    return ", ".join(LENDER_MAP.get(l.strip(), l.strip()) for l in lender_str.split(","))

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
        expand_lenders(str(row.get("Lender", ""))),
        #getSourceText(row, "Source 1"),
        #getSourceText(row, "Source 2"),
    ])


def search_project_in_articles(row, model, article_embeddings, metadatas, top_k=5, min_score=0.6):
    """
    Searches for the most relevant articles for a given CSV row using cosine similarity on embeddings.

    Args:
        row (pd.Series): A single row from the loan/project CSV.
        model (SentenceTransformer): The sentence transformer model.
        article_embeddings (List[List[float]]): Preloaded ChromaDB article embeddings.
        metadatas (List[dict]): Corresponding article metadata.
        top_k (int): How many top results to return.
        min_score (float): Minimum similarity threshold.

    Returns:
        List[dict]: Matched articles with score and metadata.
    """


    project_text = build_project_text(row)
    embedding = model.encode([project_text], normalize_embeddings=True)

    sims = cosine_similarity(embedding, article_embeddings)[0]
    top_indices = np.argsort(sims)[::-1]

    matches = []
    for idx in top_indices:
        score = float(sims[idx])
        if score < min_score:
            continue
        matches.append({
            "score": score,
            "article": metadatas[idx]
        })
        if len(matches) >= top_k:
            break

    return matches

def search_all_projects(csv_df, model, article_embeddings, metadatas, top_k=3, min_score=0.8):
    for idx, row in csv_df.iterrows():
        print(row)
        matches = search_project_in_articles(row, model, article_embeddings, metadatas, top_k=top_k, min_score=min_score)

        if not matches:
            continue  # Skip projects with no strong matches

        print(f"\n🔍 Project {idx}: {row.get('Project Name', 'Unnamed')}")

        for match in matches:
            title = match["article"].get("translatedTitle") or match["article"].get("originalTitle") or "Untitled"
            source = match["article"].get("source", "unknown")
            print(f"  ✅ {title} (from {source}) — Score: {match['score']:.2f}")


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
    # Insert if needed
    insert_into_chroma_if_needed(filtered_articles, model)

    # Search
    csv_df = pd.read_csv(CSV_PATH)
    collection = load_chroma_collection()

    print(f"📦 Chroma Collection contains {collection.count()} documents")

    # Optional debug
    print("\n🔎 DEBUGGING CHROMA QUERY")
    test_embedding = model.encode(["Test project"], normalize_embeddings=True)
    test_results = collection.query(
        query_embeddings=test_embedding,
        n_results=3,
        include=["metadatas", "distances"]
    )
    for meta, dist in zip(test_results['metadatas'][0], test_results['distances'][0]):
        score = 1 - dist
        title = meta.get("translatedTitle") or meta.get("originalTitle") or "Untitled"
        print(f"  🧪 Score: {score:.2f}, Title: {title}")

    # Step 4: Search all projects
    print("\n🚀 Searching all projects...\n")
    for idx, row in csv_df.iterrows():
        # Skip projects with no loan IDs

        loanId = row.get("BU ID")
        if not loanId or loanId not in loanIds:
            continue

        project_text = build_project_text(row)
        embedding = model.encode([project_text], normalize_embeddings=True)

        results = collection.query(
            query_embeddings=embedding,
            n_results=3,
            include=["metadatas", "distances"],
        )

        if not results["metadatas"][0]:
            continue

        print(f"\n🔍 Project {loanId}: {row.get('Project Name', 'Unnamed')}")

        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            score = 1 - dist  # cosine similarity
            title = meta.get("translatedTitle") or meta.get("originalTitle") or "Untitled"
            source = meta.get("source") or "unknown"

            if score < 0.69:
                continue

            print(f"  ✅ {title} (from {source}) — Score: {score:.2f}")

            # Patch article with matched BUID
            #patch_payload = {
            #    "Id": article_id,
            #    "Possible.BUIDs": buid,
            #}

            #requests.patch(
            #    os.getenv("NOCO_DB_URL"),  # Full PATCH URL to the article
            #    headers=headers,
            #    json=patch_payload,
            #)


if __name__ == "__main__":
    main()