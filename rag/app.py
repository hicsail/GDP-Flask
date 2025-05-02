import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import json
import os
import requests

CSV_PATH = "/app/database.csv"
CHROMA_CACHE_PATH = "/app/output/chroma_cache.json"

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

FIELD_WEIGHTS = {
    "Project Name": 5,
    "Loan Sign Year": 1,
    "Loan Type": 1,
    "Borrowing Entity": 2,
    "Country": 2,
    "Region (UN)": 1,
    "Reported Amount in millions": 1,
    "Sector": 2,
    "Sub-sector": 2,
    "Lender": 2
}

def build_project_text(row):
    return " | ".join([
        (str(row.get(field, "")) * FIELD_WEIGHTS.get(field, 1)) for field in FIELD_WEIGHTS
    ])

def load_chroma_collection():
    chroma_client = chromadb.PersistentClient('/data')
    return chroma_client.get_or_create_collection("projects", metadata={"hnsw:space": "cosine"})

def insert_projects_into_chroma(csv_df, model):
    csv_df = csv_df[csv_df["BU ID"].notna() & csv_df["BU ID"].isin(loanIds)]
    csv_df = csv_df.drop_duplicates(subset="BU ID")
    if os.path.exists(CHROMA_CACHE_PATH):
        print("📁 Skipping ChromaDB insert — already cached")
        return
    collection = load_chroma_collection()
    texts = [build_project_text(row) for _, row in csv_df.iterrows()]
    embeddings = model.encode(texts, normalize_embeddings=True)
    ids = [str(row["BU ID"]) for _, row in csv_df.iterrows()]
    metadatas = [
        {"BU ID": str(row["BU ID"]), "Project Name": row.get("Project Name", "")}
        for _, row in csv_df.iterrows()
    ]
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    with open(CHROMA_CACHE_PATH, "w") as f:
        f.write("inserted")
    print(f"🎉 Inserted {len(ids)} projects into ChromaDB")

def fetch_all_articles(page_size=100, max_records=50000):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
    offset = 0
    fetched = 0
    while True:
        params = {
            "fields": "Id,translatedTitle,webScrapedContent,translatedContent,originalContent,source",
            "offset": offset,
            "limit": page_size,
            "where": "(BU ID,isnot,null)",
        }
        response = requests.get(db_url, headers=headers, params=params)
        if response.status_code != 200:
            break
        try:
            data = response.json().get("list", [])
        except Exception as e:
            print(f"❌ Failed to decode JSON: {e}")
            break
        if not data:
            break
        for article in data:
            yield article
        offset += page_size
        fetched += len(data)
        if fetched >= max_records:
            break

def weighted_article_text(article):
    return " ".join([
        (article.get("translatedTitle") or "") * 3,
        (article.get("webScrapedContent") or "") * 2,
        (article.get("translatedContent") or ""),
        (article.get("originalContent") or ""),
        (article.get("source") or "")
    ])

def rerank(article_text, candidates):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(article_text, c["document"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)
        candidates[i]["buid"] = candidates[i]["article"].get("BU ID")
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

def patch_article_buid(article_id, buids_and_scores):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
    possible_buids = ", ".join([f"{buid} - {round(score, 2)}" for buid, score in buids_and_scores])
    patch_data = {
        "Id": article_id,
        "Possible.BUIDs": possible_buids,
    }
    try:
        response = requests.patch(f"{db_url}", headers=headers, json=patch_data)
        if response.status_code != 200:
            print(f"⚠️ Failed to patch {article_id} — {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Exception patching {article_id}: {e}")

CACHE_PATH = "/app/output/fetched_articles.json"

def save_articles_to_disk(articles, path=CACHE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(articles)} articles to {path}")

def load_articles_from_disk(path=CACHE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    csv_df = pd.read_csv(CSV_PATH)
    insert_projects_into_chroma(csv_df, model)
    collection = load_chroma_collection()

    update_map = {}
    if os.path.exists(CACHE_PATH):
        print("📁 Loading articles from disk...")
        articles = load_articles_from_disk(CACHE_PATH)
    else:
        print("🔄 Fetching articles...")
        articles = list(fetch_all_articles())
        save_articles_to_disk(articles)
    print(f"✅ Retrieved {len(articles)} articles")

    print("🔄 Processing articles...")
    total = len(articles)
    count = 0
    for article in articles:
        count += 1
        article_id = article.get("Id")
        print(f"{round((count + 1)/total* 100, 2)}%")
        if not article_id:
            continue
        article_text = weighted_article_text(article)
        embedding = model.encode([article_text], normalize_embeddings=True)
        results = collection.query(
            query_embeddings=embedding,
            n_results=30,
            include=["metadatas", "documents", "distances"]
        )
        candidates = []
        for meta, doc, dist in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
            score = 1 - dist
            candidates.append({"score": score, "article": meta, "document": doc})
        reranked = rerank(article_text, candidates)
        top_matches = [(r["buid"], r["rerank_score"]) for r in reranked][:3]
        if top_matches:
            patch_article_buid(article_id, top_matches)
            update_map[article_id] = top_matches

    print("📤 Done updating matched BU IDs.")

if __name__ == "__main__":
    main()
