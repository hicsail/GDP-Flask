import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import json
import os
import requests

CSV_PATH = "/app/database.csv"
CHROMA_CACHE_PATH = "/app/output/chroma_cache.json"
CACHE_PATH = "/app/output/fetched_articles.json"
CHROMA_PATH = "/data"

# Proven set of BU IDs to include
loanIds = set([
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
])

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
    # Use field names for context and concatenate all relevant fields, skipping empty/null
    fields = [
        ("Project Name", 5),
        ("Loan Sign Year", 1),
        ("Loan Type", 1),
        ("Borrowing Entity", 2),
        ("Country", 2),
        ("Region (UN)", 1),
        ("Reported Amount in millions", 1),
        ("Sector", 2),
        ("Sub-sector", 2),
        ("Lender", 2)
    ]
    parts = []
    for field, weight in fields:
        value = row.get(field, "")
        if pd.notnull(value) and str(value).strip():
            parts.append(f"{field}: " + (str(value) + " ") * weight)
    return "| ".join(parts)

def load_chroma_collection():
    chroma_client = chromadb.PersistentClient(CHROMA_PATH)
    return chroma_client.get_or_create_collection("projects", metadata={"hnsw:space": "cosine"})

def insert_projects_into_chroma(csv_df, model):
    csv_df = csv_df[csv_df["BU ID"].isin(loanIds) & csv_df["BU ID"].notna()]
    csv_df = csv_df.drop_duplicates(subset="BU ID")
    if os.path.exists(CHROMA_CACHE_PATH):
        print("📁 Skipping ChromaDB insert — already cached")
        return
    collection = load_chroma_collection()
    texts = [build_project_text(row) for _, row in csv_df.iterrows()]
    embeddings = model.encode(texts, normalize_embeddings=True)
    ids = [str(row["BU ID"]) for _, row in csv_df.iterrows()]
    metadatas = [
        {"BU ID": str(row["BU ID"]), "Project Name": row.get("Project Name", ""), "Country": row.get("Country", "")}
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
            "fields": "Id,BU ID,translatedTitle,webScrapedContent,translatedContent,originalTitle,originalContent,source,AIScore4_Justification,a,b,c,d",
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
    # Use field names for context and concatenate all relevant fields
    parts = []
    if article.get("translatedTitle"):
        parts.append("Title: " + article.get("translatedTitle") * 3)
    if article.get("webScrapedContent"):
        parts.append("Content: " + article.get("webScrapedContent") * 2)
    if article.get("translatedContent"):
        parts.append("TranslatedContent: " + article.get("translatedContent") * 2)
    if article.get("originalContent"):
        parts.append("OriginalContent: " + article.get("originalContent"))
    if article.get("originalTitle"):
        parts.append("OriginalTitle: " + article.get("originalTitle") * 3)
    if article.get("a"):
        parts.append("Country: " + article.get("a") * 4)
    if article.get("b"):
        parts.append("Lender: " + article.get("b") * 4)
    if article.get("c"):
        parts.append("Keywords: " + article.get("c") * 4)
    if article.get("d"):
        parts.append("Project: " + article.get("d") * 4)
    if article.get("AIScore4_Justification"):
        parts.append("AIJustification: " + article.get("AIScore4_Justification"))
    if article.get("source"):
        parts.append("Source: " + article.get("source"))

    return " | ".join(parts)

def rerank(article_text, candidates, article_country):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scored = []
    for c in candidates:
        buid_country = c["article"].get("Country", "")
        base_score = reranker.predict([(article_text, c["document"])])[0]
        if buid_country and article_country:
            if buid_country.strip().lower() == article_country.strip().lower():
                base_score += 0.05
            else:
                base_score -= 0.1
        c["rerank_score"] = float(base_score)
        c["buid"] = c["article"].get("BU ID")
        scored.append(c)
    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

def patch_article_buid(article_id, buids_and_scores):
    db_url = os.getenv("NOCO_DB_URL")
    headers = {"xc-token": os.getenv("NOCO_XC_TOKEN")}
    possible_buids = ", ".join([f"{buid} - {round(score, 2)}" for buid, score in buids_and_scores])
    patch_data = {"Id": article_id, "Possible.BUIDs": possible_buids}
    try:
        response = requests.patch(f"{db_url}", headers=headers, json=patch_data)
        if response.status_code != 200:
            print(f"⚠️ Failed to patch {article_id} — {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Exception patching {article_id}: {e}")

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

    if os.path.exists(CACHE_PATH):
        print("📁 Loading articles from disk...")
        articles = load_articles_from_disk(CACHE_PATH)
    else:
        print("🔄 Fetching articles...")
        articles = list(fetch_all_articles())
        save_articles_to_disk(articles)
    print(f"✅ Retrieved {len(articles)} articles")

    matched_count = 0
    for idx, article in enumerate(articles):
        article_id = article.get("Id")
        article_country = article.get("a")
        print(f"🔎 Processing article {idx+1}/{len(articles)}")
        if not article_id or not article_country:
            continue

        # Filter loans by country - exact match since we're using standardized country names
        country_loans = csv_df[csv_df["Country"].str.strip().str.lower() == article_country.strip().lower()]
        country_buids = set(country_loans["BU ID"].dropna().astype(str))
        if not country_buids:
            continue

        article_text = weighted_article_text(article)
        embedding = model.encode([article_text], normalize_embeddings=True)

        # Query only loans from that country with more candidates
        results = collection.query(
            query_embeddings=embedding,
            n_results=75,  # Increased from 50 to 75 to get even more candidates
            where={"BU ID": {"$in": list(country_buids)}},
            include=["metadatas", "documents", "distances"]
        )

        candidates = [
            {"score": 1 - dist, "article": meta, "document": doc}
            for meta, doc, dist in zip(results["metadatas"][0], results["documents"][0], results["distances"][0])
        ]

        reranked = rerank(article_text, candidates, article_country)
        top_matches = [(r["buid"], r["rerank_score"]) for r in reranked][:5]
        if top_matches:
            patch_article_buid(article_id, top_matches)
            matched_count += 1

    print(f"📤 Done updating matched BU IDs. Matched {matched_count}/{len(articles)} articles.")

if __name__ == "__main__":
    main()

