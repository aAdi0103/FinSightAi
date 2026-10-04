import json
import logging
import os
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
INDEX_FILE = os.path.join(MODELS_DIR, "rag_vectorizer.joblib")
MATRIX_FILE = os.path.join(MODELS_DIR, "rag_matrix.joblib")
DOC_STORE_FILE = os.path.join(MODELS_DIR, "rag_doc_store.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_prepare_documents(kb_dir: str = KNOWLEDGE_BASE_DIR) -> List[Dict[str, Any]]:
    
    documents: List[Dict[str, Any]] = []

    # 1. RBI Regulations
    rbi_path = os.path.join(kb_dir, "rbi_regulations.json")
    if os.path.exists(rbi_path):
        with open(rbi_path, "r", encoding="utf-8") as f:
            rbi_data = json.load(f)
            for item in rbi_data:
                search_text = (
                    f"Authority: RBI (Reserve Bank of India) | Title: {item['title']} | "
                    f"Section: {item['section']} | Source: {item['source']} | "
                    f"Content: {item['content']} | Triggers: {' '.join(item.get('applicable_triggers', []))}"
                )
                documents.append({
                    "doc_id": item["id"],
                    "category": "REGULATION",
                    "authority": "RBI (Reserve Bank of India)",
                    "title": item["title"],
                    "section": item["section"],
                    "source": item["source"],
                    "content": item["content"],
                    "triggers": item.get("applicable_triggers", []),
                    "searchable_text": search_text,
                })

    # 2. FATF Recommendations
    fatf_path = os.path.join(kb_dir, "fatf_recommendations.json")
    if os.path.exists(fatf_path):
        with open(fatf_path, "r", encoding="utf-8") as f:
            fatf_data = json.load(f)
            for item in fatf_data:
                search_text = (
                    f"Authority: FATF (Financial Action Task Force) | Title: {item['title']} | "
                    f"Section: {item['section']} | Source: {item['source']} | "
                    f"Content: {item['content']} | Triggers: {' '.join(item.get('applicable_triggers', []))}"
                )
                documents.append({
                    "doc_id": item["id"],
                    "category": "REGULATION",
                    "authority": "FATF (Financial Action Task Force)",
                    "title": item["title"],
                    "section": item["section"],
                    "source": item["source"],
                    "content": item["content"],
                    "triggers": item.get("applicable_triggers", []),
                    "searchable_text": search_text,
                })

    # 3. FinCEN Typologies
    fincen_path = os.path.join(kb_dir, "fincen_typologies.json")
    if os.path.exists(fincen_path):
        with open(fincen_path, "r", encoding="utf-8") as f:
            fincen_data = json.load(f)
            for item in fincen_data:
                search_text = (
                    f"Authority: FinCEN | Title: {item['title']} | "
                    f"Section: {item['section']} | Source: {item['source']} | "
                    f"Content: {item['content']} | Triggers: {' '.join(item.get('applicable_triggers', []))}"
                )
                documents.append({
                    "doc_id": item["id"],
                    "category": "REGULATION",
                    "authority": "FinCEN (Financial Crimes Enforcement Network)",
                    "title": item["title"],
                    "section": item["section"],
                    "source": item["source"],
                    "content": item["content"],
                    "triggers": item.get("applicable_triggers", []),
                    "searchable_text": search_text,
                })

    # 4. Historical Investigation Cases
    cases_path = os.path.join(kb_dir, "historical_cases.json")
    if os.path.exists(cases_path):
        with open(cases_path, "r", encoding="utf-8") as f:
            cases_data = json.load(f)
            for item in cases_data:
                search_text = (
                    f"Case Precedent: {item['case_id']} | Title: {item['case_title']} | "
                    f"Typology: {item['typology_category']} | Transaction Type: {item['transaction_type']} | "
                    f"Summary: {item['case_summary']} | Findings: {item['investigative_findings']} | "
                    f"Resolution: {item['resolution_and_action']}"
                )
                documents.append({
                    "doc_id": item["case_id"],
                    "category": "HISTORICAL_CASE",
                    "authority": "Historical Bank Investigation Precedent",
                    "title": item["case_title"],
                    "typology": item["typology_category"],
                    "transaction_type": item["transaction_type"],
                    "content": item["case_summary"],
                    "findings": item["investigative_findings"],
                    "resolution": item["resolution_and_action"],
                    "searchable_text": search_text,
                })

    logger.info("Loaded %d knowledge documents across regulations and precedent cases.", len(documents))
    return documents


def build_vector_index(
    documents: List[Dict[str, Any]],
    models_dir: str = MODELS_DIR,
) -> None:
    
    os.makedirs(models_dir, exist_ok=True)

    texts = [doc["searchable_text"] for doc in documents]
    logger.info("Building domain semantic vector index across %d documents...", len(texts))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        stop_words="english",
    )
    doc_matrix = vectorizer.fit_transform(texts)

    # Persist Index artifacts
    vec_path = os.path.join(models_dir, "rag_vectorizer.joblib")
    mat_path = os.path.join(models_dir, "rag_matrix.joblib")
    doc_store_path = os.path.join(models_dir, "rag_doc_store.json")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(doc_matrix, mat_path)

    with open(doc_store_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    logger.info("Saved Vectorizer to %s", vec_path)
    logger.info("Saved Document Matrix to %s", mat_path)
    logger.info("Saved Document Store to %s", doc_store_path)


def run_indexing_pipeline():
    """Main execution function for indexing knowledge base."""
    logger.info("=" * 60)
    logger.info("BUILDING RAG KNOWLEDGE BASE VECTOR INDEX")
    logger.info("=" * 60)
    docs = load_and_prepare_documents()
    build_vector_index(docs)
    logger.info("=" * 60)
    logger.info("INDEXING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_indexing_pipeline()
