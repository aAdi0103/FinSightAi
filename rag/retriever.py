"""
rag/retriever.py
================
Evidence Retriever for the Financial Transaction Investigation System.

Loads the vector index and metadata store to perform semantic similarity
search across RBI regulations, FATF recommendations, FinCEN advisories, and
historical fraud cases.

Author: AI-Assisted Financial Transaction Risk Investigation System
"""

import json
import logging
import os
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODELS_DIR = os.getenv("MODELS_DIR", "models")
VEC_FILE = os.path.join(MODELS_DIR, "rag_vectorizer.joblib")
MATRIX_FILE = os.path.join(MODELS_DIR, "rag_matrix.joblib")
DOC_STORE_FILE = os.path.join(MODELS_DIR, "rag_doc_store.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RETRIEVER ENGINE (SINGLETON)
# ---------------------------------------------------------------------------

class EvidenceRetriever:
    """Singleton class for fast vector similarity search."""
    _instance = None

    def __init__(self):
        self.vectorizer = None
        self.doc_matrix = None
        self.doc_store = []
        self._load()

    def _load(self):
        if not os.path.exists(VEC_FILE) or not os.path.exists(MATRIX_FILE) or not os.path.exists(DOC_STORE_FILE):
            logger.info("RAG Index or Docstore missing. Building dynamically...")
            from rag.indexer import run_indexing_pipeline
            run_indexing_pipeline()

        logger.info("Loading RAG Vectorizer from '%s'...", VEC_FILE)
        self.vectorizer = joblib.load(VEC_FILE)

        logger.info("Loading RAG Document Matrix from '%s'...", MATRIX_FILE)
        self.doc_matrix = joblib.load(MATRIX_FILE)

        logger.info("Loading Document Store from '%s'...", DOC_STORE_FILE)
        with open(DOC_STORE_FILE, "r", encoding="utf-8") as f:
            self.doc_store = json.load(f)

        logger.info("Retriever initialized with %d indexed passages.", len(self.doc_store))

    @classmethod
    def get_instance(cls) -> "EvidenceRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Execute semantic similarity search against indexed regulations and cases.

        Args:
            query: Natural language query (synthesized from risk indicators).
            top_k: Number of documents to retrieve per category.

        Returns:
            Dictionary partitioned into regulations, historical_cases, and a formatted prompt block.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()

        # Sort indices by descending cosine similarity
        sorted_indices = np.argsort(-similarities)

        regulations: List[Dict[str, Any]] = []
        historical_cases: List[Dict[str, Any]] = []

        for idx in sorted_indices:
            score = float(similarities[idx])
            doc = self.doc_store[idx]

            match_data = {
                "doc_id": doc["doc_id"],
                "similarity_score": round(score, 4),
                "title": doc["title"],
                "authority": doc.get("authority", "Regulatory Entity"),
                "category": doc["category"],
            }

            if doc["category"] == "REGULATION":
                match_data["section"] = doc.get("section", "")
                match_data["source"] = doc.get("source", "")
                match_data["content"] = doc.get("content", "")
                if len(regulations) < top_k:
                    regulations.append(match_data)
            else:
                match_data["typology"] = doc.get("typology", "")
                match_data["summary"] = doc.get("content", "")
                match_data["findings"] = doc.get("findings", "")
                match_data["resolution"] = doc.get("resolution", "")
                if len(historical_cases) < top_k:
                    historical_cases.append(match_data)

            if len(regulations) >= top_k and len(historical_cases) >= top_k:
                break

        # Build structured context block for LLM prompt
        context_lines = ["=== RELEVANT REGULATORY DIRECTIVES ==="]
        for reg in regulations:
            context_lines.append(
                f"- [{reg['doc_id']}] {reg['source']} ({reg['section']}): {reg['title']}\n  Summary: {reg['content']}"
            )

        context_lines.append("\n=== MATCHING HISTORICAL INVESTIGATION CASES ===")
        for case in historical_cases:
            context_lines.append(
                f"- [{case['doc_id']}] {case['title']} (Typology: {case['typology']})\n  Summary: {case['summary']}\n  Findings: {case['findings']}\n  Resolution: {case['resolution']}"
            )

        raw_context_block = "\n".join(context_lines)

        return {
            "query": query,
            "regulations": regulations,
            "historical_cases": historical_cases,
            "context_block": raw_context_block,
        }


def retrieve_evidence(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Public helper function to query evidence."""
    retriever = EvidenceRetriever.get_instance()
    return retriever.search(query=query, top_k=top_k)


if __name__ == "__main__":
    test_query = "rapid account draining to zero balance uncredited recipient mismatch transfer"
    print(f"\n--- Testing Evidence Retrieval for query: '{test_query}' ---")
    results = retrieve_evidence(test_query, top_k=2)
    print("\nRetrieved Regulations:")
    for r in results["regulations"]:
        print(f"  * [{r['doc_id']}] {r['title']} (Score: {r['similarity_score']})")
    print("\nRetrieved Cases:")
    for c in results["historical_cases"]:
        print(f"  * [{c['doc_id']}] {c['title']} (Score: {c['similarity_score']})")
