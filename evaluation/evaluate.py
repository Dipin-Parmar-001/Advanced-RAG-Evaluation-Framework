import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval_engine import AdvancedRetrievalEngine
from embeddings import OpenRouterEmbeddings, GeminiEmbeddings
from langchain_chroma import Chroma

from metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    ndcg_at_k
)
import pandas as pd

DB_DIR = "db"
embeddings = GeminiEmbeddings()

vector_store = Chroma(
    collection_name="pdf_embeddings",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

retrieval_engine = AdvancedRetrievalEngine()
with open("evaluation/dataset_new.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

experiments = [
    ("baseline", False),
    ("reranker", True)
]

rows = []
report = []

for experiment_name, use_reranker in experiments:
    p1_score = []
    p3_score = []
    p5_score = []
    recall_score = []
    mrr_score = []
    hit_rate_score = []
    ndcg_score = []

    for sample in dataset[0:2]:
        query = sample["query"]
        relevant_ids = sample["relevant_ids"]
        
        result = retrieval_engine.retrieve(query=query, vector_store=vector_store, k=5, rewrite=False, use_reranker=use_reranker)

        retrieved_ids = [
            chunk.document.metadata["chunk_id"] for chunk in result.chunks
        ]
        p1_score.append(
            precision_at_k(retrieved_ids, relevant_ids, k=1)
        )
        p3_score.append(
            precision_at_k(retrieved_ids, relevant_ids, k=3)
        )
        p5_score.append(
            precision_at_k(retrieved_ids, relevant_ids, k=5)
        )

        recall_score.append(
            recall_at_k(retrieved_ids, relevant_ids, k=5)
        )

        mrr_score.append(
            reciprocal_rank(retrieved_ids, relevant_ids)
        )

        hit_rate_score.append(
            hit_rate_at_k(retrieved_ids, relevant_ids, k=5)
        )

        ndcg_score.append(
            ndcg_at_k(retrieved_ids, relevant_ids, k=5)
        )

        print("Expected:", relevant_ids)
        print("Retrieved:", retrieved_ids)

        rows.append({
            "experiment": experiment_name,
            "query": query,
            "precision@1": p1_score[-1],
            "precision@3": p3_score[-1],
            "precision@5": p5_score[-1],
            "recall@5": recall_score[-1],
            "mrr": mrr_score[-1],
            "hit_rate@5": hit_rate_score[-1],
            "ndcg@5": ndcg_score[-1],
        })
    report.append({
        "Experiment": experiment_name,
        "P@1": sum(p1_score)/len(p1_score),
        "P@3": sum(p3_score)/len(p3_score),
        "P@5": sum(p5_score)/len(p5_score),
        "Recall@5": sum(recall_score)/len(recall_score),
        "MRR": sum(mrr_score)/len(mrr_score),
        "HitRate@5": sum(hit_rate_score)/len(hit_rate_score),
        "NDCG@5": sum(ndcg_score)/len(ndcg_score),
    })

print("Precision@1:", sum(p1_score) / len(p1_score))
print("Precision@3:", sum(p3_score) / len(p3_score))
print("Precision@5:", sum(p5_score) / len(p5_score))
print("Recall@5:", sum(recall_score) / len(recall_score))
print("MRR:", sum(mrr_score) / len(mrr_score))
print("Hit Rate@5:", sum(hit_rate_score) / len(hit_rate_score))
print("NDCG@5:", sum(ndcg_score) / len(ndcg_score))


df = pd.DataFrame(rows)
df.to_csv("evaluation/temp_results.csv", index=False)

report_df = pd.DataFrame(report)
print(report_df)
report_df.to_csv("evaluation/temp.csv", index=False)
