import math

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_top_k = retrieved_ids[:k]
    relevant_found = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_ids)
    return relevant_found / k

def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_top_k = retrieved_ids[:k]
    relevant_found = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_ids)
    return relevant_found / len(relevant_ids) if relevant_ids else 0.0

def reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0.0

def hit_rate_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> int:
    return int(
        any(doc_id in relevant_ids for doc_id in retrieved_ids[:k])
    )

def ndcg_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_top_k = retrieved_ids[:k]

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_top_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)

    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1 / math.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg