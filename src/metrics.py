def hit_at_k(retrieved_docs, ground_truth_doc_id, k):
    top_k_docs = retrieved_docs[:k]
    for doc in top_k_docs:
        if doc.metadata.get("id") == ground_truth_doc_id:
            return 1
    return 0

def mean_reciprocal_rank(retrieved_docs, ground_truth_doc_id):
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc.metadata.get("id") == ground_truth_doc_id:
            return 1 / rank
    return 0

def evaluate_retrieval(retriever, queries, ground_truths, top_k_list=[1,3,5]):
    hits = {f"Hit@{k}": [] for k in top_k_list}
    mrrs = []

    for query, gt_id in zip(queries, ground_truths):
        retrieved_docs = retriever.get_relevant_documents(query)
        for k in top_k_list:
            hits[f"Hit@{k}"].append(hit_at_k(retrieved_docs, gt_id, k))
        mrrs.append(mean_reciprocal_rank(retrieved_docs, gt_id))

    metrics = {k: sum(v)/len(v) for k,v in hits.items()}
    metrics["MRR"] = sum(mrrs)/len(mrrs)
    return metrics
