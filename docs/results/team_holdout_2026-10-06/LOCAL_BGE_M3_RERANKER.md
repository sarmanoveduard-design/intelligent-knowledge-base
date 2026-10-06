# Retrieval benchmark

Это retrieval benchmark, не оценка качества LLM-ответа.

## Experiment

| Field | Value |
| --- | --- |
| schema_version | 3 |
| provider | ollama |
| model_name | bge-m3 |
| retrieval_mode | dense |
| candidate_k | 20 |
| candidate_k_requested | 20 |
| final_top_k | 10 |
| bm25_k1 | N/A |
| bm25_b | N/A |
| rrf_k | N/A |
| reranker | bge-reranker-v2-m3 |
| reranker_candidate_k | 20 |
| reranker_latency_seconds | 117.2035065219984 |
| reranker_model | BAAI/bge-reranker-v2-m3 |
| reranker_device | cuda |
| reranker_batch_size | 4 |
| reranker_max_length | 512 |
| reranker_revision | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |
| reranker_revision_resolved | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |
| reranker_load_latency_seconds | 82.58920671599981 |
| reranker_requested_revision | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |
| reranker_resolved_revision | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |
| reranker_score_type | raw relevance logit |
| reranker_precision | float32 |
| hard_negative_evaluable_query_count | 88 |
| hard_negative_scope | reranker_candidate_pool |
| document_representation | plain |
| dimensions | 1024 |
| distance_metric | cosine |
| normalization | cosine norm division at search; stored vectors unchanged |
| corpus_hash | ac6cee9c82370082f5744d39a6497d0585ffcb8334b73bc89f8afec7de09c903 |
| gold_set_hash | 44c31197eb13d93dffa31715c888d73c6050d772c355440461d1a33289348ebe |
| top_k | 10 |
| timestamp_utc | 2026-10-06T06:29:20.952342+00:00 |
| python_version | 3.12.15 |
| code_commit_sha | 3c5a13317b3c7d8c1e8f3a636249e20952d40832 |
| code_dirty | False |
| code_commit_sha_source | environment_override |
| code_dirty_source | environment_override |
| chunk_count | 1655 |
| query_count | 119 |
| evaluable_query_count | 119 |
| successful_query_count | 119 |
| index_latency_seconds | 82.46188683899982 |
| query_latency_seconds | 142.74993420000237 |
| token_usage | {'total_tokens': None, 'prompt_tokens': None, 'api_calls': None, 'error_count': None, 'latency_seconds': None} |
| error_count | 0 |
| cost_per_million_tokens | N/A |
| estimated_cost | N/A |
| status | complete |
| experiment_id | 20261006T062921Z-ab00f78ed058 |

## Metrics

| Metric | Value |
| --- | --- |
| Recall@1 | 0.8921568627450981 |
| Recall@3 | 0.9523809523809523 |
| Recall@5 | 0.9845938375350141 |
| Recall@10 | 0.988795518207283 |
| Top-1 accuracy | 0.907563025210084 |
| MRR | 0.9432773109243697 |
| nDCG | 0.9525893994022965 |
| Candidate Recall@10 | 0.9782913165266107 |
| Candidate Recall@20 | 0.9915966386554622 |
| Candidate Recall@50 | N/A |
| Candidate Recall@pool | 0.9915966386554622 |
| positive-over-hard-negative | 0.9545454545454546 |
| unresolved refs | 0 |
| p50 latency seconds | 1.1574718309998389 |
| p95 latency seconds | 1.4725593739999567 |
| reranker p50 latency seconds | 0.9629896320002445 |
| reranker p95 latency seconds | 1.2466195100000732 |

MRR and binary nDCG use top_k. Recall is macro-averaged across queries.
Unresolved-reference queries are excluded; failed retrievals count as zero.
Positive-over-hard-negative: best positive score strictly exceeds best hard-negative score; ties fail.
Candidate Recall is measured before reranking; hybrid pool is the top candidate_k RRF union.
Candidate Recall@k is N/A when k exceeds the requested pool and unsearched corpus remains.
Without reranker: dense/BM25 hard-negative diagnostics use full-corpus scores; hybrid uses RRF scores (absent=0).
With reranker: hard-negative scores are post-rerank within the full candidate pool, before final top_k truncation.
Absent all positives count as zero; absent hard-negative scores otherwise mean N/A, not success.
Latency includes candidate retrieval and optional reranking; percentiles use nearest rank.
Token usage covers indexing and queries. N/A means unavailable, not zero.
Corpus and gold snapshots are identified by SHA-256 of exact input bytes.
