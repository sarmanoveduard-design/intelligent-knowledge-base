# Retrieval benchmark

Это retrieval benchmark, не оценка качества LLM-ответа.

## Experiment

| Field | Value |
| --- | --- |
| schema_version | 3 |
| provider | openai |
| model_name | text-embedding-3-large |
| retrieval_mode | dense |
| candidate_k | 20 |
| candidate_k_requested | 20 |
| final_top_k | 10 |
| bm25_k1 | N/A |
| bm25_b | N/A |
| rrf_k | N/A |
| reranker | none |
| reranker_candidate_k | N/A |
| reranker_latency_seconds | N/A |
| reranker_model | N/A |
| reranker_device | N/A |
| reranker_batch_size | N/A |
| reranker_max_length | N/A |
| reranker_revision | N/A |
| reranker_revision_resolved | N/A |
| reranker_load_latency_seconds | N/A |
| reranker_requested_revision | N/A |
| reranker_resolved_revision | N/A |
| reranker_score_type | N/A |
| reranker_precision | N/A |
| hard_negative_evaluable_query_count | 119 |
| hard_negative_scope | dense_full_corpus |
| document_representation | plain |
| dimensions | 3072 |
| distance_metric | cosine |
| normalization | cosine norm division at search; stored vectors unchanged |
| corpus_hash | ac6cee9c82370082f5744d39a6497d0585ffcb8334b73bc89f8afec7de09c903 |
| gold_set_hash | 44c31197eb13d93dffa31715c888d73c6050d772c355440461d1a33289348ebe |
| top_k | 10 |
| timestamp_utc | 2026-10-06T06:43:43.183948+00:00 |
| python_version | 3.12.13 |
| code_commit_sha | 3c5a13317b3c7d8c1e8f3a636249e20952d40832 |
| code_dirty | False |
| code_commit_sha_source | environment_override |
| code_dirty_source | environment_override |
| chunk_count | 1655 |
| query_count | 119 |
| evaluable_query_count | 119 |
| successful_query_count | 119 |
| index_latency_seconds | 40.3739671000003 |
| query_latency_seconds | 80.19685379999646 |
| token_usage | {'total_tokens': 561454, 'prompt_tokens': 561454, 'api_calls': 145, 'error_count': 0, 'latency_seconds': 65.1213387000007} |
| error_count | 0 |
| cost_per_million_tokens | 0.13 |
| estimated_cost | 0.07298902 |
| status | complete |
| experiment_id | 20261006T064343Z-111e3c5dae82 |

## Metrics

| Metric | Value |
| --- | --- |
| Recall@1 | 0.6519607843137255 |
| Recall@3 | 0.8921568627450981 |
| Recall@5 | 0.9432773109243697 |
| Recall@10 | 0.957983193277311 |
| Top-1 accuracy | 0.6722689075630253 |
| MRR | 0.7924369747899159 |
| nDCG | 0.8317435524606183 |
| Candidate Recall@10 | 0.957983193277311 |
| Candidate Recall@20 | 1.0 |
| Candidate Recall@50 | N/A |
| Candidate Recall@pool | 1.0 |
| positive-over-hard-negative | 0.8991596638655462 |
| unresolved refs | 0 |
| p50 latency seconds | 0.6703581000001577 |
| p95 latency seconds | 0.7270189999999275 |
| reranker p50 latency seconds | N/A |
| reranker p95 latency seconds | N/A |

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
