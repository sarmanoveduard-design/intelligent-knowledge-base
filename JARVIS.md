# JARVIS — Интеллектуальная база знаний

> Public, sanitized project-state record. Never include customer-confidential documents, personal/medical records, private source corpora, secrets, internal deployment details or sensitive prompts in this public repository.

## Identity
- Project: Интеллектуальная база знаний (intelligent-knowledge-base)
- Repository: sarmanoveduard-design/intelligent-knowledge-base
- Phase: R&D / BUILD / RETRIEVAL VALIDATION
- Deployment: local research prototype; no verified production service
- Project scope: reusable document-ingestion and evidence-based retrieval prototype; may be used as a team RAG experimentation stand for УКЗ-15 «Эксперт МОСТ».
- Last repository state inspected: 2026-09-26, main at 9c843d807dcf298463d0cf1efa212071273bc0eb.

## Goal
Deliver a dependable knowledge assistant that retrieves only permitted, approved source versions; provides verifiable citations with document/version/location; refuses unsupported answers and can escalate unresolved questions to an expert. Do not mistake a retriever benchmark for a verified full assistant.

## Verified in current GitHub main
- Document registration, SHA-256 exact duplicates and document lifecycle statuses.
- DOCX and text-layer PDF extraction, normalization, source blocks and chunking.
- Embedding abstraction, in-memory vector store and retriever.
- Local BGE-M3 embeddings via Ollama (1024 dimensions).
- Shared BGE-M3/BM25 benchmark scripts; latest main commit adds BM25 baseline.
- README documents 68 Python unit tests and an initial synthetic BGE-M3 experiment. This is documentation of past local runs, not a newly executed test.
- The README's "not implemented" list predates the latest benchmark/BM25 commit and must not be treated as fully current.

## Critical local-work protection
Current GitHub main is NOT proof of the state of the owner's local workstation.
A recent local `scripts/eval_hybrid.py` experiment was discussed in the project chat; this script is not in the inspected main/scripts directory. Before any git pull, reset, clean, checkout, rebase, overwrite or commit, inspect and protect local modified/untracked files and reconcile with GitHub.
Do not claim the hybrid retriever is merged or that its evaluation passes without inspecting and rerunning the current local implementation.

## Intended user experience / FRONTIER-FIRST
A real permitted-document question -> source/version/permission filtering -> current hybrid retrieval and reranking where justified -> evidence-grounded answer with precise citations -> abstention if evidence is insufficient -> expert escalation if configured.
The "WOW" value here is speed, trust, traceability and accurate workflow support, not superficial animation or unverifiable model confidence.
Check contemporary retrieval/evaluation techniques and official model/tool capabilities when making material decisions; preserve modularity so embeddings, vector stores, rerankers and LLM providers can be replaced.

## Architecture boundaries
- Registry and version statuses do not by themselves enforce retrieval eligibility.
- Approval state and authorization filters must be applied server-side before results can support answers.
- Customer/private documents and personal/medical data must not enter this public repo or public test fixtures.
- Synthetic/public approved evaluation sets only in GitHub.
- No generated answer should cite a superseded draft as if it were active.
- Retrieval ranking (Hit@k) is not answer correctness; separately evaluate grounded answers, citation correctness and refusals.
- Do not deploy as a medical information system without appropriate authorization, governance, validation and privacy controls.

## Current work
- Reconcile current local hybrid-search experiment with GitHub main.
- Investigate locally observed evaluation failure where the relevant source for an unpublished draft question was not ranked first.
- Validate BM25, BGE-M3 and hybrid on the same fixed permitted test set.
- Add server-side approved-version/role filtering before an answer pipeline.
- Design full evidence/citation/abstention tests using only authorized corpus materials.

## Chess map
- 20-step horizon: stabilize benchmark, protect local WIP, validate retrieval, source integrity and permissions.
- 50-step horizon: version-aware indexing, hybrid/rerank evaluation, grounded generation, citation checking, expert handoff, UI and observability.
- 100-step horizon: modular source connectors, multi-organization isolation if justified, continuous regression evaluations, controlled updates and provider replaceability.

## Working agreement
Follow JARVIS CONTROL CENTER's current natural-language, ШАХМАТЫ, FRONTIER-FIRST/WOW and approval rules. User need not type commands such as PROJECT INIT.
Before material changes, inspect current local status and the latest project-chat state; do not infer unseen local work from an older public README.
No commit/push/deploy or destructive local operation without explicit scoped owner approval. Record only verified FEATURE COMPLETE; PROJECT FINAL requires a full audit.

## Stable checkpoint
- Inspected GitHub main: 9c843d807dcf298463d0cf1efa212071273bc0eb (shared benchmark and BM25 baseline).
- Local working-tree status: NOT VERIFIED in this onboarding step.
- Production state: NOT DEPLOYED / NOT VERIFIED.
- Next move: recover local hybrid evaluation state safely before choosing the next implementation block.
