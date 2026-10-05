# Candidate retrieval benchmark

Это retrieval benchmark, не оценка качества LLM-ответа.

## Архитектура

Один `scripts/eval_retrieval.py` работает с теми же BenchmarkInputs и snapshot-файлами
для всех режимов. `CandidateRetriever` расположен в benchmark composition layer:
candidate retrieval -> optional Reranker -> final top_k. Общий Retriever и
OpenAI/Ollama providers не изменены. Учебный `scripts/eval_bm25.py` не импортируется.
Gold используется только для расчёта метрик, а не для токенизации или ranking.

Режимы:

- `dense` (default): прежние document embeddings, query embedding и полный cosine
  поиск через Retriever. Первые candidate_k результатов образуют pool. Для final
  top_k без reranker ranking и прежние метрики остаются прежними.
- `bm25`: настоящий Okapi BM25 без embedding-провайдера и внешних сервисов.
  CLI не создаёт OpenAI/Ollama client даже при EMBEDDING_PROVIDER=openai.
- `hybrid_rrf`: первые candidate_k dense и BM25 результатов объединяются существующим
  reciprocal_rank_fusion; top candidate_k fused results передаются дальше.
  Score diagnostics используют ту же реализацию RRF, а не другой алгоритм.

BM25 использует фиксированные k1=1.5 и b=0.75. Tokenization: Unicode NFKC,
casefold и последовательности Unicode букв/чисел (`[^\W_]+`). Есть поддержка
кириллицы и Unicode, но нет stemming, stopwords, словаря синонимов или gold boosts.
Каждый уникальный query token учитывается один раз, terms сортируются до суммирования.
IDF = ln(1 + (N - df + 0.5)/(df + 0.5)); tf нормализуется длиной документа.
При одинаковом score — исходный ordinal chunk number; нулевые scores также входят
в ranking. Поэтому запрос без пересечений даёт deterministic ties, а не особые boosts.
Это не PostgreSQL FTS.

В RRF каждая ranking вносит 1/(rrf_k + rank), rank начинается с 1.
Повторы внутри ranking игнорируются. Существующий tie-break сохранён:
first search index/position, затем ordinal chunk number. Dense ranking идёт первым.
Union может иметь до 2*candidate_k элементов; fused pool ограничен candidate_k.
`candidate_union_size` и `candidate_pool_size` записываются в per_query.csv.

## Конфигурация

| Environment variable | Default | Значение |
| --- | --- | --- |
| BENCHMARK_RETRIEVAL_MODE | dense | dense / bm25 / hybrid_rrf |
| BENCHMARK_DOCUMENT_REPRESENTATION | plain | plain / structure_aware_v1 |
| BENCHMARK_TOP_K | 10 | Final top_k, целое >=10; CLI --top-k имеет приоритет |
| BENCHMARK_CANDIDATE_K | 50 | Положительное целое >=final top_k |
| BENCHMARK_RRF_K | 60 | Положительное целое; используется hybrid |
| BENCHMARK_COST_PER_MILLION_TOKENS | отсутствует | Опциональный тариф total API tokens |
| BENCHMARK_CODE_COMMIT_SHA | отсутствует | Полный hex SHA-1 (40) или SHA-256 (64 символа) |
| BENCHMARK_CODE_DIRTY | отсутствует | Только lowercase true / false |

Все конфигурации проверяются до embedding-вызовов; CLI проверяет retrieval config
и overrides до создания provider. Неизвестные значения не принимаются молча.
Ошибки не выводят заданные environment values. `.env` автоматически не читается.
Пустые overrides не означают отсутствие: их нужно удалить из окружения, если не используются.

`candidate_k` в experiment.json — min(requested candidate_k, chunk_count).
`candidate_k_requested` сохраняет исходную конфигурацию. Маленький corpus может
иметь effective pool меньше final top_k. Валидация >=top_k выполняется до cap.
У `run_benchmark` имеются явные keyword args retrieval_mode, candidate_k, rrf_k,
document_representation; окружение для них читает CLI. Старые вызовы получают defaults.

`structure_aware_v1` остаётся экспериментальным и доступен во всех режимах:
BM25 tokenizes то же выбранное document representation, что получает dense embedding.
Query не получает prefix. Default plain сохраняет прежний baseline.

## Метрики и отчёты

Final Recall@1/3/5/10, Top-1, MRR и binary nDCG рассчитываются после optional
reranker в пределах final top_k. Candidate Recall@10/20/50 рассчитывается до
reranker по первым k элементам pool; для hybrid это fused pool, а не весь union.
Candidate Recall@pool показывает покрытие всего pool, доступного будущему reranker.
Все recalls macro-average доли найденных positives для evaluable queries.

Если candidate_k меньше k и corpus ещё не исчерпан, Candidate Recall@k = N/A:
данная глубина не запрошена. Если pool охватывает весь маленький corpus, @20/@50
измеряются по всем доступным результатам. Index/retrieval failures дают нули там,
где cutoff доступен. При ошибке reranker уже вычисленные candidate metrics остаются,
а final metrics обнуляются. Unresolved refs исключаются из denominator и делают
эксперимент incomplete, как раньше. Hard negative score ties считаются неуспехом.

Positive-over-hard-negative — диагностический score показатель до reranker:
dense использует full corpus cosine, BM25 — full corpus BM25, hybrid — RRF scores
candidate union (ref вне обоих component pools имеет score=0). Это различие явно
зафиксировано в REPORT.md; final recall и candidate recall остаются общими метриками.

Metadata schema_version=2 добавляет retrieval_mode, effective/requested candidate_k,
final_top_k (legacy top_k также сохранён), bm25_k1/b, rrf_k, reranker и источники code
metadata. Неприменимые параметры и provider/model/dimensions для bm25-only = null/N/A.
Index latency записывается отдельно, query latency включает candidate retrieval,
query embedding и optional reranking; p50/p95 используют nearest rank.
Token telemetry включает index+queries; BM25 без usage не выдумывает токены или стоимость.

corpus_hash и gold_set_hash всегда означают исходные snapshot bytes. Режимы не
меняют dataset, IDs, refs, metadata или исходные файлы. Отчёты не содержат текстов
корпуса, query, answer_hint или произвольных сообщений исключений.

## Следующий milestone: reranker

`Reranker` protocol имеет стабильное `name` и метод
`rerank(query, candidates: tuple[Candidate,...]) -> tuple[Candidate,...]`.
Candidate содержит исходный chunk_id, исходный Chunk и retrieval score.
Результат должен быть полной перестановкой переданного pool: нельзя добавить
новый документ, потерять chunk или заменить исходный Chunk. Engine проверяет это,
потом обрезает ranking до final top_k. Candidate metrics вычислены до вызова.

Будущий адаптер bge-reranker-v2-m3 реализует этот интерфейс и передаётся в
`run_benchmark(..., reranker=adapter)` из composition boundary. Реальная модель,
FlagEmbedding/transformers и CLI-загрузка reranker в этом milestone не добавлялись.
Reranker не сможет вернуть positive, отсутствующий в pool.

## Docker metadata без установки git

При доступном git сохраняется autodetection HEAD/dirty. Если git/.git недоступны
в контейнере, поля остаются null/N/A; причина не маскируется. Можно передать
BENCHMARK_CODE_COMMIT_SHA и BENCHMARK_CODE_DIRTY из проверенной host-конфигурации.
Оба overrides обходят git; один override заменяет только своё поле, другое
определяется автоматически или остаётся N/A. SHA приводится к lowercase.
`code_commit_sha_source`/`code_dirty_source` = environment_override / git / unavailable.
Override — заявленная caller provenance, не самостоятельная проверка образа по commit.
В Docker image не добавляется git; Docker/volumes здесь не менялись и не запускались.

## Будущие запуски единым runner

Из корня репозитория PowerShell:

```powershell
$env:PYTHONPATH='src'
$env:BENCHMARK_RETRIEVAL_MODE='bm25'
$env:BENCHMARK_DOCUMENT_REPRESENTATION='plain'
$env:BENCHMARK_CANDIDATE_K='50'
$env:BENCHMARK_RRF_K='60'
& .\.venv\Scripts\python.exe scripts/eval_retrieval.py --corpus data/benchmark/corpus.json --gold data/benchmark/gold.json --top-k 10
```

Для dense/hybrid замените BENCHMARK_RETRIEVAL_MODE на dense/hybrid_rrf и задайте
существующие provider environment variables. Для BGE-M3: EMBEDDING_PROVIDER=ollama,
OLLAMA_EMBEDDING_MODEL=bge-m3, OLLAMA_EMBEDDING_DIMENSIONS=1024, OLLAMA_BASE_URL.
Для OpenAI: EMBEDDING_PROVIDER=openai, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIMENSIONS;
OPENAI_API_KEY предварительно задаётся защищённым механизмом окружения.
Реальные модели в процессе реализации не запускались.

## Автоматическое offline comparison

```powershell
$env:PYTHONPATH='src'
& .\.venv\Scripts\python.exe scripts/compare_retrieval_reports.py --reports-root reports --out reports `
  --corpus-hash ef8279de8dcab4c059c1c34c75b62279f5ee80bf8b8ac75718abe1fb72be1c6b `
  --gold-hash 7eb90ca9094c49df440e58ba0215525eba14d72fc682fe40a7c72d23cffe00ca
```

Создаются/заменяются reports/COMPARISON.md и reports/comparison.csv. Скрипт читает
только experiment.json и summary.csv непосредственных дочерних report directories.
Несовместимые hashes пропускаются. Без выбора hashes допустима ровно одна dataset
группа; при нескольких группах требуется явный выбор, смешивания нет.
Повреждённые отчёты пропускаются с числовым счётчиком без вывода их содержимого.
Incomplete experiments остаются помеченными, автоматический winner не выбирается.
Помимо metric table выводятся configuration, latency, cost и commit provenance.

Legacy schema поддержана: отсутствующие retrieval_mode/document_representation
интерпретируются как dense/plain, а candidate params/metrics остаются N/A.
Сравнение использует только allowlisted поля, конечные неотрицательные числа,
проверенные SHA/labels и известные modes. Raw query/corpus artifacts, answer_hint,
arbitrary metadata и exception messages в comparison не включаются.
Одинаковые hashes не означают одинаковую конфигурацию: перед выводами проверяйте
representation, top_k, candidate_k, reranker, status/errors и commit.
