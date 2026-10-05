# Retrieval benchmark: OpenAI и локальные embeddings

Это retrieval benchmark, не оценка качества LLM-ответа.
Generation API и GPT-5.6 Luna не используются. Реальные результаты OpenAI пока не получены.

## Архитектура и совместимость

`OpenAIEmbeddingProvider` реализует существующий `EmbeddingProvider`: `dimension`,
`embed_texts`, `embed_query`. `Retriever` и cosine `InMemoryVectorStore` общие для
всех моделей. `embedding_config.PROVIDER_FACTORIES` — точка регистрации новых
адаптеров для Qwen3-Embedding, EmbeddingGemma, multilingual-e5, Yandex и
GigaEmbeddings. Эти адаптеры пока не реализованы; benchmark от SDK не зависит.

OpenAI: `text-embedding-3-small` — 1536 dimensions по умолчанию;
`text-embedding-3-large` — 3072. Поддерживается сокращение, включая 1024.
Если `OPENAI_EMBEDDING_DIMENSIONS` отсутствует или пустая, используется native
dimension выбранной модели. Значение 1536 из `.env.example` нужно изменить или
убрать при native-запуске large. `.env` автоматически не читается.

Источник ключа — только `OPENAI_API_KEY` в окружении процесса. Нет аргумента CLI
для ключа. SDK использует официальный endpoint; retries отключены для прозрачного
учёта вызовов и ошибок. Timeout задаётся в секундах. `batch_size` ограничивает
число текстов в одном синхронном embedding-запросе, это не OpenAI Batch API.
Тексты не обрезаются: слишком длинные входы приводят к безопасной ошибке API.
Количество токенов всех текстов батча также должно укладываться в лимиты API;
при необходимости уменьшите размер батча. Никаких скрытых изменений корпуса.

Хранилище привязывается к provider/model/dimensions при создании Retriever;
каждый index/search повторно проверяет идентификатор. Модель нельзя заменить,
даже когда dimensions совпадают. Заполненное вручную хранилище без идентичности
не подключается к Retriever. Низкоуровневые `add`/`add_many` принимают голые
векторы и не могут установить их происхождение: для индексации используйте
`Retriever.index_chunks`. Для legacy-провайдеров без `identity` используется
тип класса и атрибут `model`; адаптеры с несколькими конфигурациями должны
объявлять явный `EmbeddingIdentity`.

## Локальные входные файлы

Подготовьте два JSON-файла UTF-8. `data/benchmark/` исключён из Git.
Не нужно читать или копировать закрытые документы для unit-тестов.

`data/benchmark/corpus.json` — один зафиксированный snapshot готовых chunks:

```json
{"chunks": [
  {"id": "chunk-001", "text": "Разрешённый текст первого фрагмента."},
  {"id": "chunk-002", "text": "Разрешённый текст второго фрагмента."}
]}
```

`data/benchmark/gold.json` — вопросы и ссылки именно на эти chunk IDs:

```json
{"queries": [
  {"text": "Тестовый вопрос?", "positive_chunk_ids": ["chunk-001"],
   "hard_negative_chunk_ids": ["chunk-002"]}
]}
```

ID чанков уникальны; positives непустые, без дублей и пересечения с hard negatives.
Hard negatives необязательны. Если их нет, показатель отображается как N/A.
Snapshot/version идентифицируется SHA-256 точных байтов каждого файла, поэтому
изменение даже форматирования создаёт новый hash. Сравнивайте одинаковые файлы,
chunking, gold, top_k, distance metric и normalization для всех моделей.
Старый `scripts/benchmark_data.py` содержит только маленький синтетический пример,
не утверждённый gold-набор заказчика.

## Будущие реальные запуски без Docker

Команды выполняются из корня существующего репозитория. SDK установлен в `.venv`.
Перед запуском задайте `OPENAI_API_KEY` через защищённый механизм окружения.
Ниже ключ не устанавливается и не печатается. Эти команды отправляют корпус и
вопросы embedding API; используйте согласованный для такого запуска корпус.

Small, native 1536:

```powershell
$env:PYTHONPATH='src'
$env:EMBEDDING_PROVIDER='openai'
$env:OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
$env:OPENAI_EMBEDDING_DIMENSIONS='1536'
& .\.venv\Scripts\python.exe scripts/eval_retrieval.py --corpus data/benchmark/corpus.json --gold data/benchmark/gold.json --top-k 10
```

Large, native 3072:

```powershell
$env:PYTHONPATH='src'
$env:EMBEDDING_PROVIDER='openai'
$env:OPENAI_EMBEDDING_MODEL='text-embedding-3-large'
$env:OPENAI_EMBEDDING_DIMENSIONS='3072'
& .\.venv\Scripts\python.exe scripts/eval_retrieval.py --corpus data/benchmark/corpus.json --gold data/benchmark/gold.json --top-k 10
```

Для контролируемого сравнения обеих моделей с BGE-M3 замените dimensions на `1024`.
Для BGE-M3 используйте тот же runner и входные файлы:

```powershell
$env:PYTHONPATH='src'
$env:EMBEDDING_PROVIDER='ollama'
$env:OLLAMA_EMBEDDING_MODEL='bge-m3'
$env:OLLAMA_EMBEDDING_DIMENSIONS='1024'
$env:OLLAMA_BASE_URL='http://localhost:11434'
& .\.venv\Scripts\python.exe scripts/eval_retrieval.py --corpus data/benchmark/corpus.json --gold data/benchmark/gold.json --top-k 10
```

Последняя команда требует уже доступную Ollama с BGE-M3. Она не запускает Docker.
Цена опциональна: `BENCHMARK_COST_PER_MILLION_TOKENS` задаёт стоимость за миллион
total tokens в выбранной валюте. Цена автоматически не берётся из сети;
estimated_cost отражает переданный тариф, не фактический счёт провайдера.

## Артефакты и определения

Каждый реальный запуск создаёт `reports/<UTC timestamp>-<random id>/`:
`REPORT.md`, `experiment.json`, `summary.csv`, `per_query.csv`, `errors.csv`.
`reports/` исключён из Git. Отчёты unit-тестов создаются только во временных папках.
Нет API key, username, hostname, home path, путей исходников, исходных ID,
текстов запросов, документов, vectors или сообщений SDK. В per_query сохраняются
только ordinal query number, ordinal chunk numbers (нумерация чанков с нуля),
метрики, status и latency. Для расшифровки нужны локальные snapshot-файлы.

Metadata включает provider/model/dimensions, cosine, политику normalization
(деление на нормы в cosine, без изменения хранимых векторов), hashes corpus/gold,
top_k, UTC timestamp, Python version, commit SHA, dirty flag, количество chunks,
queries, успешных и оцениваемых запросов, indexing/query latency, API telemetry,
error count и опциональный тариф/стоимость. Dirty flag важен: SHA не включает
незакоммиченные изменения. Код этого этапа не коммитится.

Recall@k — macro average доли найденных positives; Top-1 — первый результат
релевантен; MRR и binary nDCG считаются в пределах top_k (минимум 10).
Positive-over-hard-negative — лучший positive строго выше лучшего hard negative;
равные scores считаются неуспехом. Для этого сравнения поиск ранжирует весь
in-memory corpus. p50/p95 — nearest-rank latency embedding query + полный поиск,
включая неудачные попытки. Индексация измеряется отдельно, usage включает оба
этапа. Провайдер без usage получает N/A, а не выдуманное число токенов.

Запросы с unresolved refs исключаются из denominator; ошибки index/query
учитываются как нулевые метрики для разрешимых запросов. Любая ошибка делает
status `incomplete`, CLI возвращает exit code 1. При ошибке конфигурации/формата
до эксперимента отчёт не создаётся. Не сравнивайте incomplete run как успешный.

Проверка без сети и настоящего API:

```powershell
$env:PYTHONPATH='src'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Официальная документация:
[Embeddings guide](https://developers.openai.com/api/docs/guides/embeddings),
[Create embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create).
