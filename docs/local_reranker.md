# Локальный BGE reranker

Это retrieval benchmark, не оценка качества LLM-ответа.

## Архитектура и выбор эксперимента

Pipeline: dense BGE-M3 / 1024 / plain → candidate Top-20 →
BAAI/bge-reranker-v2-m3 → final Top-10 → существующие benchmark/reporting.
`BGEReranker` реализует существующий `Reranker.rerank(query, candidates)`;
общий Retriever и embedding providers не изменены.

По результатам уже выполненного dev benchmark (169 chunks, 18 queries),
dense BGE Candidate Recall@20 = 1.0. Поэтому первый эксперимент использует
candidate_k=20 для проверки ranking. Это не означает 100% production quality.
`structure_aware_v1` остаётся rejected experimental result; BM25/RRF остаются
benchmark alternatives, выбранный pipeline использует dense/plain.

Используется прямой Transformers `AutoModelForSequenceClassification` без
FlagEmbedding и его дополнительных зависимостей. Это вариант из
[официальной model card](https://huggingface.co/BAAI/bge-reranker-v2-m3).
Вход — пары `(query, original candidate passage)`, выход — raw relevance logits,
больший score означает более высокий rank. Gold, answer_hint и boosts не участвуют.
Равные scores сохраняют исходный порядок candidate pool. Engine проверяет полную
перестановку исходных chunks; final top_k применяется после reranking.

Модель работает в eval/inference mode, fp32, eager attention, с seed=0 и
deterministic algorithms. Воспроизводимость scores между разными устройствами,
версиями PyTorch и revisions не гарантируется; фиксируйте runtime и model revision.

## Конфигурация

| Variable | Default | Допустимые значения |
| --- | --- | --- |
| BENCHMARK_RERANKER | none | none / bge-reranker-v2-m3 |
| BGE_RERANKER_MODEL | BAAI/bge-reranker-v2-m3 | Только эта модель |
| BGE_RERANKER_DEVICE | auto | auto / cpu / cuda |
| BGE_RERANKER_BATCH_SIZE | 4 | Целое 1..1024 |
| BGE_RERANKER_MAX_LENGTH | 512 | Целое 8..8192 |
| BGE_RERANKER_REVISION | main | main или полный 40-значный hex commit SHA модели |
| HF_HOME | /cache/huggingface в Compose | Persistent cache directory |

512 — token budget всей пары query/passage с tokenizer truncation и padding,
а не число символов и не изменение corpus chunks. Длинные passages могут
обрезаться при scoring; dataset остаётся неизменным. Batch=4 — начальный
параметр, не обещание вместимости в VRAM. При OOM уменьшите batch или используйте
CPU; автоматического повторного scoring на CPU после CUDA/OOM нет.
`auto` выбирает CUDA, если PyTorch сообщает о доступности; иначе CPU.
Явный `cuda` без доступного GPU завершается безопасной ошибкой.

Default `none` не импортирует torch/transformers и сохраняет прежний benchmark.
Обычный requirements.txt и основной Dockerfile не требуют optional runtime.
`.env` не читается. Cache paths, ошибки модели, query/chunk text и answer_hint
не включаются в reports. Для публичной модели загрузка выполняется с `token=False`
и `trust_remote_code=False`. Runner отключает implicit token и telemetry.

## Один запуск в PowerShell

Из корня существующего репозитория:

```powershell
& .\scripts\run_local_reranker_benchmark.ps1
```

Требуются подготовленные `data/benchmark/corpus.json` и `gold.json`, Docker Desktop
с Compose и существующая BGE-M3 в project volume `ollama-models`.
Это default DEV inputs. Для TEAM HOLDOUT задайте пути к подготовленным snapshots
относительно корня репозитория (пути с пробелами заключайте в кавычки):

```powershell
& .\scripts\run_local_reranker_benchmark.ps1 `
    -CorpusPath 'data/team/holdout/benchmark/approved/corpus.json' `
    -GoldPath 'data/team/holdout/benchmark/approved/gold.json'
```

Для всех HOLDOUT вопросов используйте каталог `benchmark/all/`. Runner проверяет
и хеширует выбранные файлы, передаёт их в evaluation и использует их SHA256
для сравнения отчётов; копирование или перезапись datasets не требуется.

Runner использует `compose.yaml` + `compose.gpu.yaml` + `compose.reranker.yaml`;
для auto/cuda добавляется `compose.reranker.gpu.yaml`. Optional зависимости
устанавливаются только в `Dockerfile.reranker`, основной образ не меняется.
Native `.venv-reranker` больше не используется one-command runner.

Preflight проверяет Docker/Compose, project-scoped Ollama service, затем выполняет
`check_reranker_ollama.py` **в reranker container**. Он обращается только к
`http://ollama:11434/api/tags`, проверяя доступность network и наличие
`bge-m3` / `bge-m3:latest`. Embeddings, generation и pull не выполняются preflight.
Если service остановлен, runner выполняет только `up -d --no-recreate ollama`.
Если BGE-M3 отсутствует, выдаётся ошибка; runner её не скачивает.
Работающий Ollama не перезапускается. `ollama-models` не удаляется.

Ollama и reranker находятся в одном Compose project/default network.
Порт 11434 не публикуется на Windows host. Ни localhost, ни host.docker.internal
не используются для доступа к Ollama. Имя существующего нестандартного project
определяется по Compose labels с working_dir этого репозитория; явный process
`COMPOSE_PROJECT_NAME` имеет приоритет. При нескольких project для той же папки
runner требует явно выбрать имя и не угадывает. Compose получает пустой временный
`--env-file`, поэтому `.env` и секреты не читаются. Временный файл удаляется.

Согласованная конфигурация не требует ручной установки environment variables:

```text
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_EMBEDDING_DIMENSIONS=1024
BENCHMARK_RETRIEVAL_MODE=dense
BENCHMARK_DOCUMENT_REPRESENTATION=plain
BENCHMARK_CANDIDATE_K=20
BENCHMARK_TOP_K=10
BENCHMARK_RERANKER=bge-reranker-v2-m3
BGE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
BGE_RERANKER_DEVICE=auto
BGE_RERANKER_BATCH_SIZE=4
BGE_RERANKER_MAX_LENGTH=512
BGE_RERANKER_REVISION=main
HF_HOME=/cache/huggingface
```

Runner получает git SHA/dirty, выполняет benchmark, затем comparison с hashes
исходных corpus/gold snapshots. Выводит только понятный итог и пути REPORT/COMPARISON.
Incomplete experiment сохраняется как incomplete и runner возвращает ошибку.
Native/model exceptions и Docker/build output не выводятся.

Параметры CPU или фиксированной model revision:

```powershell
& .\scripts\run_local_reranker_benchmark.ps1 -Device cpu -BatchSize 2 -MaxLength 512
& .\scripts\run_local_reranker_benchmark.ps1 -Revision '<reranker_resolved_revision из experiment.json>'
```

CPU исключает GPU overlay reranker. Auto/cuda запрашивают NVIDIA GPU, как существующий
GPU runtime проекта; `auto` внутри runtime выбирает CPU, если CUDA недоступна PyTorch.
При невозможности выделить GPU контейнер не стартует: используйте `-Device cpu`.
Ollama сохраняет GPU-конфигурацию существующего compose.gpu.yaml.
Наличие GPU, driver compatibility и вместимость VRAM будут проверены при реальном
запуске; при разработке Docker и настоящая модель не запускались.

## Persistent cache и revision

Named volume `<project>_reranker-hf-cache` монтируется в `/cache/huggingface`.
`HF_HOME` и явный `cache_dir` tokenizer/model указывают туда. `run --rm` удаляет
временный container, сохраняя volume. Rebuild runtime также не удаляет cache.
Одинаковая revision использует уже загруженные artifacts; возможна проверка
небольших HF metadata, повторная загрузка GB weights не требуется при целостном cache.
Если `main` изменится upstream, новый snapshot может требовать download.

experiment.json / REPORT.md / COMPARISON записывают:
`reranker_requested_revision` — main или запрошенный SHA;
`reranker_resolved_revision` — фактический commit из `model.config._commit_hash`.
Tokenizer использует этот же resolved commit, если он доступен.
Если metadata не содержит корректный SHA, JSON хранит null, Markdown показывает N/A;
запрошенный SHA не подставляется вместо фактически разрешённого.
Legacy поля `reranker_revision`/`reranker_revision_resolved` сохранены.
После успешного первого запуска передайте resolved SHA через `-Revision` для
следующего воспроизводимого benchmark. Для заполненного cache можно задать
`HF_HUB_OFFLINE=1`; его значение Compose передаёт runtime. Первое скачивание
reranker требует сеть. Cache и реальные reports не включаются в Git.

## Метрики и отчёты

Candidate Recall@10/20/50/pool вычисляется до reranking. Final Recall@1/3/5/10,
Top-1, MRR и nDCG вычисляются по reranked final Top-10.
Positive-over-hard-negative использует reranker scores полного candidate pool,
до final truncation. Positive отсутствует в pool → 0. Если positive есть,
но хотя бы один требуемый hard negative отсутствует в pool → N/A, а не победа.
При равных scores результат 0. `hard_negative_scope` и число оцениваемых запросов
показывают область и denominator: reranker_candidate_pool отличается от full-corpus
диагностики baseline. Сравнивайте этот показатель с учётом scope/count.

experiment.json / REPORT.md содержат reranker/model/device/candidate_k/batch/max_length,
revision и resolved revision, total reranker latency и отдельно model load latency.
summary.csv содержит reranker p50/p95; per_query.csv — reranker latency.
Query latency включает retrieval и reranking, model load измеряется до queries.
Comparison сохраняет конфигурацию, total latency и p50/p95.
Числа реального reranked benchmark появятся только после явного запуска;
синтетические unit tests не являются оценкой модели.

## Offline verification

Тесты используют fake backend/client и не требуют optional пакетов, GPU или модели.
Проверка всего suite с заблокированными socket connections:

```powershell
$env:PYTHONPATH='src'
& .\.venv\Scripts\python.exe -c "import socket,unittest; from unittest.mock import patch; suite=unittest.defaultTestLoader.discover('tests'); blocked=patch.object(socket.socket,'connect',side_effect=AssertionError('Network disabled')); blocked.start(); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not result.wasSuccessful())"
```
