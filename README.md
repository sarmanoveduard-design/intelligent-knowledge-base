# Интеллектуальная база знаний

**Статус:** исследовательский прототип RAG-ядра: подготовка документов, embeddings, retrieval, reranking и воспроизводимые benchmark-эксперименты. Retrieval holdout и generation benchmark для основного набора из 119 approved-вопросов уже проведены. Это не готовый production RAG-ассистент и не медицинская информационная система.

Проект универсальный: он не привязан к одному заказчику или предметной области. В рамках стажировки УКЗ-15 «Эксперт МОСТ» его можно использовать как отдельный стенд для экспериментов группы RAG.

В публичном репозитории находятся код, синтетические примеры, намеренно опубликованный frozen generation benchmark для MAIN119 и выбранные итоговые отчёты. Исходные приватные документы, полный внутренний corpus заказчика, секреты и иные закрытые материалы публиковать нельзя.

## Что уже работает

- Регистрация документов, SHA-256, выявление точных дубликатов и базовое версионирование (draft / active / superseded / archived).
- Извлечение и нормализация текста DOCX, в том числе текста гиперссылок, и PDF с текстовым слоем.
- Универсальные SourceBlock и Chunk с привязкой к документу, версии и доступным позициям в источнике (страница PDF / абзац DOCX).
- Интерфейс EmbeddingProvider, тестовое хранилище векторов в памяти и Retriever для поиска подходящих фрагментов.
- Локальная мультиязычная embedding-модель BGE-M3 через Ollama: на входе текст, на выходе вектор из 1024 чисел.
- OpenAI embedding provider для `text-embedding-3-small` и `text-embedding-3-large` с настраиваемой размерностью.
- Экспериментальные BM25 и RRF для сравнительных retrieval-экспериментов.
- Локальный reranking через `BAAI/bge-reranker-v2-m3`: BGE-M3 dense → Top-20 candidates → reranker → Top-10.
- Воспроизводимый retrieval benchmark с Recall@k, Top-1 accuracy, MRR, nDCG, hard-negative diagnostics, latency, token usage, cost estimate и SHA-256 snapshots входных corpus/gold.
- Импорт и строгая валидация TEAM HOLDOUT, включая проверку ссылок на chunks, hard negatives, статусов approved/provisional и provenance.
- Independent TEAM HOLDOUT: 9 документов, 1655 chunks, 149 вопросов всего, основной approved-only subset — 119 вопросов.
- Frozen generation benchmark MAIN119: один и тот же Top-10 контекст и один prompt для Qwen3:8B, GPT-5.6 Luna и GPT-5.6 Terra.
- Детерминированная проверка source validity и citation-format для всех 119 ответов каждой модели.
- Дополнительная blind-проверка готовых ответов моделью GPT-5.6 Sol в роли LLM-judge: 119 сравнений, ответы A/B/C обезличены и переставляются между вопросами.
- Colab notebook для generation benchmark.
- На текущей ветке полный Docker unit/integration suite: **283 теста, 8 skipped**.

Полностью автоматизированный production-путь «новый файл → обновление индекса → retrieval → проверяемый LLM-ответ → безопасный отказ/эскалация эксперту» ещё не собран. Retrieval и generation сейчас проверяются как отдельные воспроизводимые экспериментальные этапы.

## Что пока не реализовано

- Извлечение таблиц DOCX и OCR для сканированных PDF.
- Постоянная production-векторная БД.
- Интеграция экспериментальных BM25/RRF и benchmark-reranker в основной production Retriever.
- Серверная фильтрация поиска по утверждённым версиям, организациям и правам доступа.
- Production-генерация ответа с обязательной проверкой цитат, отказом при недостатке источников и очередью эксперта.
- Отдельный negative/refusal benchmark: основной MAIN119 содержит answerable-вопросы и не измеряет корректность отказа при недостатке данных.
- EXTENDED-набор из всех 149 вопросов ещё не зафиксирован как отдельный generation snapshot: 30 provisional-вопросов требуют отдельного диагностического режима.

**Важно:** наличие статусов в реестре документов ещё не означает, что поиск автоматически исключает черновики. Пока не реализован соответствующий production-фильтр, нельзя использовать прототип для ответов по реальным нормативным материалам без дополнительной проверки.

## Структура

| Путь | Назначение |
| --- | --- |
| `src/knowledge_base/` | Регистрация, разбор документов, chunks, embeddings, retrieval и benchmark-логика |
| `src/knowledge_base/ollama_embeddings.py` | Адаптер к Ollama API для BGE-M3 |
| `src/knowledge_base/openai_embeddings.py` | OpenAI embeddings adapter |
| `src/knowledge_base/bge_reranker.py` | Адаптер локального `bge-reranker-v2-m3` |
| `src/knowledge_base/fusion.py` | RRF: объединение ранжированных результатов |
| `src/knowledge_base/team_holdout_benchmark.py` | Подготовка и валидация TEAM HOLDOUT |
| `scripts/eval_retrieval.py` | Универсальный retrieval benchmark runner |
| `scripts/compare_retrieval_reports.py` | Offline-сравнение совместимых retrieval reports |
| `scripts/run_local_reranker_benchmark.ps1` | One-command запуск локального BGE-M3 + reranker benchmark |
| `scripts/prepare_team_holdout_benchmark.py` | Подготовка frozen corpus/gold snapshots для TEAM HOLDOUT |
| `benchmarks/generation/main_119/` | Публичный frozen input MAIN119 и generation prompt |
| `notebooks/generation_benchmark_colab.ipynb` | Colab notebook для generation benchmark |
| `docs/retrieval_benchmark.md` | Методика retrieval benchmark |
| `docs/local_reranker.md` | Локальный reranker и воспроизводимый запуск |
| `docs/team_holdout.md` | Формат и подготовка TEAM HOLDOUT |
| `docs/results/team_holdout_2026-10-06/` | Опубликованные результаты retrieval holdout |
| `docs/results/generation_holdout_2026-10-06/` | Результаты generation benchmark, Sol judge и critical audit |
| `tests/` | Автоматические unit/integration tests |
| `compose.yaml` | Основной Python-сервис |
| `compose.cpu.yaml` | Ollama без GPU |
| `compose.gpu.yaml` | Ollama и Python-сервис с доступом к NVIDIA GPU |

## Быстрый старт: код и тесты без Ollama

Понадобятся **Git и Docker Desktop** (или Docker Engine с Compose). Команды ниже подходят для PowerShell; на macOS/Linux они практически такие же.

Склонировать репозиторий (если ещё не клонирован):

~~~powershell
git clone https://github.com/sarmanoveduard-design/intelligent-knowledge-base.git
cd intelligent-knowledge-base
~~~

Собрать Python-контейнер и запустить тесты:

~~~powershell
docker compose build app
docker compose run --rm app python -m unittest discover -s tests -v
~~~

На текущей ветке полный suite: **283 теста, 8 skipped**. Для основной offline-проверки не требуются OpenAI API, BGE-M3, reranker model или GPU; runtime/integration checks, которым нужен внешний model runtime, могут быть пропущены.

## Эксперимент с BGE-M3: вариант A — без NVIDIA (CPU)

Подходит для компьютера без подходящей видеокарты. На CPU обработка может занимать заметно больше времени.

После клонирования и сборки Python-контейнера:

~~~powershell
docker compose -f compose.yaml -f compose.cpu.yaml up -d ollama
docker compose -f compose.yaml -f compose.cpu.yaml exec -T ollama ollama pull bge-m3
docker compose -f compose.yaml -f compose.cpu.yaml run --rm app python scripts/smoke_bge_m3.py
docker compose -f compose.yaml -f compose.cpu.yaml run --rm app python scripts/eval_bge_m3.py
docker compose -f compose.yaml -f compose.cpu.yaml run --rm app python scripts/eval_hybrid.py
~~~

BM25 можно проверить **отдельно, без Ollama и скачивания модели**, после сборки контейнера:

~~~powershell
docker compose -f compose.yaml -f compose.cpu.yaml run --rm --no-deps app python scripts/eval_bm25.py
~~~

Первая команда скачает и запустит Docker-образ Ollama; вторая один раз скачает BGE-M3 в отдельный постоянный Docker-том. Устанавливать Ollama как программу на Windows не нужно.

## Эксперимент с BGE-M3: вариант B — NVIDIA GPU

Нужен Docker с поддержкой NVIDIA GPU. Этот вариант проверен на Windows с RTX 2060 SUPER (8 ГБ VRAM). На другом оборудовании сначала проверьте доступ Docker к видеокарте.

После клонирования и сборки Python-контейнера:

~~~powershell
docker compose -f compose.yaml -f compose.gpu.yaml up -d ollama
docker compose -f compose.yaml -f compose.gpu.yaml exec -T ollama ollama pull bge-m3
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app python scripts/smoke_bge_m3.py
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app python scripts/eval_bge_m3.py
docker compose -f compose.yaml -f compose.gpu.yaml run --rm app python scripts/eval_hybrid.py
~~~

Не используйте `compose.gpu.yaml` на компьютере без настроенной поддержки NVIDIA: и сервис `app`, и сервис `ollama` в этой конфигурации запрашивают GPU.

## Эксперименты и результаты

### Ранний synthetic smoke/dev benchmark

Старые скрипты используют набор из `scripts/benchmark_data.py`: 6 коротких вымышленных фрагментов и 12 вопросов.

| Метод | Hit@1 | Hit@3 |
| --- | ---: | ---: |
| BGE-M3 dense | 11/12 | 12/12 |
| BM25 | 10/12 | 10/12 |
| BGE-M3 + BM25 (RRF, k=60) | 11/12 | 12/12 |

Этот набор оставлен как быстрый smoke/dev test. Он **не используется для финальных выводов о качестве retrieval**.

### DEV benchmark по 152-ФЗ

Отдельный DEV benchmark: 169 embeddable chunks и 18 gold-вопросов. Он использовался для предварительного сравнения конфигураций и выбора frozen pipeline перед независимой проверкой.

DEV не считается независимым holdout и не должен использоваться для финального выбора после просмотра holdout-результатов.

### Independent TEAM HOLDOUT — retrieval, 2026-10-06

Для независимой проверки подготовлен TEAM HOLDOUT:

- 9 документов;
- 1655 chunks;
- 149 вопросов всего;
- основной `approved-only` subset: **119 вопросов**;
- unresolved refs: **0**;
- одинаковые frozen corpus/gold snapshots для local и cloud прогонов.

Полное сравнение: [`docs/results/team_holdout_2026-10-06/COMPARISON.md`](docs/results/team_holdout_2026-10-06/COMPARISON.md).

#### LOCAL

Pipeline: `BGE-M3 / 1024 → dense Top-20 → bge-reranker-v2-m3 → Top-10`.

| Метрика | Значение |
| --- | ---: |
| Recall@1 | 89.22% |
| Recall@3 | 95.24% |
| Recall@5 | 98.46% |
| Recall@10 | 98.88% |
| Top-1 accuracy | 90.76% |
| MRR | 0.9433 |
| nDCG | 0.9526 |
| Candidate Recall@20 | 99.16% |
| Positive > Hard Negative | 95.45% |
| p50 latency | 1.16 s |
| p95 latency | 1.47 s |
| errors | 0 |

#### CLOUD

Pipeline: `OpenAI text-embedding-3-large / 3072 → dense Top-20 candidates → final Top-10`, без reranker.

| Метрика | Значение |
| --- | ---: |
| Recall@1 | 65.20% |
| Recall@3 | 89.22% |
| Recall@5 | 94.33% |
| Recall@10 | 95.80% |
| Top-1 accuracy | 67.23% |
| MRR | 0.7924 |
| nDCG | 0.8317 |
| Candidate Recall@20 | 100% |
| Positive > Hard Negative | 89.92% |
| p50 latency | 0.67 s |
| p95 latency | 0.73 s |
| errors | 0 |
| estimated embedding cost | ~$0.073 |

**Корректный вывод:** на этом frozen holdout локальный pipeline с отдельным reranker показал более сильное итоговое ranking-качество, а OpenAI `text-embedding-3-large` был быстрее и дал Candidate Recall@20 = 100%. Это **не прямое сравнение BGE-M3 embedding против OpenAI embedding**, потому что LOCAL включает отдельный reranker.

Holdout используется как независимая проверка. Не следует продолжать тюнинг retrieval под эти 119 вопросов и затем считать тот же набор независимым тестом.

Исходные приватные документы, полный corpus и raw retrieval reports в Git не публикуются. Для последующего generation benchmark отдельно и намеренно опубликован frozen MAIN119 snapshot с вопросами и уже отобранным Top-10 контекстом: [`benchmarks/generation/main_119/`](benchmarks/generation/main_119/).

### Generation benchmark MAIN119 — 2026-10-06

На одном и том же frozen Top-10 и одном prompt были сравнены:

- `Qwen3:8B` — локально;
- `GPT-5.6 Luna` — cloud;
- `GPT-5.6 Terra` — cloud.

Каждая модель дала ответы на все **119 approved-вопросов**, без generation errors.

После этого готовые ответы были отдельно проверены `GPT-5.6 Sol` в роли blind LLM-judge. Sol не являлся четвёртым участником benchmark: ему передавались вопрос, frozen Top-10, reference answer_hint и обезличенные ответы A/B/C. Названия моделей были скрыты, а порядок A/B/C менялся между вопросами.

Полный пакет результатов: [`docs/results/generation_holdout_2026-10-06/`](docs/results/generation_holdout_2026-10-06/).

#### Sol judge

| Модель | Итог / 22 | Correctness / 4 | Faithfulness / 4 | Побед | Raw critical flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Terra | **21.7815** | **3.9244** | **3.9748** | **34** | 2 |
| GPT-5.6 Luna | 21.6134 | 3.8824 | 3.9160 | 15 | 2 |
| Qwen3:8B | 19.9328 | 3.5126 | 3.6555 | 14 | 13 |

Ничьих: **56 / 119**.

Judge usage: 450498 input tokens, 47945 output tokens; оценочная стоимость Standard-прогона — **$2.7609**.

#### Детерминированная проверка источников

| Модель | Валидные источники | Строгий формат цитирования |
| --- | ---: | ---: |
| GPT-5.6 Terra | **119 / 119** | **119 / 119** |
| GPT-5.6 Luna | **119 / 119** | 108 / 119 |
| Qwen3:8B | 115 / 119 | 111 / 119 |

После Sol все critical-флаги дополнительно просмотрены по тому контексту, который реально был доступен моделям. В вопросах **57** и **116** часть информации, необходимой для полного reference-ответа, отсутствовала в frozen Top-10. Поэтому raw critical-флаги Luna и Terra на этих двух вопросах нельзя трактовать как чистую ошибку генерации: это одновременно сигнал о проблеме retrieval/context coverage.

У Qwen большинство отмеченных Sol проблем подтверждаются исходными фрагментами; для отдельных случаев степень серьёзности флага может быть дискуссионной, поэтому raw judge output и подробный audit сохранены отдельно.

Ключевые файлы:

- [`README результатов`](docs/results/generation_holdout_2026-10-06/README.md) — краткая методика и сводка;
- [`SOL_JUDGE_SUMMARY.json`](docs/results/generation_holdout_2026-10-06/SOL_JUDGE_SUMMARY.json) — итог Sol;
- [`SOL_JUDGE_PER_MODEL.csv`](docs/results/generation_holdout_2026-10-06/SOL_JUDGE_PER_MODEL.csv) — оценки по каждому вопросу и каждой модели;
- [`DETERMINISTIC_V2_SUMMARY.json`](docs/results/generation_holdout_2026-10-06/DETERMINISTIC_V2_SUMMARY.json) — source validity и citation format;
- [`CRITICAL_AUDIT.txt`](docs/results/generation_holdout_2026-10-06/CRITICAL_AUDIT.txt) — вопрос, Top-10, реальные ответы трёх моделей и объяснения Sol для отмеченных случаев;
- [`benchmarks/generation/main_119/`](benchmarks/generation/main_119/) — frozen model input и prompt.

**Ограничение:** MAIN119 состоит из answerable-вопросов. По этому набору нельзя делать вывод о качестве безопасного отказа при недостатке данных — для этого нужен отдельный negative/refusal benchmark.

### Остановка Ollama

Для CPU:

~~~powershell
docker compose -f compose.yaml -f compose.cpu.yaml stop ollama
~~~

Для GPU:

~~~powershell
docker compose -f compose.yaml -f compose.gpu.yaml stop ollama
~~~

Модель остаётся в постоянном Docker-томе: после обычной остановки скачивать её заново не требуется. Не используйте `down -v`, если хотите сохранить загруженные модели.

## Как коллегам использовать проект

**Чтобы посмотреть код или запустить тесты**, достаточно Git и Docker. Для реального BGE-M3 нужен контейнер Ollama и однократная загрузка модели. Для reranker benchmark дополнительно используется отдельный optional runtime и persistent Hugging Face cache; подробности — в [`docs/local_reranker.md`](docs/local_reranker.md).

Текущий `OllamaEmbeddingProvider` обращается к Ollama по внутреннему адресу Docker `http://ollama:11434`. Порт Ollama наружу не публикуется; Python-сервис общается с ней внутри сети Compose.

Для полного воспроизведения retrieval TEAM HOLDOUT из исходных документов нужны приватные team inputs — они в GitHub не публикуются. Для generation MAIN119 публично зафиксированы уже подготовленный Top-10 input и prompt в [`benchmarks/generation/main_119/`](benchmarks/generation/main_119/), а опубликованные результаты находятся в [`docs/results/generation_holdout_2026-10-06/`](docs/results/generation_holdout_2026-10-06/).

Colab notebook: [`notebooks/generation_benchmark_colab.ipynb`](notebooks/generation_benchmark_colab.ipynb).

### Работа с документами

Для регистрации локальных файлов можно положить разрешённые материалы в `data/incoming` и выполнить:

~~~powershell
docker compose run --rm app python -m knowledge_base.cli scan
~~~

Этот шаг регистрирует документы; он **не запускает автоматически** embeddings или RAG-ответы.

Не загружайте в репозиторий документы заказчика, медицинские данные, ключи или иные закрытые материалы. `data/incoming/`, `data/processed/`, `data/team/`, `.env`, `reports/` и `docs/private/` исключены через `.gitignore`; перед каждым коммитом дополнительно проверяйте `git status` и `git diff --cached`.

## Следующие задачи

1. Не тюнинговать retrieval или generation под MAIN119 и не считать повторные прогоны на том же наборе новой независимой проверкой.
2. Разобрать retrieval/context coverage для вопросов 57 и 116 и проверить улучшения уже на новом контрольном наборе.
3. Подготовить отдельный negative/refusal benchmark для проверки корректного отказа при недостатке источников.
4. Подготовить EXTENDED149 как отдельный диагностический generation-набор после проверки 30 provisional-вопросов.
5. Перенести пригодные generation/evaluation-компоненты из экспериментального контура в стабильный project tooling и затем собрать production-путь с проверяемыми цитатами и эскалацией эксперту.
6. Продолжить production-задачи: persistent vector DB, source/version/access filters и безопасное обновление базы знаний.

Никакие компоненты медицинского исполнения не входят в этот поисковый прототип.

---

Вопросы, замечания и предложения по интеграции можно оформлять через GitHub Issues и Pull Requests. Перед публикацией результатов убедитесь, что в них нет закрытых материалов заказчика.
