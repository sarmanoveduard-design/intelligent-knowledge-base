# Импорт командного Retrieval Benchmark

Это retrieval benchmark, не оценка качества LLM-ответа.

`TeamCorpusAdapter` и `TeamGoldAdapter` преобразуют командные файлы в независимые
от CSV модели `CorpusChunk` и `BenchmarkCase`. Подготовка не импортирует embedding
провайдеры, не читает `.env`, не вызывает API и не запускает Docker.

## Подготовка

Поместите локальные исходники, например, в `data/team/`. Реальные данные не
добавляются в Git. В каталоге корпуса ищутся непосредственно лежащие файлы
`*_embeddable.csv` и необязательные `*_embedding_map.csv`, без рекурсивного обхода.
Gold передаётся отдельным аргументом и может находиться в другом каталоге.

POSIX shell, при активном Python-окружении:

```sh
PYTHONPATH=src python scripts/prepare_team_benchmark.py \
  --corpus-dir data/team/corpus \
  --gold data/team/gold_qa.csv \
  --out data/benchmark
```

Windows PowerShell, из корня существующего репозитория:

```powershell
$env:PYTHONPATH='src'
& .\.venv\Scripts\python.exe scripts/prepare_team_benchmark.py `
  --corpus-dir 'data/team/corpus' `
  --gold 'data/team/gold_qa.csv' `
  --out 'data/benchmark'
```

Для JSONL замените аргумент `--gold` на `data/team/gold_qa.jsonl`.
Пути с пробелами заключайте в кавычки. В PowerShell перенос обозначается
обратным апострофом, после него не должно быть пробелов.

Результат: `data/benchmark/corpus.json`, `gold.json`, `manifest.json`.
Повторный успешный запуск заменяет эти три файла. Проверка входов полностью
выполняется до записи: при ошибке в данных существующий dataset не меняется.
CLI возвращает 1 и структурный код ошибки; тексты строк и absolute paths
в диагностике не выводятся. Нерезолвленные refs останавливают подготовку,
их количество выводится в ошибке. Успешный manifest всегда содержит 0 unresolved refs.

## Corpus и embedding map

CSV должен быть UTF-8 (BOM допустим), с заголовком. Поддержаны разделители
запятая, точка с запятой и табуляция, стандартные CSV-кавычки и переносы строк
внутри полей. Обязательные колонки:

Разделитель определяется только по первой физической строке header: для `;`,
`,` и табуляции проверяются обязательные колонки и уникальность имён.
Ни один подходящий вариант — `missing_or_duplicate_csv_columns`, несколько —
`ambiguous_csv_delimiter`. Тело файла не используется для выбора разделителя;
после выбора весь CSV читается в strict-режиме с поддержкой кавычек и multiline.
Header должен занимать одну физическую строку.

```csv
section_ref,chapter,article_title,text
article_1,Chapter 1,Article 1,Synthetic text only.
article_2,Chapter 1,Article 2,Another synthetic section.
```

Имя `example(doc_0026)_embeddable.csv` задаёт document_id `doc_0026`.
Также допустимо `doc_0026_embeddable.csv`. Формат ID строгий: `doc_` + четыре
цифры. ID не извлекается из текста или `source_document`. Необязательная
CSV-колонка `document_id`, если она есть, должна совпасть с именем файла.

Каждая строка становится `CorpusChunk`: document_id, section_ref, исходный
text и metadata chapter/article_title/source_file (только basename).
Пробелы по краям section_ref удаляются, текст сохраняется без переформулировки
или добавления заголовков. Пустые text и section_ref запрещены.

chunk_id = document_id + `:` + SHA-256 canonical JSON пары [document_id, section_ref].
Он не зависит от текста, порядка строк или пути каталога. В JSON поле `id`
сохраняет существующий контракт runner. Повтор section_ref внутри документа
(включая разные файлы одного документа) запрещён. Совпадающие section_ref
в разных документах допустимы и дают разные chunk IDs.

Если найдены `*_embedding_map.csv`, каждый должен иметь командные колонки:
`section_ref`, `chapter`, `article_title`, `embedded`, `point_id`, `covered_by`
и document_id в имени по тем же правилам. Map содержит все структурные refs,
а embeddable.csv — только выбранные для embeddings фрагменты.

`embedded` принимает только русские `да`/`нет`: краевые пробелы удаляются,
регистр не учитывается (например, ` ДА ` допустимо). Пустое значение,
английские yes/no, true/false и числа не допускаются. Множество refs map с
`embedded=да` должно точно совпасть с corpus refs соответствующего документа.
Повторы section_ref в map запрещены, включая строки `embedded=нет`.

Строки `embedded=нет` не создают CorpusChunk. Их непустой `covered_by` должен
содержать section_ref строки `embedded=да` того же документа; порядок строк
не важен. Ссылка на отсутствующий ref, неиндексируемую строку или другой
документ запрещена. `covered_by` без document_id задаёт локальный ref.
Пустой covered_by допустим: это oversized parent, разложенный на детей.
`chapter`, `article_title` и `point_id` map не добавляются в retrieval text;
необязательный `document_id`, если указан, также проверяется.
При отсутствии map импорт разрешён, а embedding_map_csv_count равен 0.

## Gold CSV

Обязательные колонки: query_id, query, difficulty, document_id, section_ref.
document_id/section_ref задают primary positive. Пример только на синтетике:

```csv
query_id,query,difficulty,answer_hint,document_id,section_ref,hard_negative_refs,source_document
q1,Which section is relevant?,easy,For later review only,doc_0026,article_1,doc_0026:article_2,Synthetic example
```

query_id непустой и уникальный после удаления краевых пробелов; query непустой;
difficulty строго easy/medium/hard. Все refs обязаны существовать в загруженном
корпусе. Повторы refs и пересечение positives/hard negatives запрещены.

Для нескольких positives можно добавить колонку `positive_refs`. Она задаёт
positive-набор вместо primary, но document_id/section_ref всё равно валидируются.
Поддержанные значения refs:

- JSON-массив объектов `[{"document_id":"doc_0026","section_ref":"article_1"}]`;
- JSON-массив строк `["doc_0026:article_1","doc_0030:article_2"]`;
- строка `doc_0026:article_1;doc_0030:article_2` (также допустим разделитель `#`
  между document_id и section_ref);
- section_ref без document_id, если document_id явно задан в строке gold.

Для hard negatives используйте одну из колонок `hard_negative_refs` или
`hard_negative_section_refs` с теми же правилами. Альтернатива для одного
hard negative: `hard_negative_document_id` + `hard_negative_section_ref`;
без первого используется document_id строки. Одновременные заполненные
форматы неоднозначны и запрещены. Неизвестные заполненные колонки с `negative`
в названии вызывают ошибку, чтобы ссылки не пропали молча.

Фактический командный gold CSV поддерживается с точным заголовком:

```csv
query_id;query;difficulty;answer_hint;document_id;section_ref;hard_negative_1_document_id;hard_negative_1_section_ref;hard_negative_2_document_id;hard_negative_2_section_ref;hard_negative_3_document_id;hard_negative_3_section_ref;source_document
```

Три numbered-пары задают от 0 до 3 hard negatives. Каждая пара либо полностью
пустая, либо содержит оба значения document_id и section_ref. Краевые пробелы
удаляются; заполнение половины пары — ошибка, document_id positive не
подставляется автоматически. Пустые промежуточные пары допустимы.
Для всех refs проверяются формат document_id, существование в corpus,
duplicates и отсутствие пересечения с positives.

Если одновременно заполнены numbered-пары и aggregate-формат
(`hard_negative_refs` / `hard_negative_section_refs`), импорт завершается
ошибкой ambiguous. Numbered-пары также нельзя смешивать с прежней отдельной
singular-парой. Пустые колонки альтернативного формата не мешают импорту;
поддержка прежних aggregate CSV/JSONL сохранена.

В CSV JSON-массивы должны быть оформлены стандартными CSV-кавычками
(удвоенные кавычки внутри поля). Hard negatives необязательны.

## Gold JSONL

Каждая непустая строка — один JSON-объект:

```json
{"query_id":"q1","query":"Which section is relevant?","difficulty":"hard","positive_refs":[{"document_id":"doc_0026","section_ref":"article_1"}],"hard_negative_refs":[{"document_id":"doc_0030","section_ref":"article_2"}],"answer_hint":"For later review only","source_document":"Synthetic example"}
```

positive_refs обязателен и непустой. Для refs без document_id нужен явно
указанный document_id строки. Необязательный section_ref, если указан,
также проверяется вместе с document_id. Остальная валидация совпадает с CSV.

`answer_hint` и `source_document` сохраняются исключительно в metadata
BenchmarkCase и локальном gold.json. В embedding query передаётся только query.
Hints не попадают в corpus text, manifest или benchmark-отчёты. Loader сохраняет
query_id, difficulty и metadata, а legacy JSON без этих полей остаётся совместимым.

## Deterministic snapshots и manifest

Corpus сортируется по document_id/section_ref, gold — по query_id, refs — по
chunk ID. JSON ключи сортируются, компактные separators фиксированы, кодировка
UTF-8 без BOM, окончание — LF. Одинаковые входы дают одинаковые байты
corpus.json/gold.json и hashes; перестановка строк также не меняет snapshots.
Переименование исходника меняет source_file metadata и corpus snapshot.

manifest содержит UTC timestamp, source_csv_count (все corpus/map CSV и gold,
если он CSV), отдельные embeddable_csv_count/embedding_map_csv_count, document_count,
chunk_count, query_count, difficulty_counts, source_files с basename/role/SHA-256,
corpus_snapshot_hash, gold_snapshot_hash и unresolved_refs_count.
Hashes источников вычисляются по тем же байтам, которые разобраны адаптером.
Snapshot hashes — SHA-256 точных canonical bytes, совпадающий с hashes runner.
Timestamp находится только в manifest и не входит в snapshot hashes.
В manifest нет текстов корпуса, queries, hints или абсолютных путей.

## Будущие запуски подготовленного dataset

Эти команды в рамках реализации не запускались. Для OpenAI предполагается
заранее заданный `OPENAI_API_KEY` в окружении; ключ здесь не читается и не печатается.
Для контролируемого сравнения с BGE-M3 обе cloud-модели используют 1024 dimensions.

OpenAI small, PowerShell:

```powershell
$env:PYTHONPATH='src'
$env:EMBEDDING_PROVIDER='openai'
$env:OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
$env:OPENAI_EMBEDDING_DIMENSIONS='1024'
& .\.venv\Scripts\python.exe scripts/eval_retrieval.py --corpus data/benchmark/corpus.json --gold data/benchmark/gold.json --top-k 10
```

OpenAI large:

```powershell
$env:PYTHONPATH='src'
$env:EMBEDDING_PROVIDER='openai'
$env:OPENAI_EMBEDDING_MODEL='text-embedding-3-large'
$env:OPENAI_EMBEDDING_DIMENSIONS='1024'
& .\.venv\Scripts\python.exe scripts/eval_retrieval.py --corpus data/benchmark/corpus.json --gold data/benchmark/gold.json --top-k 10
```

BGE-M3, при уже доступной локальной Ollama:

```powershell
$env:PYTHONPATH='src'
$env:EMBEDDING_PROVIDER='ollama'
$env:OLLAMA_EMBEDDING_MODEL='bge-m3'
$env:OLLAMA_EMBEDDING_DIMENSIONS='1024'
$env:OLLAMA_BASE_URL='http://localhost:11434'
& .\.venv\Scripts\python.exe scripts/eval_retrieval.py --corpus data/benchmark/corpus.json --gold data/benchmark/gold.json --top-k 10
```

Native dimensions: small=1536, large=3072. Можно изменить соответствующую
переменную; input snapshots должны оставаться одинаковыми.

## Git и тесты

`.gitignore` исключает data/benchmark, reports, data/team, data/corpus,
data/corpora, data/eval, data/evaluation, data/datasets, datasets и корневые corpus/eval.
Дополнительно исключены командные файлы *_embeddable.csv, *_embedding_map.csv,
gold_qa.csv, gold_qa.jsonl в любом каталоге. Unit-тесты создают только синтетические
данные во временных директориях; реальные командные dataset-файлы не включены.

```powershell
$env:PYTHONPATH='src'
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
