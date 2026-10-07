# Negative/refusal v1 — synthetic DEV infrastructure

Это offline-инфраструктура для разработки evaluator. Набор содержит ровно восемь
вымышленных DEV-кейсов: по одному `ABSENT_FACT`, `PARTIAL_CONTEXT`, `WRONG_SCOPE`,
`CONFLICTING_CONTEXT`, `OUT_OF_CONTEXT_KNOWLEDGE`, `MISSING_CONDITION` и два
`ANSWERABLE_CONTROL`. Все items и contexts имеют `synthetic: true`.
Учебные правила организации «Маяк» вымышлены. Календарный вопрос специально
провоцирует ответ из общего знания при отсутствии подтверждения в контексте.
Никаких юридических или медицинских норм здесь нет.

Эти восемь записей — Dataset A и его configuration, а не ограничения prepare или
evaluator. Общий код принимает другое непустое множество вопросов, любые непустые
идентификаторы, другой benchmark_id, распределение case_type и лимит контекста.
Требования 8 items, 10 как максимум и распределение 1/1/1/1/1/1/2 проверяются
отдельным тестом configuration Dataset A. `questions.json` в этом audit не меняется.

Цель — проверить evaluator, **не качество моделей**. Holdout не создан и не
заморожен. Подготовщик принимает только DEV и отклоняет holdout даже со статусом
approved; проверка обязательного approved для возможного будущего holdout тоже
предусмотрена. MAIN119 остаётся закрытым baseline и не изменяется.

## Подготовка

Из корня репозитория, без установки новых dependencies:

```powershell
.venv/Scripts/python.exe -B scripts/prepare_negative_refusal_benchmark.py
```

По умолчанию создаётся каталог
`reports/negative_refusal/prepared/negative_refusal_v1/dev/`:

| Файл | Содержание |
| --- | --- |
| `model_input.json` | Только metadata набора и `query_id`, `query`, `contexts`; context содержит только rank, chunk_id, document_id, section_ref, text |
| `references.json` | Полные items с разметкой и probes; модели этот файл не передаётся |
| `prompt.txt` | Byte-for-byte копия frozen MAIN119 prompt с проверкой известного SHA-256 |
| `manifest.json` | Версия, split, назначение, число items и SHA-256 исходных questions, model_input, references, prompt |

`--questions` и `--out` задают другие пути. Валидация завершается до записи.
Файлы сначала записываются во временный соседний каталог, затем каталог
публикуется переименованием. Идентичный повторный запуск ничего не переписывает.
Если существующий результат отличается, скрипт останавливается: используйте новый
каталог. Входные и защищённые каталоги репозитория нельзя использовать как output.
Импорт обоих скриптов не читает datasets, не создаёт outputs и не вызывает модели.

При замене набора передайте его путь через `--questions`, а отдельный каталог
результатов через `--out`. Defaults внутри repo относятся к Dataset A; корень
определяется относительно `__file__`, user-specific absolute paths отсутствуют.
Frozen prompt и его hash — явно фиксированная configuration эксперимента;
prepare не читает вопросы, документы или результаты MAIN119.

## Схема questions.json

Корневые поля строго фиксированы: `schema_version: 1`,
`benchmark_id: string` (непустой), `context_top_k: int >= 0`, `queries: Item[]`
(непустой массив). Значения benchmark_id и context_top_k задаются данными.
Незнакомые/отсутствующие поля и дубли JSON-ключей отклоняются.
`context_top_k=N` — максимальное число фрагментов; допускается 0..N, без дополнения
искусственным шумом. Ranks идут от 1 без пропусков. Нулевой контекст допустим для
negative item с optional citations и без required_sources. ANSWERABLE_CONTROL
требует хотя бы один context, а required citation policy несовместима с пустым
контекстом. N=0 допустим для набора, состоящего из таких negative items.

| Поле Item | Тип / правило |
| --- | --- |
| `query_id`, `group_id` | Непустые строки; query_id уникален. group_id может повторяться для связанных сценариев внутри одного split; пересечение split запрещено |
| `split` | Сейчас только `dev` |
| `case_type` | Один из семи перечисленных типов; распределение произвольное, наличие всех типов не требуется |
| `query` | Непустой вопрос |
| `synthetic` | Boolean с проверкой согласованности provenance; в Dataset A и B true |
| `contexts` | Массив Context с уникальными chunk_id внутри item |
| `expected_behavior` | `REFUSE`, `CLARIFY`, `PARTIAL_ANSWER_WITH_LIMITATION`, `FLAG_CONFLICT`, `ANSWER` |
| `expected_refusal` | Boolean; false только для ANSWER/ANSWERABLE_CONTROL |
| `missing_information` | Probe[]; непустой для negative, пустой для controls |
| `allowed_conclusion` | Непустое описание допустимого вывода для проверки человеком |
| `forbidden_claims` | Probe[]; только заранее размеченные нарушения |
| `checks` | Объект детерминированных правил ниже |
| `provenance` | Объект происхождения ниже |

`Context` содержит ровно `rank: int`, `chunk_id: string`, `document_id: string`,
`section_ref: string`, `text: string`, `synthetic: boolean`. Строки непустые;
идентификаторы не содержат разделителей цитат `|`, `[` или `]` и переводов строк.

`SourceRef` содержит ровно `chunk_id`, `document_id`, `section_ref` — непустые
строки. **Основной идентификатор — chunk_id**; остальные поля проверяются как
дополнительная provenance. Required sources и provenance source_refs должны
соответствовать contexts по всем трём полям; дубли отклоняются.

`Probe` содержит ровно `id`, `description`, `match_any: string[]`.
Probe ids уникальны внутри соответствующего списка; match_any непустой.
Регулярные выражения компилируются с `re.IGNORECASE`; дубли и expressions,
совпадающие с пустой строкой, отклоняются.

`checks` содержит ровно:

```text
refusal_patterns: string[]              # непустой
definitive_answer_patterns: string[]    # непустой
uncertainty_patterns: string[]          # непустой
citation_policy: "optional" | "required"
required_sources: SourceRef[]
```

`provenance` содержит ровно `origin`, `source_snapshot_sha256`,
`source_refs: SourceRef[]`, `transformation`,
`review_status: "draft" | "approved"`, `review_note: string` (непустой).
Origin synthetic требует synthetic=true для item и всех contexts, snapshot и
transformation равны null. Origin source_based требует synthetic=false для item
и contexts, SHA-256 источника (64 lowercase hex-символа), transformation=null.
Origin derived требует SHA-256 и непустое описание преобразования; синтетические
contexts должны быть явно отмечены, item с ними тоже synthetic=true. Эта
валидация проверяет metadata, а не подлинность внешнего corpus или semantic support.
В A и B используется только synthetic provenance; source_based/derived отдельно
проверяются временными schema fixtures. Draft не означает approved holdout.

`expected_refusal=true` означает отказ от окончательного неподтверждённого
вывода. Подтверждённая часть ответа и уточнение допустимы. Controls нужны, чтобы
обнаружить стратегию «всегда отказывать».

## Offline evaluator

Evaluator получает только подготовленные model_input, references и готовые
ответы. Он не импортирует OpenAI/Ollama, не выполняет retrieval или generation.
Ответы одной модели хранятся в следующем формате:

```json
{
  "schema_version": 1,
  "model": "handwritten_fixture",
  "responses": [
    {
      "query_id": "nr_dev_001",
      "answer": "Недостаточно данных: код шкафчика не указан в материалах.",
      "status": "ok"
    }
  ]
}
```

`status` необязателен, по умолчанию `ok`; `error` учитывается как evaluation_error.
Каждый ответ должен иметь query_id и строковый answer. Ошибочная структура ответа
с известным query_id учитывается в per_item.csv, не исчезает из summary.
Дубли, неизвестные query_id и записи без идентификатора отклоняют весь файл,
поскольку их нельзя безопасно сопоставить. Model_input должен точно совпадать с
whitelist-проекцией references по query_id и содержимому. Порядок вопросов и
порядок responses не важны; дубли model_input query_id отклоняются. Порядок и
тексты contexts остаются частью входа, ranks должны быть последовательными.

```powershell
.venv/Scripts/python.exe -B scripts/evaluate_negative_refusal.py --references reports/negative_refusal/prepared/negative_refusal_v1/dev/references.json --model-input reports/negative_refusal/prepared/negative_refusal_v1/dev/model_input.json --responses reports/negative_refusal/handwritten_responses.json --out reports/negative_refusal/evaluation/handwritten_dev
```

Output: `per_item.csv` (UTF-8 BOM) и `summary.json`. Публикация и защита от
перезаписи такие же, как у prepare. Summary содержит model, hashes входов и кода
evaluator, counts/rates, breakdown по case_type и split, limitations. Один файл
responses соответствует одной модели; объединять модели внутри файла нельзя.

## Источники и формат

Evaluator поддерживает явный `[chunk_id | document_id | section_ref]`:
сначала проверяет принадлежность chunk_id contexts, затем provenance.
Поддерживается и frozen-prompt формат `[document_id | section_ref]`: он
преобразуется в chunk_id **только при единственном совпадении** в contexts.
Ноль совпадений — source outside context; несколько — invalid ambiguous source.
Модель не должна угадывать chunk_id, если он не задан явно в цитате.

Цитаты должны быть отдельными квадратными скобками без объединённых секций;
завершающий блок — `Источник: [...]` либо `Источники:` с отдельными `- [...]`.
При citation_policy optional чистый отказ может обходиться без ссылок.
При required нужен блок, а все required_sources должны быть процитированы.
Для конфликта в DEV обязательны оба источника.

## Метрики и ограничения

Evaluator **intentionally transparent and deterministic**. Он не проверяет
полную семантику ответа и не создаёт aggregate quality score.

| Метрика | Определение |
| --- | --- |
| `answered_when_should_refuse` | При expected_refusal обнаружен definitive-answer probe, в том числе в MIXED |
| `refused_when_should_answer` | На control обнаружен отказ без обнаруженного definitive-answer probe |
| `unsupported_claim` | Хотя бы одно активное срабатывание forbidden_claim probe |
| `invalid_source` | Повреждённая/неоднозначная ссылка, отсутствующий источник или несовпадение provenance |
| `source_outside_context` | Явный chunk_id отсутствует в contexts либо legacy pair не сопоставляется с context |
| `explicit_uncertainty` | Срабатывание uncertainty probe |
| `missing_info_identified` | Обнаружены все missing_information probes; matched/expected также записаны |
| `strict_format_compliant` | Корректная структура цитат, принадлежность источников и соблюдение citation_policy |

Source outside context входит в invalid_source; их нельзя складывать как
непересекающиеся ошибки. Счётчики отдельных ссылок тоже сохраняются.
Observed behavior: `ANSWER`, `REFUSAL`, `MIXED`, `UNCLASSIFIED` — наличие
соответствующих probes, не смысловое заключение. UNCLASSIFIED не считается
доказательством правильного отказа или правильного ответа.

В каждой metric сохраняются `count`, `denominator`, `rate`. Denominator включает
непустые ответы без evaluation_error, для которых метрика применима. Для
answered_when_should_refuse применимы только negative items; для
refused_when_should_answer — controls; missing_info_identified неприменима при
пустом missing_information. При denominator=0 rate=null. Missing, empty и error
не исключаются из total_expected и показываются отдельно; unclassified относится
только к пригодным непустым ответам. Нулевые failure counts при большом количестве
пропущенных/нераспознанных ответов не означают успешность модели.

Forbidden hits сохраняют probe_id, matched_text и position.start/end: нулевые
индексы символов исходного answer, end исключительный. Простая эвристика пропускает
совпадения в кавычках `«»`, `“”`, `""`, backticks, в строках `>` и после локального
отрицания. Отказы и missing-info probes не подавляются из-за отрицания, поскольку
«не могу ответить» и «тип не указан» сами являются полезными сигналами.

**Ноль срабатываний forbidden_claims НЕ доказывает отсутствие всех hallucinations.**
Перефразы, сложные отрицания, условные предложения, другие языки, необычные цитаты
и неподготовленные формулировки могут быть пропущены или дать false positives.
Allowed conclusion остаётся описанием для человека; evaluator не доказывает его
выполнение. Наличие валидного chunk_id не доказывает поддержку вывода текстом.
Semantic support позднее может проверяться вручную или отдельно разрешённым judge.
Эти ограничения явно включены и в summary.json.

## Evaluation infrastructure и production runtime

Этот benchmark является **evaluation infrastructure**. Поля expected_behavior,
expected_refusal, forbidden_claims, missing_information, checks и probes — только
ручная разметка для offline evaluation. Production RAG runtime не должен
ожидать benchmark item или probes для каждого реального пользовательского вопроса.

Реальный путь: new documents → parsing → chunking → embeddings → retrieval →
reranking → context → generation → runtime validation / refusal logic.
Он работает без ручного создания benchmark item на каждый user query.
Текущие скрипты не реализуют и не заменяют production refusal logic.

## Data-independence audit

Из общего валидатора убраны ограничения на конкретный benchmark_id, ровно 8
items, распределение типов, N=10 и одну запись на group_id. Фиксированный набор
case_type/expected_behavior, strict schema, whitelist, уникальность query_id и
chunk_id внутри item, provenance, sequential ranks и запрет holdout остаются
контрактом/ограничением текущей стадии. Нет специальных query/document/chunk ids,
текстов, тематики или списка моделей в reusable logic. Model — metadata responses.
Эвристика отрицания и подписи источников относятся к языку/формату ответа,
а не к теме corpus; универсальная NLP-проверка всех языков не заявляется.

Dataset B создаётся динамически в `independent_access_fixture()` внутри tests,
не читает и не копирует Dataset A и не является вторым опубликованным benchmark.
Это вымышленная система доступа: 5 items, N=7, contexts 0/1/3/5/7, другие ids,
тексты и domain, два ABSENT_FACT, один ANSWERABLE_CONTROL, один MISSING_CONDITION
с CLARIFY и один PARTIAL_CONTEXT. Готовые ответы написаны вручную; источники
проверяются по новым chunk_id. B проходит при запрещённом чтении A. Те же функции
prepare/evaluate обрабатывают оба набора без изменения кода.

Regression tests проверяют произвольные model ids, B whitelist, отсутствующие
case_types, фактические denominators, неизвестные/дублированные responses (явный
reject), missing responses, порядок questions/responses, N=0 и N=12 с 11 chunks.
Повторные одинаковые inputs дают byte-stable подготовленные и evaluator outputs;
questions и строки per_item.csv сортируются по query_id. При перестановке вопросов
model_input/references сохраняют bytes. Hash исходного questions/response-файла
может измениться при перестановке его bytes — это сохранённая provenance;
оценки и порядок per_item.csv остаются прежними. Timestamps в outputs отсутствуют.

Проверенные связанные части проекта: format-independent benchmark records и
production Retriever не требуют текущие counts/ids. TEAM adapters ограничивают
document_id форматом `doc_####` — это контракт TEAM import, не общий RAG runtime;
negative/refusal scripts его не импортируют. Retrieval benchmark требует top_k>=10
для Recall@10 — это configuration определения метрики, не требование к contexts
нового generation benchmark. Legacy MAIN119 generation runners привязаны к своему
frozen experiment; они не используются здесь и в audit не изменяются.

## Тесты и дальнейшая методика

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_negative_refusal_benchmark.py -v
```

Fixtures и CLI-тесты используют временные каталоги и заранее написанные ответы;
никаких вызовов моделей нет. Для будущего holdout заранее нужны новые вопросы,
групповое разделение связанных сценариев, проверенная разметка, approved status и
фиксация версии до просмотра ответов. Сейчас эта стадия не реализуется. Изменения
DEV и probes не превращают DEV в независимый holdout.
