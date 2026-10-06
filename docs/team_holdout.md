# TEAM HOLDOUT importer

HOLDOUT независим от DEV `data/team/gold_qa.jsonl`. Importer читает только
заданный raw-каталог, не меняет источники, не вызывает модели, API или Docker.
Код использует стандартную библиотеку Python 3.10+; установка Excel-библиотек
не нужна. Все team inputs и outputs остаются под игнорируемым `data/team/`.
Тесты содержат только синтетические данные.

## Запуск из корня проекта (PowerShell)

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_team_holdout.py -v
.venv/Scripts/python.exe scripts/prepare_team_holdout.py
```

Аргументы: `--raw-dir` (по умолчанию `data/team/holdout/questions/raw`),
`--out` (`data/team/holdout/normalized`), `--near-threshold` (0.92).
Читаются все CSV/XLSX непосредственно в каталоге, в стабильном порядке.
Выходной каталог должен находиться вне raw. Повторный запуск заменяет четыре
outputs; при неизменных inputs/параметрах их байты одинаковы.

## Форматы и правила

CSV: UTF-8 с/без BOM, UTF-16 с BOM, fallback CP1251 (кодировка записана
в manifest). Автоматический выбор разделителя: запятая, точка с запятой,
tab или `|`. Quoted fields и многострочные поля обрабатываются CSV parser.
`source_row` — физическая начальная строка записи, с нумерацией от 1.

XLSX: все worksheets через workbook relationships, shared strings, rich text,
inline strings, sparse cells. `source_row` — номер строки Excel. Формулы
не вычисляются: строки с формулами или Excel errors сохраняются как provisional
и отмечаются malformed. Это importer значений, не Excel calculation engine.
Пустые строки, включая оформленные Excel строки, не считаются вопросами.

Заголовок ищется среди непустых строк по присутствию query и положительного ref.
Преамбула сохраняется в validation. Повторные заголовки распознаются.
Неизвестные колонки перечисляются в metadata; дубли названий одной логической
колонки считаются неоднозначными, raw-строка сохраняется в отчёте.

| Поле | Примеры заголовков |
| --- | --- |
| query | query, question, Вопрос |
| query_id | query_id, question_id, ID вопроса |
| difficulty | difficulty, Сложность |
| answer_hint | answer_hint, Подсказка для проверки |
| document_id | document_id, doc_id, ID документа |
| positive_section_refs | section_ref, positive_section_refs, Правильный section_ref, article_tag |
| hard negatives | hard_negative_1_section_ref/document_id, hard_negative_1, Похожий неверный section_ref |
| status | status, Статус |

Нумерованные hard negatives поддерживают любое число slots, включая русские
заголовки с номером. Несколько refs разделяются `;`, запятой, `|` или переводом
строки, либо передаются JSON массивом строк. Например,
`clause_11_1; clause_11_2` становится двумя positive refs.
`hard_negative_refs` принимает разделённые refs, включая `doc_0999:clause_1`;
`hard_negatives` также принимает JSON массив объектов.
Пустые negatives допустимы. Если negative ref есть, а его document_id отсутствует,
используется document_id вопроса. Negative document_id без ref — unresolved.

При отсутствии document_id используется единственный `doc_XXXX` из имени файла,
затем из query_id; это отражается в validation. ID не перенумеровываются.
Отсутствующие query_id/query/positive refs, невалидные IDs/refs и пересечение
positive/negative refs отмечаются unresolved; строка остаётся в gold как provisional.
Неоднозначные поля, лишние CSV cells и Excel formula/error cells — malformed.
Одна строка может попасть в обе категории. Повреждённые/нераспознанные источники
учитываются отдельно как unresolved sources, остальные файлы продолжают импорт.

Синтаксис document_id: `doc_` и четыре цифры. Section refs — непробельные
идентификаторы с начальной буквой и буквами/цифрами/`_`/`.`/`-` далее.
Кириллические буквы допустимы, например `clause_5_а`.
**Проверка существования refs в корпусе не выполняется**; это явно записано в
manifest/validation. Она необходима позже в benchmark adapter.
Текст и кириллица сохраняются в UTF-8, у полей убираются только внешние пробелы.

## Статусы и будущий benchmark

`status` нормализован в `approved` или `provisional`. Признаки draft/черновик,
pending/согласование/review в статусе, имени файла, листа или преамбуле дают provisional.
`human_checked=НЕТ/no/false/0`, неизвестный непустой статус и невалидные строки
тоже дают provisional. Признак файла имеет приоритет над row approved.
Явные approved/утвержден/согласовано/готово принимаются как approved.
При отсутствии статуса и этих признаков используется **approved по умолчанию**;
это правило импорта, а не свидетельство отдельного человеческого согласования.
Исходный статус и human_checked сохранены в validation.

Для будущего runner предоставлена функция `select_questions(records, mode)`:
`approved-only` (default) возвращает approved, `all` возвращает все строки.
Она не удаляет duplicates. Подготовка snapshot для существующего runner
описана ниже; сам runner не изменён.

## Подготовка HOLDOUT для team benchmark

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe scripts/prepare_team_holdout_benchmark.py
.venv/Scripts/python.exe scripts/prepare_team_holdout_benchmark.py --mode all
```

Первый запуск выбирает только `status=approved` и пишет `corpus.json`, `gold.json`,
`manifest.json` в `data/team/holdout/benchmark/approved/`. Режим `all` сохраняет
все вопросы в `data/team/holdout/benchmark/all/`. Аргументы `--corpus-dir`,
`--gold`, `--out` позволяют задать другие локальные пути. Это только подготовка
и проверка файлов; retrieval, embeddings, модели, API и Docker не запускаются.

`TeamHoldoutGoldAdapter` проверяет точную normalized schema, типы, provenance
и оба допустимых статуса **до фильтрации**. В памяти превращает
`positive_section_refs` в `positive_refs`, а `hard_negatives` в
`hard_negative_refs`. `TeamGoldAdapter` проверяет все вопросы, включая исключённые
provisional: IDs, difficulty, несколько positive refs, hard negatives, дубли и
пересечения refs. Каждый ref должен существовать в загруженном corpus; covered_by
не используется для неявного перенаправления на другой chunk.

Используются существующие `TeamCorpusAdapter`, map validation, canonical chunk IDs
и общий serializer benchmark snapshots. Для каждого corpus document обязателен
ровно один embedding map. Подготовленный gold сохраняет status, source_file и
source_row в metadata, answer_hint остаётся metadata существующего pipeline.
Normalized gold и исходные файлы не изменяются. При любой ошибке возвращается
exit code 1, новые outputs не создаются и существующие outputs не перезаписываются.
При пустой выборке также возвращается ошибка.

Все три outputs детерминированы; manifest не содержит времени запуска. Он включает
mode, source/selected question counts, approved/provisional counts исходного набора
и выбранных вопросов, document IDs/count, chunk count, unresolved_refs_count=0,
SHA256 snapshot corpus/gold и SHA256 всех прочитанных corpus/map/normalized gold
sources. Source hashes описывают именно прочитанные и проверенные байты.

## Дубли и outputs

Автоматического удаления нет. Отдельно выявляются группы одинаковых query_id
во всём HOLDOUT и пары одинаковых/почти одинаковых query внутри document_id.
Для сравнения query используется NFKC, casefold, свёртка пробелов/пунктуации
и ё→е; near duplicates — `SequenceMatcher(autojunk=False).ratio() >= threshold`.
Это текстовый эвристический сигнал для проверки, не семантическая модель.
Exact и near пары не пересекаются. Query ID groups и excess rows имеют отдельные
счётчики. Record indices в duplicate report — номера строк gold **от 0**;
`record_locations` связывает их с исходными файлом, листом и строкой.

- `gold.jsonl`: все импортированные candidate question rows в единой структуре,
  включая provisional и невалидные. Ни одна candidate row не удаляется.
- `manifest.json`: количества файлов/вопросов/document IDs, список IDs,
  approved/provisional, duplicate counts, malformed/unresolved counts,
  SHA256 каждого raw файла и normalized gold, обнаруженные схемы и параметры.
- `validation.json`: machine-readable issues, source statuses, raw invalid rows,
  preamble, duplicate groups/pairs и sheet provenance.
- `VALIDATION.md`: сводка, источники с SHA256 и findings для ручной проверки.

Exit code 0 означает отсутствие malformed/unresolved, но не отсутствие дублей
или provisional. Exit code 1 сигнализирует malformed/unresolved либо невозможность
запуска. При проблемах отдельных источников отчёты всё равно создаются.
Внешний `questions/MANIFEST.sha256.txt` не изменяется; importer считает SHA256
фактически прочитанных raw файлов и проверяет неизменность во время импорта.

Перед добавлением кода в Git проверьте `git status --short` и
`git check-ignore data/team/holdout/normalized/gold.jsonl`.
Не используйте `git add -f` для team inputs/outputs.
