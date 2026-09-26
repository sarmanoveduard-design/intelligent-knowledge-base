from knowledge_base.chunker import Chunk
from knowledge_base.ollama_embeddings import OllamaEmbeddingProvider
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore


documents = (
    "Для поиска используются только опубликованные и утверждённые версии документов.",
    "При изменении документа сохраняется история его предыдущих версий.",
    "Руководитель может просматривать календарь обучения сотрудников.",
    "Если ответ не подтверждается источником, вопрос передают человеку-эксперту.",
    "Персональные медицинские сведения запрещено помещать в поисковый индекс.",
    "После завершения обучения сотрудник проходит итоговую проверку знаний.",
)

questions = (
    ("Можно ли отвечать на основании неопубликованного черновика?", 0),
    ("К кому обращаться, если в документах не найден ответ?", 3),
    ("Можно ли индексировать индивидуальные медицинские сведения?", 4),
    ("Где руководитель может посмотреть расписание обучения?", 2),

    ("Какие документы разрешено использовать: утверждённые или любые?", 0),
    ("Зачем сохраняется история предыдущих версий документов?", 1),
    ("Что происходит со старыми версиями после обновления?", 1),
    ("Где руководитель смотрит календарь обучения?", 2),
    ("Кому передать вопрос без подтверждающего источника?", 3),
    ("Разрешается ли добавлять персональные медданные в индекс?", 4),
    ("Как проверяют знания после завершения обучения?", 5),
    ("Проходит ли сотрудник итоговый тест после курса?", 5),
)

chunks = tuple(
    Chunk(
        document_id=f"synthetic-{i}",
        version_id="v1",
        chunk_index=i,
        text=text,
        source_file=f"synthetic-{i}.txt",
        source_format="txt",
        block_indices=(0,),
        page_numbers=(),
        paragraph_indices=(),
        section_titles=(),
    )
    for i, text in enumerate(documents)
)

provider = OllamaEmbeddingProvider()
store = InMemoryVectorStore(dimension=provider.dimension)
retriever = Retriever(provider, store)
retriever.index_chunks(chunks)

hits_at_1 = 0
hits_at_3 = 0

for question, expected in questions:
    results = retriever.search(question, top_k=3)
    found = [r.chunk.chunk_index for r in results]

    hits_at_1 += int(found[0] == expected)
    hits_at_3 += int(expected in found)

    print("\nВОПРОС:", question)
    print("ОЖИДАЛСЯ ДОКУМЕНТ:", expected)
    print("НАЙДЕННЫЕ ДОКУМЕНТЫ:", found)
    print("TOP-1:", results[0].chunk.text)
    print("ОЦЕНКИ:", [(r.chunk.chunk_index, round(r.score, 4)) for r in results])

total = len(questions)

print("\n===== РЕЗУЛЬТАТ =====")
print(f"Hit@1: {hits_at_1}/{total}")
print(f"Recall@3: {hits_at_3}/{total}")
