from knowledge_base.chunker import Chunk
from knowledge_base.ollama_embeddings import OllamaEmbeddingProvider
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore


texts = (
    "Для поиска доступны только утверждённые версии документов. Черновики не публикуются.",
    "Сотрудникам предоставляется доступ к рабочему календарю организации.",
    "После завершения обучения пользователь проходит проверочный тест.",
)

chunks = tuple(
    Chunk(
        document_id="demo",
        version_id="v1",
        chunk_index=i,
        text=text,
        source_file="synthetic-demo.txt",
        source_format="txt",
        block_indices=(i,),
        page_numbers=(),
        paragraph_indices=(),
        section_titles=(),
    )
    for i, text in enumerate(texts)
)

provider = OllamaEmbeddingProvider()
store = InMemoryVectorStore(dimension=provider.dimension)
retriever = Retriever(
    embedding_provider=provider,
    vector_store=store,
)

retriever.index_chunks(chunks)

question = "Какие версии документов разрешено использовать при поиске?"

print("ВОПРОС:", question)
print("\nРЕЗУЛЬТАТЫ ПОИСКА:")

for result in retriever.search(question, top_k=3):
    print(
        f"{result.score:.4f} | "
        f"{result.chunk.text}"
    )
