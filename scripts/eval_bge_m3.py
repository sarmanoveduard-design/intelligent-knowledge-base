from benchmark_data import documents, questions
from knowledge_base.chunker import Chunk
from knowledge_base.ollama_embeddings import OllamaEmbeddingProvider
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore






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
