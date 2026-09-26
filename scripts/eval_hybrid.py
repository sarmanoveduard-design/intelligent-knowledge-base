from benchmark_data import documents, questions
from eval_bm25 import bm25_score

from knowledge_base.fusion import reciprocal_rank_fusion
from knowledge_base.chunker import Chunk
from knowledge_base.ollama_embeddings import OllamaEmbeddingProvider
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import InMemoryVectorStore


# Используем те же документы, что в предыдущем эксперименте.
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


def search_bm25(question):
    scores = [
        (i, bm25_score(question, i))
        for i in range(len(documents))
    ]

    ranked = sorted(
        (item for item in scores if item[1] > 0),
        key=lambda item: (-item[1], item[0]),
    )

    return [i for i, _ in ranked[:3]]



hits_at_1 = 0
hits_at_3 = 0

for number, (question, expected) in enumerate(questions, start=1):
    dense = [
        result.chunk.chunk_index
        for result in retriever.search(question, top_k=3)
    ]

    sparse = search_bm25(question)
    hybrid = reciprocal_rank_fusion((dense, sparse))

    hits_at_1 += int(hybrid[0] == expected)
    hits_at_3 += int(expected in hybrid)

    print(f"\nВОПРОС {number}: {question}")
    print("BGE-M3:", dense)
    print("BM25:  ", sparse)
    print("ГИБРИД:", hybrid)
    print("ОЖИДАЛСЯ:", expected)

    if hybrid[0] != expected:
        print("ВНИМАНИЕ: нужный источник не на первом месте")

print("\n===== ГИБРИДНЫЙ ПОИСК =====")
print(f"Hit@1: {hits_at_1}/{len(questions)}")
print(f"Hit@3: {hits_at_3}/{len(questions)}")
