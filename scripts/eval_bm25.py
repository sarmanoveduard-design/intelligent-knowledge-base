import math
import re
from collections import Counter

from benchmark_data import documents, questions


def tokenize(text):
    return re.findall(r"[^\W_]+", text.casefold())


# Индексируем те же документы, которые использовали для BGE-M3.
tokens = [tokenize(text) for text in documents]
frequencies = [Counter(words) for words in tokens]

count = len(documents)
avg_length = sum(map(len, tokens)) / count

document_frequency = Counter(
    word
    for words in tokens
    for word in set(words)
)


def bm25_score(question, document_id):
    k1 = 1.5
    b = 0.75

    length = len(tokens[document_id])
    frequency = frequencies[document_id]
    score = 0.0

    for word in set(tokenize(question)):
        tf = frequency[word]
        if not tf:
            continue

        df = document_frequency[word]
        idf = math.log(1 + (count - df + 0.5) / (df + 0.5))

        score += idf * (
            tf * (k1 + 1)
            / (tf + k1 * (1 - b + b * length / avg_length))
        )

    return score


if __name__ == '__main__':
    hits_at_1 = 0
    hits_at_3 = 0

    for question, expected in questions:
        scores = [
            (i, bm25_score(question, i))
            for i in range(count)
        ]

        # Не показываем документы с нулевым совпадением.
        ranked = sorted(
            (item for item in scores if item[1] > 0),
            key=lambda item: (-item[1], item[0]),
        )[:3]

        found = [i for i, score in ranked]

        hits_at_1 += int(bool(found) and found[0] == expected)
        hits_at_3 += int(expected in found)

        print("\nВОПРОС:", question)
        print("ОЖИДАЛСЯ ДОКУМЕНТ:", expected)
        print("НАЙДЕНО:", found)
        print("ОЦЕНКИ:", [(i, round(s, 4)) for i, s in ranked])

    print("\n===== BM25 =====")
    print(f"Hit@1: {hits_at_1}/{len(questions)}")
    print(f"Hit@3: {hits_at_3}/{len(questions)}")
