"""Объединение результатов поисковых систем по их позициям."""


def reciprocal_rank_fusion(rankings, k=60, limit=3):
    if k <= 0 or limit <= 0:
        raise ValueError("k и limit должны быть положительными")

    scores = {}
    first_positions = {}

    for search_index, ranked in enumerate(rankings):
        for position, document_id in enumerate(dict.fromkeys(ranked), 1):
            scores[document_id] = (
                scores.get(document_id, 0)
                + 1 / (k + position)
            )
            first_positions.setdefault(
                document_id, (search_index, position)
            )

    return sorted(
        scores,
        key=lambda i: (
            -scores[i],
            first_positions[i],
            i,
        ),
    )[:limit]
