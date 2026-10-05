"""Объединение результатов поисковых систем по их позициям."""


def _fusion_scores(rankings, k):
    if k <= 0:
        raise ValueError("k должен быть положительным")
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

    return scores, first_positions


def reciprocal_rank_fusion_scores(rankings, k=60):
    """Scores from the same algorithm used by reciprocal_rank_fusion."""
    return _fusion_scores(rankings, k)[0]


def reciprocal_rank_fusion(rankings, k=60, limit=3):
    if limit <= 0:
        raise ValueError("limit должен быть положительным")
    scores, first_positions = _fusion_scores(rankings, k)
    return sorted(
        scores,
        key=lambda i: (
            -scores[i],
            first_positions[i],
            i,
        ),
    )[:limit]
