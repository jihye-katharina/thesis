"""Comparable lecture-level output allowances for segmentation experiments."""


def allocate_summary_budget(source_sizes, budget, per_chunk_limit, minimum=4):
    """Allocate an integer token allowance to every chunk, with no dropped chunks.

    Proportional allocation keeps the same lecture-level cap across segmenters.
    It cannot guarantee equal actual summary lengths: generation may stop early.
    """
    if not source_sizes or any(size <= 0 for size in source_sizes):
        raise ValueError("Output allocation needs nonempty, positive source sizes")
    count = len(source_sizes)
    if per_chunk_limit < minimum or not count * minimum <= budget <= count * per_chunk_limit:
        raise ValueError(
            f"Lecture output budget {budget} cannot fit {count} chunks: require "
            f"{count * minimum}..{count * per_chunk_limit} tokens. Adjust the budget, "
            "segmentation granularity, or per-chunk output limit on development data."
        )
    targets = [budget * size / sum(source_sizes) for size in source_sizes]
    allowances = [minimum] * count
    for _ in range(budget - count * minimum):
        available = [i for i in range(count) if allowances[i] < per_chunk_limit]
        winner = max(available, key=lambda i: (targets[i] - allowances[i], -i))
        allowances[winner] += 1
    return allowances
