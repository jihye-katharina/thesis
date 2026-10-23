"""Semantic/fixed segmentation, MMR evidence selection, and SEDL assembly."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from time import perf_counter

import numpy as np

from . import __version__
from .backends import Embedder, Summarizer, normalize
from .budgets import allocate_summary_budget
from .data import Transcript, Unit, make_units


@dataclass(frozen=True)
class Config:
    chunking: str = "semantic"
    evidence: str = "mmr"
    min_chunk_tokens: int = 80
    max_chunk_tokens: int = 900
    evidence_tokens: int = 512
    max_evidence_units: int = 8
    boundary_threshold: float = 0.35
    boundary_window: int = 2
    mmr_lambda: float = 0.7
    sadc_percentile: float = 10.0
    summary_budget_tokens: int | None = None

    def validate(self, summarizer: Summarizer):
        if self.chunking not in {"semantic", "fixed", "sadc"}:
            raise ValueError("chunking must be semantic, fixed, or sadc")
        if self.evidence not in {"mmr", "raw", "lead"}:
            raise ValueError("evidence must be mmr, raw, or lead")
        if not 1 <= self.min_chunk_tokens <= self.max_chunk_tokens <= summarizer.input_limit:
            raise ValueError("Require 1 <= min_chunk_tokens <= max_chunk_tokens <= BART input capacity")
        if not 1 <= self.evidence_tokens <= summarizer.input_limit:
            raise ValueError("evidence_tokens must fit the summarizer")
        if self.max_evidence_units < 1 or self.boundary_window < 1:
            raise ValueError("Evidence count and boundary window must be positive")
        if not math.isfinite(self.boundary_threshold) or not 0 <= self.boundary_threshold <= 2:
            raise ValueError("boundary_threshold must be in [0, 2]")
        if not math.isfinite(self.mmr_lambda) or not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if not math.isfinite(self.sadc_percentile) or not 0 <= self.sadc_percentile <= 100:
            raise ValueError("sadc_percentile must be in [0, 100]")
        if self.summary_budget_tokens is not None and (
                isinstance(self.summary_budget_tokens, bool) or
                not isinstance(self.summary_budget_tokens, int) or self.summary_budget_tokens < 1):
            raise ValueError("summary_budget_tokens must be a positive integer or null")


def joined(units: list[Unit]) -> str:
    return "\n".join(unit.text for unit in units)


def segment_boundary_scores(units, vectors, n_segments, window):
    """Cosine distance between local left/right windows of segment embeddings.

    Within a segment, use character-weighted sentence/unit means. Each segment
    has equal weight in the temporal window, irrespective of sentence count.
    """
    means = []
    for i in range(n_segments):
        indices = [j for j, unit in enumerate(units) if unit.segment_index == i]
        weights = [len(units[j].text) for j in indices]
        means.append(np.average(vectors[indices], axis=0, weights=weights))
    means = normalize(np.asarray(means))
    scores = {}
    for i in range(1, n_segments):
        left = normalize(means[max(0, i - window):i].mean(axis=0))
        right = normalize(means[i:min(n_segments, i + window)].mean(axis=0))
        scores[i] = float(np.clip(1 - np.dot(left, right), 0, 2))
    return scores


def sadc_boundary_scores(vectors, percentile):
    """SAHR Section 3.1 adaptation: adjacent UNIT similarity, document percentile.

    Units may be token-split sentences; the common encoder need not be BGE-M3.
    This is a controlled adaptation, not an exact reproduction of the paper.
    """
    if len(vectors) < 2:
        return {}, None
    similarities = np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1, 1)
    threshold = float(np.percentile(similarities, percentile))
    return {i + 1: float(1 - similarity) for i, similarity in enumerate(similarities)}, threshold


def chunk_units(units, scores, config, count_tokens, sadc_threshold=None):
    chunks = []
    current = []
    start_reason = {"reason": "lecture_start", "score": None}
    for unit_index, unit in enumerate(units):
        score = scores.get(unit_index if config.chunking == "sadc" else unit.segment_index)
        new_segment = bool(current and unit.segment_index != current[-1].segment_index)
        forced = bool(current and count_tokens(joined(current + [unit])) > config.max_chunk_tokens)
        if config.chunking == "sadc":
            # Strict comparison matches sim_i < percentile(sim, p). The paper
            # has no minimum-length rule, so min_chunk_tokens is not applied.
            semantic = (bool(current) and score is not None and sadc_threshold is not None
                        and score > 1 - sadc_threshold)
        else:
            semantic = (config.chunking == "semantic" and new_segment and
                        score is not None and score >= config.boundary_threshold and
                        count_tokens(joined(current)) >= config.min_chunk_tokens)
        if forced or semantic:
            reason = {"reason": "token_limit" if forced else "semantic_change",
                      "score": score if new_segment or config.chunking == "sadc" else None,
                      "at_segment_id": unit.segment_id,
                      "at_char": unit.char_start,
                      "before_unit_id": unit.id,
                      "before_unit_index": unit_index}
            chunks.append((current, start_reason, reason))
            current = []
            start_reason = reason
        current.append(unit)
    if current:
        chunks.append((current, start_reason, {"reason": "lecture_end", "score": None}))
    return chunks


def select_mmr(units, vectors, count_tokens, budget, limit, weight):
    """Greedy MMR; serialize selected evidence chronologically, rank separately."""
    vectors = normalize(vectors)
    centroid = normalize(np.average(vectors, axis=0, weights=[len(u.text) for u in units]))
    relevance = vectors @ centroid
    selected = []
    details = {}
    while len(selected) < min(limit, len(units)):
        candidates = []
        for i in range(len(units)):
            if i in details:
                continue
            ordered = sorted(selected + [i])
            if count_tokens(joined([units[j] for j in ordered])) > budget:
                continue
            redundancy = float(np.max(vectors[selected] @ vectors[i])) if selected else 0.0
            score = weight * float(relevance[i]) - (1 - weight) * redundancy
            candidates.append((score, -i, i, redundancy))
        if not candidates:
            break
        score, _, i, redundancy = max(candidates)
        selected.append(i)
        details[i] = {"selection_rank": len(selected), "relevance": float(relevance[i]),
                      "redundancy": redundancy, "mmr_score": score}
    if not selected:
        raise ValueError("Evidence budget cannot fit any unit")
    return [(i, details[i]) for i in sorted(selected)]


def select_lead(units, count_tokens, budget, limit):
    """Chronological evidence control with MMR's same token and unit caps.

    Stop at the first non-fitting unit rather than cherry-picking shorter later
    units. Units themselves fit the evidence budget due to shared unitization.
    """
    selected = []
    for i in range(min(len(units), limit)):
        if count_tokens(joined(units[:i + 1])) > budget:
            break
        selected.append((i, {"selection_rank": i + 1, "relevance": None,
                             "redundancy": None, "mmr_score": None}))
    if not selected:
        raise ValueError("Evidence budget cannot fit any unit")
    return selected


_STOP = set("a an and are as at be been but by can for from has have how i if in into is it its "
            "of on or our so that the their these they this those to was we were what when which "
            "will with you your not than then also each using use through".split())


def keywords(text, limit=6):
    counts = Counter(word for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
                     if word not in _STOP)
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def run(transcript: Transcript, embedder: Embedder, summarizer: Summarizer,
        config: Config = Config()) -> dict:
    started = perf_counter()
    config.validate(summarizer)
    # Atomic units fit BOTH encoder windows and MMR's evidence budget. The same
    # settings produce identical units for mmr/raw, keeping the ablation comparable.
    units = make_units(transcript, [
        (embedder.count_tokens, embedder.input_limit),
        (summarizer.count_tokens, min(config.max_chunk_tokens, config.evidence_tokens)),
    ])
    unitization_finished = perf_counter()
    vectors = np.asarray(embedder.encode([unit.text for unit in units]), dtype=float)
    if (vectors.ndim != 2 or vectors.shape[0] != len(units) or
            vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
        raise ValueError("Embedder returned invalid vectors")
    vectors = normalize(vectors)
    embedding_finished = perf_counter()
    if config.chunking == "sadc":
        scores, sadc_threshold = sadc_boundary_scores(vectors, config.sadc_percentile)
        candidate_units = {i: units[i] for i in scores}
    else:
        scores = segment_boundary_scores(units, vectors, len(transcript.segments), config.boundary_window)
        sadc_threshold = None
        first_by_segment = {}
        for unit in units:
            first_by_segment.setdefault(unit.segment_index, unit)
        candidate_units = {i: first_by_segment[i] for i in scores}
    chunks = chunk_units(units, scores, config, summarizer.count_tokens, sadc_threshold)
    source_sizes = [summarizer.count_tokens(joined(chunk)) for chunk, _, _ in chunks]
    allocations = (allocate_summary_budget(source_sizes, config.summary_budget_tokens,
                                            summarizer.max_new_tokens)
                   if config.summary_budget_tokens is not None else [None] * len(chunks))
    segmentation_finished = perf_counter()
    by_id = {unit.id: i for i, unit in enumerate(units)}
    entries = []
    selection_seconds = generation_seconds = 0.0
    for i, (chunk, start_boundary, end_boundary) in enumerate(chunks):
        selection_started = perf_counter()
        chunk_vectors = vectors[[by_id[unit.id] for unit in chunk]]
        if config.evidence == "mmr":
            selection = select_mmr(chunk, chunk_vectors, summarizer.count_tokens,
                                   config.evidence_tokens, config.max_evidence_units, config.mmr_lambda)
        elif config.evidence == "lead":
            selection = select_lead(chunk, summarizer.count_tokens,
                                    config.evidence_tokens, config.max_evidence_units)
        else:
            selection = [(j, {"selection_rank": j + 1, "relevance": None,
                              "redundancy": None, "mmr_score": None}) for j in range(len(chunk))]
        evidence_text = joined([chunk[j] for j, _ in selection])
        selection_seconds += perf_counter() - selection_started
        generation_started = perf_counter()
        summary = (summarizer.summarize(evidence_text, max_new_tokens=allocations[i])
                   if allocations[i] is not None else summarizer.summarize(evidence_text))
        generation_seconds += perf_counter() - generation_started
        if not summary.strip():
            raise ValueError(f"Summarizer produced an empty summary for chunk {i}")
        evidence = [{**asdict(chunk[j]), **details} for j, details in selection]
        entries.append({
            "id": f"chunk-{i:04d}", "start": min(u.start for u in chunk),
            "end": max(u.end for u in chunk),
            "source_unit_ids": [u.id for u in chunk],
            "source_segment_ids": list(dict.fromkeys(u.segment_id for u in chunk)),
            "source_tokens": summarizer.count_tokens(joined(chunk)),
            "generation_input_tokens": summarizer.count_tokens(evidence_text),
            "summary_tokens": summarizer.count_tokens(summary),
            "allocated_output_tokens": allocations[i],
            "summary": summary, "evidence": evidence,
            "keywords": keywords(evidence_text),
            "boundary": {"start": start_boundary, "end": end_boundary},
        })
    source = [asdict(s) for s in transcript.segments]
    digest = hashlib.sha256(json.dumps({"text": transcript.text, "segments": source},
                                       ensure_ascii=False, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    demo = embedder.name.startswith("demo/") or summarizer.name.startswith("demo/")
    unit_records = [asdict(unit) for unit in units]
    unitization_hash = hashlib.sha256(json.dumps(unit_records, sort_keys=True,
                                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    total_summary_tokens = sum(entry["summary_tokens"] for entry in entries)
    if config.summary_budget_tokens is not None and total_summary_tokens > config.summary_budget_tokens:
        raise ValueError("Decoded summaries exceeded the lecture output token budget")
    return {
        "schema_version": "1.1", "implementation_version": __version__,
        "transcript_id": transcript.id, "source_sha256": digest,
        "unitization_sha256": unitization_hash,
        "mode": "offline_demo" if demo else "neural",
        "config": asdict(config),
        "models": {"embedding": embedder.name, "embedding_revision": embedder.revision,
                   "summarizer": summarizer.name, "summarizer_revision": summarizer.revision},
        "provenance_note": "Evidence times are source segment intervals, not sentence alignments. "
                           "Chunk-level evidence does not prove individual summary claims.",
        "source_segments": source,
        "source_full_text": transcript.text,
        "source_metadata": {
            "source": transcript.source,
            "language": transcript.language,
            "provenance": deepcopy(transcript.provenance),
        },
        "source_units": unit_records,
        "segmentation": {
            "method": config.chunking,
            "score_type": "cosine_distance",
            "sadc_similarity_threshold": sadc_threshold,
            "sadc_percentile": config.sadc_percentile if config.chunking == "sadc" else None,
            "paper_reproduction": False,
        },
        "boundary_candidates": [{"before_segment_id": candidate_units[i].segment_id,
                                 "before_unit_id": candidate_units[i].id,
                                 "before_unit_index": by_id[candidate_units[i].id],
                                 "char_start": candidate_units[i].char_start,
                                 "time": candidate_units[i].start, "score": score}
                                for i, score in scores.items()],
        "statistics": {
            "chunks": len(chunks), "source_units": len(units),
            "source_tokens": sum(source_sizes),
            "generation_input_tokens": sum(e["generation_input_tokens"] for e in entries),
            "summary_tokens": total_summary_tokens,
            "requested_output_budget": config.summary_budget_tokens,
            "allocated_output_budget": sum(allocations) if config.summary_budget_tokens is not None else None,
        },
        "timings_seconds": {
            "unitization": unitization_finished - started,
            "embedding": embedding_finished - unitization_finished,
            "segmentation": segmentation_finished - embedding_finished,
            "selection": selection_seconds, "generation": generation_seconds,
            "total": perf_counter() - started,
            "note": "Wall-clock stage timings; use runtime metrics for CUDA-synchronised total.",
        },
        "entries": entries,
    }
