"""Sentence-window chunking, MMR evidence, and source-timestamped SEDL export.

The default strategy embeds the flat chronological unit list once, finds topic
boundaries using neighbouring unit windows, and reuses those vectors for MMR.
Original time segments provide source references and playback intervals.
"""

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
from .references import unit_ref


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


def unit_boundary_scores(vectors, window):
    """Compare left/right windows of units across the whole lecture.

    Original time-segment boundaries do not restrict candidate topic boundaries.
    Each unit receives equal weight within its window.
    """
    scores = {}
    for i in range(1, len(vectors)):
        left = normalize(vectors[max(0, i - window):i].mean(axis=0))
        right = normalize(vectors[i:min(len(vectors), i + window)].mean(axis=0))
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


def chunk_units(locations, scores, config, count_tokens, sadc_threshold=None):
    """Group ordered (parent segment, unit) locations by temporary list indices."""
    chunks = []
    current = []
    start_reason = {"reason": "lecture_start", "score": None}
    for unit_index, (segment, unit) in enumerate(locations):
        score = scores.get(unit_index)
        current_units = [locations[j][1] for j in current]
        forced = bool(current and count_tokens(joined(current_units + [unit])) > config.max_chunk_tokens)
        if config.chunking == "sadc":
            # Strict comparison matches sim_i < percentile(sim, p). The paper
            # has no minimum-length rule, so min_chunk_tokens is not applied.
            semantic = (bool(current) and score is not None and sadc_threshold is not None
                        and score > 1 - sadc_threshold)
        else:
            semantic = (config.chunking == "semantic" and bool(current) and
                        score is not None and score >= config.boundary_threshold and
                        count_tokens(joined(current_units)) >= config.min_chunk_tokens)
        if forced or semantic:
            reason = {"reason": "token_limit" if forced else "semantic_change",
                      "score": score,
                      "at_char": unit.char_start,
                      "before_unit_ref": unit_ref(segment.id, unit.id),
                      "before_unit_index": unit_index}
            chunks.append((current, start_reason, reason))
            current = []
            start_reason = reason
        current.append(unit_index)
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
    segments_by_id = {segment.id: segment for segment in transcript.segments}
    locations = [(segments_by_id[unit.segment_id], unit) for unit in units]
    unitization_finished = perf_counter()
    vectors = np.asarray(embedder.encode([unit.text for unit in units]), dtype=float)
    if (vectors.ndim != 2 or vectors.shape[0] != len(units) or
            vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
        raise ValueError("Embedder returned invalid vectors")
    vectors = normalize(vectors)
    embedding_finished = perf_counter()
    if config.chunking == "sadc":
        scores, sadc_threshold = sadc_boundary_scores(vectors, config.sadc_percentile)
    else:
        scores = unit_boundary_scores(vectors, config.boundary_window)
        sadc_threshold = None
    chunks = chunk_units(locations, scores, config, summarizer.count_tokens, sadc_threshold)
    source_sizes = [summarizer.count_tokens(joined([units[j] for j in indices]))
                    for indices, _, _ in chunks]
    allocations = (allocate_summary_budget(source_sizes, config.summary_budget_tokens,
                                            summarizer.max_new_tokens)
                   if config.summary_budget_tokens is not None else [None] * len(chunks))
    segmentation_finished = perf_counter()
    entries = []
    selection_seconds = generation_seconds = 0.0
    for i, (indices, start_boundary, end_boundary) in enumerate(chunks):
        selection_started = perf_counter()
        chunk = [units[j] for j in indices]
        parents = [locations[j][0] for j in indices]
        chunk_vectors = vectors[indices]
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
        references = [unit_ref(parent.id, unit.id) for parent, unit in zip(parents, chunk)]
        # Export source timing for EACH selected passage, even when its chunk
        # spans several time segments. Never invent finer sentence timestamps.
        evidence = [{**chunk[j].to_evidence(parents[j]), **details} for j, details in selection]
        entries.append({
            "id": f"chunk-{i:04d}", "start": min(s.start for s in parents),
            "end": max(s.end for s in parents),
            "source_unit_refs": references,
            "source_segment_ids": list(dict.fromkeys(s.id for s in parents)),
            "source_tokens": summarizer.count_tokens(joined(chunk)),
            "generation_input_tokens": summarizer.count_tokens(evidence_text),
            "summary_tokens": summarizer.count_tokens(summary),
            "allocated_output_tokens": allocations[i],
            "summary": summary, "evidence": evidence,
            "keywords": keywords(evidence_text),
            "boundary": {"start": start_boundary, "end": end_boundary},
        })
    source = [asdict(segment) for segment in transcript.segments]
    digest = hashlib.sha256(json.dumps({"text": transcript.text, "segments": source},
                                       ensure_ascii=False, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    demo = embedder.name.startswith("demo/") or summarizer.name.startswith("demo/")
    unit_records = [asdict(unit) for unit in units]
    unitization_hash = hashlib.sha256(json.dumps({"segments": source, "units": unit_records}, sort_keys=True,
                                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    total_summary_tokens = sum(entry["summary_tokens"] for entry in entries)
    if config.summary_budget_tokens is not None and total_summary_tokens > config.summary_budget_tokens:
        raise ValueError("Decoded summaries exceeded the lecture output token budget")
    return {
        "schema_version": "3.0", "implementation_version": __version__,
        "transcript_id": transcript.id, "source_sha256": digest,
        "unitization_sha256": unitization_hash,
        "mode": "offline_demo" if demo else "neural",
        "config": asdict(config),
        "models": {"embedding": embedder.name, "embedding_revision": embedder.revision,
                   "summarizer": summarizer.name, "summarizer_revision": summarizer.revision},
        "provenance_note": "Units form one chronological list with lecture-wide IDs and source segment_id. "
                           "Each evidence passage includes start/end and jump_time from its original "
                           "segment; timestamp_precision is source_segment, not sentence alignment. "
                           "Chunk-level evidence does not prove individual summary claims.",
        "strategy": {
            "unitization": "sentence_like_with_token_limit_splitting",
            "unit_embedding_passes": 1,
            "mmr_reuses_unit_embeddings": config.evidence == "mmr",
            "evidence_scope": "summary_chunk",
            "timestamp_precision": "source_segment",
            "jump_time_basis": "original_segment_start_seconds",
        },
        "source_segments": source,
        "source_units": unit_records,
        "source_full_text": transcript.text,
        "source_metadata": {
            "source": transcript.source,
            "language": transcript.language,
            "provenance": deepcopy(transcript.provenance),
        },
        "segmentation": {
            "method": config.chunking,
            "score_type": "cosine_distance",
            "semantic_window_basis": "units" if config.chunking == "semantic" else None,
            "boundary_window_units": config.boundary_window if config.chunking == "semantic" else None,
            "sadc_similarity_threshold": sadc_threshold,
            "sadc_percentile": config.sadc_percentile if config.chunking == "sadc" else None,
            "paper_reproduction": False,
        },
        "boundary_candidates": [{"before_unit_ref": unit_ref(locations[i][0].id, locations[i][1].id),
                                 "before_unit_index": i,
                                 "char_start": locations[i][1].char_start,
                                 "time": locations[i][0].start, "score": score}
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
