"""Semantic/fixed segmentation, MMR evidence selection, and SEDL assembly."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np

from . import __version__
from .backends import Embedder, Summarizer, normalize
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

    def validate(self, summarizer: Summarizer):
        if self.chunking not in {"semantic", "fixed"}:
            raise ValueError("chunking must be semantic or fixed")
        if self.evidence not in {"mmr", "raw"}:
            raise ValueError("evidence must be mmr or raw")
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


def chunk_units(units, scores, config, count_tokens):
    chunks = []
    current = []
    start_reason = {"reason": "lecture_start", "score": None}
    for unit in units:
        score = scores.get(unit.segment_index)
        new_segment = bool(current and unit.segment_index != current[-1].segment_index)
        forced = bool(current and count_tokens(joined(current + [unit])) > config.max_chunk_tokens)
        semantic = (config.chunking == "semantic" and new_segment and
                    score is not None and score >= config.boundary_threshold and
                    count_tokens(joined(current)) >= config.min_chunk_tokens)
        if forced or semantic:
            reason = {"reason": "token_limit" if forced else "semantic_change",
                      "score": score if new_segment else None,
                      "at_segment_id": unit.segment_id,
                      "at_char": unit.char_start}
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


_STOP = set("a an and are as at be been but by can for from has have how i if in into is it its "
            "of on or our so that the their these they this those to was we were what when which "
            "will with you your not than then also each using use through".split())


def keywords(text, limit=6):
    counts = Counter(word for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
                     if word not in _STOP)
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def run(transcript: Transcript, embedder: Embedder, summarizer: Summarizer,
        config: Config = Config()) -> dict:
    config.validate(summarizer)
    # Atomic units fit BOTH encoder windows and MMR's evidence budget. The same
    # settings produce identical units for mmr/raw, keeping the ablation comparable.
    units = make_units(transcript, [
        (embedder.count_tokens, embedder.input_limit),
        (summarizer.count_tokens, min(config.max_chunk_tokens, config.evidence_tokens)),
    ])
    vectors = normalize(embedder.encode([unit.text for unit in units]))
    if (vectors.ndim != 2 or vectors.shape[0] != len(units) or
            not np.isfinite(vectors).all()):
        raise ValueError("Embedder returned invalid vectors")
    scores = segment_boundary_scores(units, vectors, len(transcript.segments), config.boundary_window)
    chunks = chunk_units(units, scores, config, summarizer.count_tokens)
    by_id = {unit.id: i for i, unit in enumerate(units)}
    entries = []
    for i, (chunk, start_boundary, end_boundary) in enumerate(chunks):
        chunk_vectors = vectors[[by_id[unit.id] for unit in chunk]]
        if config.evidence == "mmr":
            selection = select_mmr(chunk, chunk_vectors, summarizer.count_tokens,
                                   config.evidence_tokens, config.max_evidence_units, config.mmr_lambda)
        else:
            selection = [(j, {"selection_rank": j + 1, "relevance": None,
                              "redundancy": None, "mmr_score": None}) for j in range(len(chunk))]
        evidence_text = joined([chunk[j] for j, _ in selection])
        summary = summarizer.summarize(evidence_text)
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
            "summary": summary, "evidence": evidence,
            "keywords": keywords(evidence_text),
            "boundary": {"start": start_boundary, "end": end_boundary},
        })
    source = [asdict(s) for s in transcript.segments]
    digest = hashlib.sha256(json.dumps({"text": transcript.text, "segments": source},
                                       ensure_ascii=False, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    demo = embedder.name.startswith("demo/") or summarizer.name.startswith("demo/")
    return {
        "schema_version": "1.0", "implementation_version": __version__,
        "transcript_id": transcript.id, "source_sha256": digest,
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
        "source_units": [asdict(unit) for unit in units],
        "boundary_candidates": [{"before_segment_id": transcript.segments[i].id,
                                 "time": transcript.segments[i].start, "score": score}
                                for i, score in scores.items()],
        "entries": entries,
    }
