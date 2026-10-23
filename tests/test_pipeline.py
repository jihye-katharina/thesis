import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from sedl.backends import ExtractiveDemo, HashEmbedder
from sedl.data import Transcript, make_units
from sedl.evaluation import boundary_f1, evaluate_boundaries
from sedl.pipeline import Config, run, select_mmr
from sedl.references import evidence_unit_ref, index_source_units, iter_source_units, reference_key


def transcript(texts):
    return Transcript.from_dict({"id": "test", "segments": [
        {"start": i * 10, "end": (i + 1) * 10, "text": text}
        for i, text in enumerate(texts)]})


class TopicEmbedder(HashEmbedder):
    """Known geometry to test actual boundary decisions, independent of a model."""

    def encode(self, texts):
        return np.array([[1, 0] if "apple" in t else [0, 1] for t in texts], dtype=float)


def test_semantic_boundary_and_raw_ablation():
    source = transcript(["apple grows.", "apple harvest.", "ocean waves.", "ocean tides."])
    config = Config(min_chunk_tokens=1, max_chunk_tokens=50, evidence_tokens=50,
                    boundary_window=1, boundary_threshold=0.5)
    semantic = run(source, TopicEmbedder(), ExtractiveDemo(), config)
    assert [e["start"] for e in semantic["entries"]] == [0, 20]
    assert semantic["entries"][0]["boundary"]["end"]["reason"] == "semantic_change"
    assert semantic["entries"][0]["boundary"]["end"]["score"] == pytest.approx(1.0)
    raw = run(source, TopicEmbedder(), ExtractiveDemo(), replace(config, evidence="raw"))
    assert [e["source_unit_refs"] for e in semantic["entries"]] == [e["source_unit_refs"] for e in raw["entries"]]
    assert all(e["source_unit_refs"] == [evidence_unit_ref(u) for u in e["evidence"]] for e in raw["entries"])
    fixed = run(source, TopicEmbedder(), ExtractiveDemo(), replace(config, chunking="fixed"))
    assert len(fixed["entries"]) == 1


@pytest.mark.parametrize("chunking", ["semantic", "fixed", "sadc"])
@pytest.mark.parametrize("evidence", ["mmr", "raw", "lead"])
def test_long_segments_pack_whole_sentences_without_losing_source(chunking, evidence):
    original = " ".join(f"Sentence{i} has words." for i in range(70))
    source = transcript([original, "Another subject. And another sentence."])
    generator = ExtractiveDemo()
    config = Config(chunking=chunking, evidence=evidence, min_chunk_tokens=5,
                    max_chunk_tokens=30, evidence_tokens=12)
    result = run(source, HashEmbedder(), generator, config)
    all_refs = [ref for entry in result["entries"] for ref in entry["source_unit_refs"]]
    assert all_refs == [ref for ref, _, _ in iter_source_units(result)]
    assert len(all_refs) == len({reference_key(ref) for ref in all_refs})
    covered = set()
    for ref, segment, unit in iter_source_units(result):
        assert segment["text"][unit["char_start"]:unit["char_end"]] == unit["text"]
        assert "start" not in unit and "end" not in unit
        if ref["segment_id"] == source.segments[0].id:
            covered.update(range(unit["char_start"], unit["char_end"]))
    assert all(i in covered for i, char in enumerate(original) if not char.isspace())
    for entry in result["entries"]:
        assert entry["source_tokens"] <= 30
        assert entry["generation_input_tokens"] <= (30 if evidence == "raw" else 12)
        lookup = index_source_units(result)
        assert entry["summary"] == "\n".join(lookup[reference_key(evidence_unit_ref(e))][1]["text"]
                                             for e in entry["evidence"])


def test_mmr_reduces_duplicate_evidence_and_retains_selection_order():
    source = transcript(["First statement. Duplicate statement. Complementary evidence."])
    units = make_units(source)
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    chosen = select_mmr(units, vectors, ExtractiveDemo().count_tokens, 50, 2, 0.5)
    assert [index for index, _ in chosen] == [0, 2]
    assert [details["selection_rank"] for _, details in chosen] == [1, 2]


def test_mmr_budget_rechecks_joined_text_and_skips_nonfitting_units():
    source = transcript(["A long sentence has many words. Short. Small."])
    generator = ExtractiveDemo()
    units = make_units(source)
    chosen = select_mmr(units, np.eye(3), generator.count_tokens, 4, 3, 0.7)
    assert [i for i, _ in chosen] == [1, 2]


def test_zero_vectors_stay_finite():
    class EmptyEmbedder(HashEmbedder):
        def encode(self, texts):
            return np.zeros((len(texts), 4))
    result = run(transcript(["One sentence.", "Second sentence."]), EmptyEmbedder(),
                 ExtractiveDemo(), Config(min_chunk_tokens=1))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"segments": []},
    {"segments": [{"start": 0, "end": float("nan"), "text": "test"}]},
    {"segments": [{"start": True, "end": 3, "text": "test"}]},
    {"segments": [{"start": 1, "end": 0, "text": "test"}]},
    {"segments": [{"start": 0, "end": 1, "text": "  "}]},
    {"text": ["wrong"], "segments": [{"start": 0, "end": 1, "text": "test"}]},
])
def test_bad_input_rejected(change):
    with pytest.raises(ValueError):
        Transcript.from_dict(change)


def test_out_of_order_and_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="ordered"):
        Transcript.from_dict({"segments": [{"start": 1, "end": 2, "text": "a"},
                                            {"start": 0, "end": 1, "text": "b"}]})
    with pytest.raises(ValueError, match="Duplicate"):
        Transcript.from_dict({"segments": [{"id": "x", "start": 0, "end": 1, "text": "a"},
                                            {"id": "x", "start": 1, "end": 2, "text": "b"}]})


def test_full_text_preserved_and_input_identity_changes():
    source = transcript(["Hello lecture."])
    first = run(source, HashEmbedder(), ExtractiveDemo())
    second = run(replace(source, text="Hello lecture."), HashEmbedder(), ExtractiveDemo())
    assert second["source_full_text"] == "Hello lecture."
    assert first["source_sha256"] != second["source_sha256"]


def test_overlapping_timestamps_remain_source_aligned():
    source = Transcript.from_dict({"segments": [{"start": 0, "end": 30, "text": "First."},
                                                {"start": 10, "end": 20, "text": "Second."}]})
    result = run(source, HashEmbedder(), ExtractiveDemo(), Config(min_chunk_tokens=1,
                                                               boundary_threshold=2))
    assert result["entries"][0]["end"] == 30


@pytest.mark.parametrize("changes", [
    {"min_chunk_tokens": 0}, {"max_chunk_tokens": 2048}, {"evidence_tokens": 0},
    {"mmr_lambda": float("nan")}, {"boundary_threshold": -1}, {"boundary_window": 0},
])
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        run(transcript(["Hello."]), HashEmbedder(), ExtractiveDemo(), Config(**changes))


def test_boundary_one_to_one_matching_and_empty_conventions():
    result = boundary_f1([10, 11, 30], [10, 30], 2)
    assert result["matched"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1
    assert boundary_f1([], [], 5)["f1"] == 1
    assert boundary_f1([], [10], 5)["f1"] == 0
    assert boundary_f1([10], [], 5)["f1"] == 0
    assert boundary_f1([11, 20], [10, 12], 9)["matched"] == 2
    with pytest.raises(ValueError):
        boundary_f1([10, 10], [10], 1)
    with pytest.raises(ValueError):
        boundary_f1([float("inf")], [], 1)


def test_annotation_identity_checked():
    result = run(transcript(["Hello."]), HashEmbedder(), ExtractiveDemo())
    with pytest.raises(ValueError, match="transcript_id"):
        evaluate_boundaries(result, {"transcript_id": "another", "boundaries": []})


def test_example_cli_round_trip(tmp_path, capsys):
    from sedl.cli import main

    root = Path(__file__).resolve().parents[1]
    result = tmp_path / "result.json"
    main(["summarize", str(root / "examples/lecture.json"), "-o", str(result),
          "--backend", "demo", "--min-chunk-tokens", "20", "--boundary-window", "1"])
    sedl = json.loads(result.read_text())
    assert sedl["mode"] == "offline_demo"
    assert sedl["models"]["summarizer"] == "demo/extractive-echo"
    assert sedl["entries"]
    evaluation = tmp_path / "evaluation.json"
    main(["evaluate-boundaries", str(result), str(root / "examples/boundaries.json"),
          "-o", str(evaluation)])
    assert "f1" in json.loads(evaluation.read_text())["boundary_f1"]
    exported = tmp_path / "judges.jsonl"
    main(["export-judges", str(result), "-o", str(exported)])
    records = [json.loads(line) for line in exported.read_text().splitlines()]
    assert len(records) == len(sedl["entries"])
    assert records[0]["response"] == sedl["entries"][0]["summary"]
    lookup = index_source_units(sedl)
    assert records[0]["retrieved_contexts"] == [
        lookup[reference_key(evidence_unit_ref(e))][1]["text"] for e in sedl["entries"][0]["evidence"]]
