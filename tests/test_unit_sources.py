"""Flat lecture-wide units, independent topic boundaries, and source tracing."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from sedl.audit import export_audit
from sedl.backends import ExtractiveDemo, HashEmbedder
from sedl.cli import main
from sedl.data import Transcript, make_units
from sedl.evaluation import boundary_annotation_template, evaluate_boundaries, judge_records
from sedl.pipeline import Config, run
from sedl.references import index_source_units, reference_key


def lecture():
    return Transcript.from_dict({"video_id": "lecture-A", "segments": [
        {"id": "source/alpha", "start": 0, "end": 30,
         "text": "Alpha one. Alpha two. Alpha three."},
        {"id": "source:beta", "start": 30, "end": 60,
         "text": "Beta one. Beta two. Beta three."}]})


class KnownVectors(HashEmbedder):
    def encode(self, texts):
        return np.array([[0., 1.] if t in {"Alpha three.", "Beta one."} else [1., 0.]
                         for t in texts])


def output(evidence="mmr"):
    return run(lecture(), KnownVectors(), ExtractiveDemo(),
               Config(chunking="fixed", evidence=evidence, min_chunk_tokens=1,
                      max_chunk_tokens=9, evidence_tokens=9, max_evidence_units=1))


def test_make_units_returns_a_flat_lecture_list_with_segment_references():
    source = lecture()
    units = make_units(source, [(ExtractiveDemo().count_tokens, 20)])
    assert [u.id for u in units] == ["u0", "u1", "u2", "u3", "u4", "u5"]
    assert [u.text for u in units] == ["Alpha one.", "Alpha two.", "Alpha three.",
                                     "Beta one.", "Beta two.", "Beta three."]
    assert [u.segment_id for u in units] == ["source/alpha"] * 3 + ["source:beta"] * 3
    segments = {s.id: s for s in source.segments}
    for unit in units:
        assert segments[unit.segment_id].text[unit.char_start:unit.char_end] == unit.text
        assert not any(hasattr(unit, field) for field in ("segment_index", "start", "end"))
    with pytest.raises(TypeError, match="expects a Transcript"):
        make_units(source.segments[0], [(len, 20)])


def test_unit_numbering_continues_across_segments_and_restarts_for_another_lecture():
    source = lecture()
    units = make_units(source, [(ExtractiveDemo().count_tokens, 2)])
    assert [u.id for u in units] == [f"u{i}" for i in range(len(units))]
    first_beta = next(u for u in units if u.segment_id == "source:beta")
    assert first_beta.id != "u0"
    assert make_units(source, [(ExtractiveDemo().count_tokens, 2)])[0].id == "u0"
    assert not hasattr(source, "with_units")
    assert not hasattr(source.segments[0], "units")


def test_making_units_is_repeatable_and_does_not_mutate_the_transcript():
    source = lecture()
    snapshot = asdict(source)
    first_units = make_units(source, [(ExtractiveDemo().count_tokens, 2)])
    assert make_units(source, [(ExtractiveDemo().count_tokens, 2)]) == first_units
    config = Config(chunking="fixed", min_chunk_tokens=1, max_chunk_tokens=9)
    original = run(source, KnownVectors(), ExtractiveDemo(), config)
    repeated = run(source, KnownVectors(), ExtractiveDemo(), config)
    assert repeated["source_sha256"] == original["source_sha256"]
    assert repeated["unitization_sha256"] == original["unitization_sha256"]
    assert repeated["entries"] == original["entries"]
    assert repeated["source_segments"] == original["source_segments"]
    assert asdict(source) == snapshot


def test_mmr_keeps_correct_embedding_rows_across_source_segments():
    result = output()
    assert result["schema_version"] == "3.0"
    assert len(result["source_units"]) == 6
    assert all("units" not in s for s in result["source_segments"])
    assert [entry["summary"] for entry in result["entries"]] == ["Alpha one.", "Beta two."]
    assert result["entries"][0]["evidence"][0]["segment_id"] == "source/alpha"
    assert result["entries"][0]["evidence"][0]["unit_id"] == "u0"
    assert result["entries"][1]["evidence"][0]["segment_id"] == "source:beta"
    assert result["entries"][1]["evidence"][0]["unit_id"] == "u4"
    assert [(e["start"], e["end"]) for e in result["entries"]] == [(0, 30), (30, 60)]
    records = list(judge_records(result))
    assert records[0]["retrieved_contexts"] == ["Alpha one."]
    assert records[1]["retrieved_contexts"] == ["Beta two."]
    assert records[0]["source_chunk"] == "Alpha one.\nAlpha two.\nAlpha three."
    assert records[1]["source_chunk"] == "Beta one.\nBeta two.\nBeta three."


def test_flat_unit_source_is_resolved_correctly_in_audit(tmp_path):
    result = output("lead")
    assert [e["summary"] for e in result["entries"]] == ["Alpha one.", "Beta one."]
    path = tmp_path / "audit.csv"
    export_audit(result, path)
    import csv
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "Alpha one." in rows[0]["selected_evidence"]
    assert "Beta one." not in rows[0]["selected_evidence"]
    assert "Beta one." in rows[1]["selected_evidence"]
    assert '"segment_id": "source:beta", "unit_id": "u3"' in rows[1]["selected_evidence"]


def test_boundaries_reject_unknown_or_duplicate_source_references():
    result = output()
    annotation = boundary_annotation_template(result)
    assert "boundary_unit_ids" not in annotation
    assert annotation["source_units"][3]["id"] == "u3"
    assert annotation["source_units"][3]["segment_id"] == "source:beta"
    annotation.update(annotation_status="complete", boundary_unit_refs=[
        {"segment_id": "source:beta", "unit_id": "u3"}])
    assert evaluate_boundaries(result, annotation, tolerance_units=0)["boundary_f1"]["f1"] == 1
    annotation["boundary_unit_refs"] = [{"segment_id": "source/alpha", "unit_id": "u1"}]
    assert evaluate_boundaries(result, annotation, tolerance_units=0)["boundary_f1"]["f1"] == 0
    annotation["boundary_unit_refs"] = ["u0"]
    with pytest.raises(ValueError, match="segment_id and unit_id"):
        evaluate_boundaries(result, annotation)
    annotation["boundary_unit_refs"] = [{"segment_id": "missing", "unit_id": "u0"}]
    with pytest.raises(ValueError, match="known units"):
        evaluate_boundaries(result, annotation)
    annotation["boundary_unit_refs"] = [result["entries"][1]["source_unit_refs"][0]] * 2
    with pytest.raises(ValueError, match="unique"):
        evaluate_boundaries(result, annotation)


def test_flat_source_rejects_duplicate_unit_ids_even_with_different_segments():
    result = output()
    assert len(index_source_units(result)) == 6
    result["source_units"][3]["id"] = "u0"
    with pytest.raises(ValueError, match="Duplicate lecture-wide unit ID"):
        index_source_units(result)


def test_cli_roundtrip_preserves_flat_units_and_source_annotation(tmp_path):
    input_path = tmp_path / "lecture.json"
    input_path.write_text(json.dumps(asdict(lecture())), encoding="utf-8")
    # from_dict accepts id as well as video_id.
    result_path, annotation_path, score_path = [tmp_path / name for name in
                                               ("result.json", "annotation.json", "scores.json")]
    main(["summarize", str(input_path), "-o", str(result_path), "--backend", "demo",
          "--chunking", "fixed", "--min-chunk-tokens", "1", "--max-chunk-tokens", "9"])
    result = json.loads(result_path.read_text())
    assert result["source_units"][0]["id"] == "u0"
    assert result["source_units"][3]["id"] == "u3"
    assert result["entries"][1]["source_unit_refs"][0] == {"segment_id": "source:beta", "unit_id": "u3"}
    main(["export-boundaries", str(result_path), "-o", str(annotation_path)])
    annotation = json.loads(annotation_path.read_text())
    annotation.update(annotation_status="complete", boundary_unit_refs=[
        {"segment_id": "source:beta", "unit_id": "u3"}])
    annotation_path.write_text(json.dumps(annotation))
    main(["evaluate-boundaries", str(result_path), str(annotation_path),
          "--tolerance-units", "0", "-o", str(score_path)])
    assert json.loads(score_path.read_text())["boundary_f1"]["f1"] == 1


def legacy_output():
    """Earlier exported schema, constructed independently of the new writer."""
    return {"schema_version": "1.1", "transcript_id": "old", "unitization_sha256": "old-hash",
            "source_sha256": "old-source", "source_segments": [
                {"id": "s00000", "start": 0, "end": 10, "text": "First."},
                {"id": "s00001", "start": 10, "end": 20, "text": "Second."}],
            "source_units": [
                {"id": "u000000", "segment_index": 0, "segment_id": "s00000", "start": 0,
                 "end": 10, "char_start": 0, "char_end": 6, "text": "First."},
                {"id": "u000001", "segment_index": 1, "segment_id": "s00001", "start": 10,
                 "end": 20, "char_start": 0, "char_end": 7, "text": "Second."}],
            "entries": [
                {"id": "c0", "start": 0, "end": 10, "source_unit_ids": ["u000000"],
                 "summary": "First.", "evidence": [{"id": "u000000", "segment_id": "s00000", "text": "First."}],
                 "boundary": {"start": {"reason": "lecture_start"}}},
                {"id": "c1", "start": 10, "end": 20, "source_unit_ids": ["u000001"],
                 "summary": "Second.", "evidence": [{"id": "u000001", "segment_id": "s00001", "text": "Second."}],
                 "boundary": {"start": {"at_char": 0}}}]}


def test_old_export_and_old_boundary_annotations_are_still_readable(tmp_path):
    old = legacy_output()
    original = deepcopy(old)
    assert list(judge_records(old))[1]["retrieved_contexts"] == ["Second."]
    assert evaluate_boundaries(old, {"transcript_id": "old", "boundaries": [10]}, 0)["boundary_f1"]["f1"] == 1
    annotation = {"transcript_id": "old", "unitization_sha256": "old-hash", "annotation_status": "complete",
                  "boundary_unit_ids": ["u000001"]}
    assert evaluate_boundaries(old, annotation, tolerance_units=0)["boundary_f1"]["f1"] == 1
    export_audit(old, tmp_path / "old-audit.csv")
    template = boundary_annotation_template(old)
    assert template["source_units"][1]["text"] == "Second."
    assert old == original


def test_old_unit_annotation_cannot_be_silently_reused_with_a_new_unitization():
    result = output()
    annotation = {"transcript_id": result["transcript_id"], "unitization_sha256": "old-hash",
                  "annotation_status": "complete", "boundary_unit_ids": ["u0"]}
    with pytest.raises(ValueError, match="unitization_sha256"):
        evaluate_boundaries(result, annotation)


def test_reference_key_is_scoped_to_its_transcript_not_a_global_registry():
    first = output("raw")
    second = deepcopy(first)
    second["transcript_id"] = "lecture-B"
    second["source_units"][0]["text"] = "Different lecture content."
    key = reference_key({"segment_id": "source/alpha", "unit_id": "u0"})
    assert index_source_units(first)[key][1]["text"] == "Alpha one."
    assert index_source_units(second)[key][1]["text"] == "Different lecture content."


def test_semantic_windows_follow_units_even_inside_one_time_segment():
    class TopicVectors(HashEmbedder):
        def encode(self, texts):
            return np.array([[1., 0.] if text.startswith("Alpha") else [0., 1.] for text in texts])

    separate = lecture()
    combined = Transcript.from_dict({"video_id": "one-time-block", "segments": [
        {"id": "time-block", "start": 0, "end": 30,
         "text": " ".join(s.text for s in separate.segments)}]})
    config = Config(chunking="semantic", evidence="raw", min_chunk_tokens=1,
                    max_chunk_tokens=100, evidence_tokens=100, boundary_window=2, boundary_threshold=0.5)
    one_block = run(combined, TopicVectors(), ExtractiveDemo(), config)
    two_blocks = run(separate, TopicVectors(), ExtractiveDemo(), config)
    assert [e["summary"] for e in one_block["entries"]] == [
        "Alpha one.\nAlpha two.\nAlpha three.", "Beta one.\nBeta two.\nBeta three."]
    assert [e["summary"] for e in two_blocks["entries"]] == [e["summary"] for e in one_block["entries"]]
    assert one_block["entries"][1]["boundary"]["start"]["before_unit_ref"] == {
        "segment_id": "time-block", "unit_id": "u3"}
    assert [(e["start"], e["end"]) for e in one_block["entries"]] == [(0, 30), (0, 30)]
    assert len(one_block["boundary_candidates"]) == 5
    assert one_block["segmentation"]["semantic_window_basis"] == "units"
    assert one_block["segmentation"]["boundary_window_units"] == 2


def test_nested_v2_exports_with_repeated_local_ids_remain_readable(tmp_path):
    old = legacy_output()
    old["schema_version"] = "2.0"
    old["unitization_sha256"] = "nested-hash"
    flat = old.pop("source_units")
    for segment, unit, entry in zip(old["source_segments"], flat, old["entries"]):
        segment["units"] = [{"id": "u0", **{key: unit[key] for key in ("char_start", "char_end", "text")}}]
        entry.pop("source_unit_ids")
        entry["source_unit_refs"] = [{"segment_id": segment["id"], "unit_id": "u0"}]
        entry["evidence"] = [{"segment_id": segment["id"], "unit_id": "u0"}]
    snapshot = deepcopy(old)
    assert len(index_source_units(old)) == 2
    assert [row["retrieved_contexts"] for row in judge_records(old)] == [["First."], ["Second."]]
    annotation = boundary_annotation_template(old)
    assert [u["id"] for u in annotation["source_units"]] == ["u0", "u0"]
    annotation.update(annotation_status="complete", boundary_unit_refs=[
        {"segment_id": "s00001", "unit_id": "u0"}])
    assert evaluate_boundaries(old, annotation, tolerance_units=0)["boundary_f1"]["f1"] == 1
    export_audit(old, tmp_path / "old-nested-audit.csv")
    assert old == snapshot
