"""Regression checks for segment-local IDs and unambiguous source tracing."""

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


def test_make_units_takes_one_segment_and_keeps_parent_data_on_segment():
    source = lecture()
    units = make_units(source.segments[0], [(ExtractiveDemo().count_tokens, 20)])
    assert [u.id for u in units] == ["u0", "u1", "u2"]
    assert [u.text for u in units] == ["Alpha one.", "Alpha two.", "Alpha three."]
    for unit in units:
        assert source.segments[0].text[unit.char_start:unit.char_end] == unit.text
        assert not any(hasattr(unit, field) for field in ("segment_id", "segment_index", "start", "end"))
    with pytest.raises(TypeError, match="one Segment"):
        make_units(source, [(len, 20)])


def test_transcript_owns_segments_and_segments_own_independently_numbered_units():
    source = lecture()
    prepared = source.with_units([(ExtractiveDemo().count_tokens, 20)])
    assert source.segments[0].units == ()  # Preparing does not mutate the input.
    assert [u.id for u in prepared.segments[0].units] == ["u0", "u1", "u2"]
    assert [u.id for u in prepared.segments[1].units] == ["u0", "u1", "u2"]
    assert prepared.segments[1].start == 30
    assert prepared.segments[1].units[0].text == "Beta one."
    assert asdict(prepared)["segments"][1]["units"][0]["id"] == "u0"


def test_prepared_transcript_can_be_processed_again_without_changing_source_identity():
    source = lecture()
    prepared = source.with_units([(ExtractiveDemo().count_tokens, 2)])
    config = Config(chunking="fixed", min_chunk_tokens=1, max_chunk_tokens=9)
    original = run(source, KnownVectors(), ExtractiveDemo(), config)
    repeated = run(prepared, KnownVectors(), ExtractiveDemo(), config)
    assert repeated["source_sha256"] == original["source_sha256"]
    assert repeated["unitization_sha256"] == original["unitization_sha256"]
    assert repeated["entries"] == original["entries"]
    assert repeated["source_segments"] == original["source_segments"]


def test_mmr_keeps_correct_embedding_rows_when_unit_ids_repeat_across_segments():
    result = output()
    assert result["schema_version"] == "2.0"
    assert "source_units" not in result
    assert [entry["summary"] for entry in result["entries"]] == ["Alpha one.", "Beta two."]
    assert result["entries"][0]["evidence"][0]["segment_id"] == "source/alpha"
    assert result["entries"][0]["evidence"][0]["unit_id"] == "u0"
    assert result["entries"][1]["evidence"][0]["segment_id"] == "source:beta"
    assert result["entries"][1]["evidence"][0]["unit_id"] == "u1"
    assert [(e["start"], e["end"]) for e in result["entries"]] == [(0, 30), (30, 60)]
    records = list(judge_records(result))
    assert records[0]["retrieved_contexts"] == ["Alpha one."]
    assert records[1]["retrieved_contexts"] == ["Beta two."]
    assert records[0]["source_chunk"] == "Alpha one.\nAlpha two.\nAlpha three."
    assert records[1]["source_chunk"] == "Beta one.\nBeta two.\nBeta three."


def test_same_local_id_is_resolved_under_the_correct_parent_in_audit(tmp_path):
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
    assert '"segment_id": "source:beta", "unit_id": "u0"' in rows[1]["selected_evidence"]


def test_boundaries_use_parent_and_local_id_and_reject_ambiguous_or_duplicate_refs():
    result = output()
    annotation = boundary_annotation_template(result)
    assert "boundary_unit_ids" not in annotation
    assert annotation["source_segments"][1]["units"][0]["id"] == "u0"
    annotation.update(annotation_status="complete", boundary_unit_refs=[
        {"segment_id": "source:beta", "unit_id": "u0"}])
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


def test_nested_source_rejects_duplicate_ids_within_one_segment():
    result = output()
    assert len(index_source_units(result)) == 6  # u0 in two parents is valid.
    result["source_segments"][0]["units"] += (result["source_segments"][0]["units"][0],)
    with pytest.raises(ValueError, match="Duplicate unit reference"):
        index_source_units(result)


def test_cli_roundtrip_preserves_nested_units_and_composite_annotation(tmp_path):
    input_path = tmp_path / "lecture.json"
    input_path.write_text(json.dumps(asdict(lecture())), encoding="utf-8")
    # from_dict accepts id as well as video_id and regenerates derived units.
    result_path, annotation_path, score_path = [tmp_path / name for name in
                                               ("result.json", "annotation.json", "scores.json")]
    main(["summarize", str(input_path), "-o", str(result_path), "--backend", "demo",
          "--chunking", "fixed", "--min-chunk-tokens", "1", "--max-chunk-tokens", "9"])
    result = json.loads(result_path.read_text())
    assert result["source_segments"][0]["units"][0]["id"] == "u0"
    assert result["source_segments"][1]["units"][0]["id"] == "u0"
    assert result["entries"][1]["source_unit_refs"][0] == {"segment_id": "source:beta", "unit_id": "u0"}
    main(["export-boundaries", str(result_path), "-o", str(annotation_path)])
    annotation = json.loads(annotation_path.read_text())
    annotation.update(annotation_status="complete", boundary_unit_refs=[
        {"segment_id": "source:beta", "unit_id": "u0"}])
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
    assert template["source_segments"][1]["units"][0]["text"] == "Second."
    assert old == original


def test_old_unit_annotation_cannot_be_silently_reused_for_new_local_ids():
    result = output()
    annotation = {"transcript_id": result["transcript_id"], "unitization_sha256": result["unitization_sha256"],
                  "annotation_status": "complete", "boundary_unit_ids": ["u0"]}
    with pytest.raises(ValueError, match="boundary_unit_refs"):
        evaluate_boundaries(result, annotation)


def test_reference_key_is_scoped_to_its_transcript_not_a_global_registry():
    first = output("raw")
    second = deepcopy(first)
    second["transcript_id"] = "lecture-B"
    second["source_segments"][0]["units"][0]["text"] = "Different lecture content."
    key = reference_key({"segment_id": "source/alpha", "unit_id": "u0"})
    assert index_source_units(first)[key][1]["text"] == "Alpha one."
    assert index_source_units(second)[key][1]["text"] == "Different lecture content."
