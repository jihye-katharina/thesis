"""Behavioral checks for controlled comparisons, not model-quality claims."""

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from sedl.audit import export_audit, score_audit
from sedl.backends import ExtractiveDemo, HashEmbedder
from sedl.budgets import allocate_summary_budget
from sedl.cli import main
from sedl.data import Transcript, make_units
from sedl.evaluation import boundary_annotation_template, evaluate_boundaries, segmentation_errors
from sedl.experiments import run_experiment, variants
from sedl.pipeline import Config, run, sadc_boundary_scores, select_lead, select_mmr
from sedl.references import iter_source_units


def lecture():
    # Multiple semantic changes inside ONE timestamped segment.
    return Transcript.from_dict({"video_id": "test-lecture", "segments": [
        {"start": 0, "end": 60,
         "text": "Apple grows. Apple harvest. Ocean waves. Ocean tides. Apple returns."}]})


class Topics(HashEmbedder):
    def encode(self, texts):
        return np.array([[1., 0.] if "Apple" in t else [0., 1.] for t in texts])


def test_sadc_percentile_strict_comparison_and_within_segment_cuts():
    config = Config(chunking="sadc", evidence="raw", sadc_percentile=50,
                    min_chunk_tokens=80, max_chunk_tokens=100, evidence_tokens=100)
    result = run(lecture(), Topics(), ExtractiveDemo(), config)
    refs = [ref for ref, _, _ in iter_source_units(result)]
    assert [e["source_unit_refs"] for e in result["entries"]] == [refs[:2], refs[2:4], refs[4:]]
    # SADC deliberately ignores the local-window method's minimum length.
    assert all(e["source_tokens"] < 80 for e in result["entries"])
    assert result["segmentation"]["sadc_similarity_threshold"] == 0.5
    assert result["segmentation"]["paper_reproduction"] is False
    assert all(e["start"] == 0 for e in result["entries"])
    assert sadc_boundary_scores(np.array([[1., 0.]]), 50) == ({}, None)
    # At p=0, similarity equal to the minimum does NOT trigger a split.
    tied = run(lecture(), Topics(), ExtractiveDemo(), replace(config, sadc_percentile=0))
    assert len(tied["entries"]) == 1


@pytest.mark.parametrize("chunking", ["fixed", "semantic", "sadc"])
def test_hard_limits_and_shared_unitization_for_all_evidence_controls(chunking):
    source = Transcript.from_dict({"segments": [{"start": 0, "end": 60,
                                                "text": "word " * 150 + "."}]})
    config = Config(chunking=chunking, min_chunk_tokens=1, max_chunk_tokens=20,
                    evidence_tokens=7, max_evidence_units=2)
    results = [run(source, HashEmbedder(), ExtractiveDemo(), replace(config, evidence=e))
               for e in ("raw", "lead", "mmr")]
    assert len({r["unitization_sha256"] for r in results}) == 1
    for result in results:
        assert [ref for e in result["entries"] for ref in e["source_unit_refs"]] == [
            ref for ref, _, _ in iter_source_units(result)]
        assert all(e["source_tokens"] <= 20 for e in result["entries"])
        if result["config"]["evidence"] != "raw":
            assert all(e["generation_input_tokens"] <= 7 and len(e["evidence"]) <= 2
                       for e in result["entries"])


def test_lead_is_a_prefix_and_mmr_changes_choice_under_same_caps():
    source = lecture()
    counter = ExtractiveDemo().count_tokens
    units = make_units(source, [(counter, 100)])[:3]
    vectors = np.array([[1., 0.], [1., 0.], [0., 1.]])
    lead = select_lead(units, counter, 6, 2)
    mmr = select_mmr(units, vectors, counter, 6, 2, 0.5)
    assert [i for i, _ in lead] == [0, 1]
    assert [i for i, _ in mmr] == [0, 2]


def test_output_allocation_exact_total_and_infeasible_budget_rejected():
    values = allocate_summary_budget([10, 30, 60], 50, 30)
    assert sum(values) == 50
    assert all(4 <= v <= 30 for v in values)
    assert values[0] < values[1] < values[2]
    assert allocate_summary_budget([1, 1], 9, 10) == [5, 4]
    for budget in (7, 21):
        with pytest.raises(ValueError, match="cannot fit"):
            allocate_summary_budget([1, 2], budget, 10)
    config = Config(chunking="sadc", sadc_percentile=50, summary_budget_tokens=30)
    result = run(lecture(), Topics(), ExtractiveDemo(), config)
    assert sum(e["allocated_output_tokens"] for e in result["entries"]) == 30
    assert result["statistics"]["summary_tokens"] <= 30


def test_unit_annotations_do_not_collapse_same_timestamp_cuts():
    result = run(lecture(), Topics(), ExtractiveDemo(), Config(chunking="sadc", sadc_percentile=50))
    annotation = boundary_annotation_template(result)
    with pytest.raises(ValueError, match="complete"):
        evaluate_boundaries(result, annotation)
    annotation["annotation_status"] = "complete"
    annotation["boundary_unit_refs"] = [e["source_unit_refs"][0] for e in result["entries"][1:]]
    score = evaluate_boundaries(result, annotation, tolerance_units=0)
    assert score["boundary_f1"]["predicted"] == 2
    assert score["boundary_f1"]["f1"] == 1
    assert score["segmentation_errors"]["pk"] == 0
    annotation["unitization_sha256"] = "wrong"
    with pytest.raises(ValueError, match="sha256"):
        evaluate_boundaries(result, annotation)


def test_pk_and_windowdiff_penalise_missed_and_extra_boundaries():
    assert segmentation_errors([4], [4], 8) == {"pk": 0, "window_diff": 0, "window_units": 2}
    assert segmentation_errors([], [4], 8)["pk"] == pytest.approx(2 / 6)
    assert segmentation_errors([2, 4, 6], [4], 8)["window_diff"] > 0
    assert segmentation_errors([], [], 1)["pk"] is None


def test_audit_exports_unrated_and_scores_only_explicit_labels(tmp_path):
    result = run(lecture(), Topics(), ExtractiveDemo(), Config())
    path = tmp_path / "audit.csv"
    export_audit(result, path)
    score = score_audit(path)
    assert score["ratings"]["source_support"]["supported_fraction_determinate"] is None
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["source_chunk"]
    assert rows[0]["selected_evidence"]
    rows[0]["source_support"] = "supported"
    rows[0]["evidence_support"] = "unclear"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    scored = score_audit(path)
    assert scored["ratings"]["source_support"]["supported"] == 1
    assert scored["ratings"]["source_support"]["unrated"] == len(rows) - 1
    assert scored["ratings"]["evidence_support"]["unclear"] == 1


def test_experiment_cli_five_conditions_and_no_overwrite(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/study.json"
    output = tmp_path / "study"
    main(["experiment", "--manifest", str(example), "--output-dir", str(output), "--backend", "demo"])
    report = json.loads((output / "results.json").read_text())
    assert len(report["runs"]) == 5
    assert {r["status"] for r in report["runs"]} == {"completed"}
    assert len({r["output_budget_tokens"] for r in report["runs"]}) == 1
    results = [json.loads((output / r["output"]).read_text()) for r in report["runs"]]
    assert len({r["unitization_sha256"] for r in results}) == 1
    assert all(r["statistics"]["summary_tokens"] <= 128 for r in results)
    assert all(r["runtime"]["seconds"] > 0 for r in results)
    assert (output / "results.csv").exists()
    with pytest.raises(ValueError, match="empty"):
        run_experiment(example, output, HashEmbedder(), ExtractiveDemo())


def test_infeasible_experiment_reports_failures_and_cli_nonzero(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = {"split": "development", "transcripts": [str(root / "examples/lecture.json")],
                "evidence_chunker": "semantic", "base_config": {"summary_budget_tokens": 1}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as exit_info:
        main(["experiment", "--manifest", str(path), "--output-dir", str(tmp_path / "out"), "--backend", "demo"])
    assert exit_info.value.code == 1
    runs = json.loads((tmp_path / "out/results.json").read_text())["runs"]
    assert all(r["status"] == "failed" and "cannot fit" in r["error"] for r in runs)


def test_test_manifest_rejects_development_overlap_and_changed_configuration(tmp_path):
    source = {"video_id": "dev", "segments": [{"start": 0, "end": 1, "text": "Hello."}]}
    (tmp_path / "dev.json").write_text(json.dumps(source))
    source["video_id"] = "test"
    source["segments"][0]["text"] = "Different evaluation lecture."
    (tmp_path / "test.json").write_text(json.dumps(source))
    manifest = {"split": "development", "transcripts": ["dev.json"], "evidence_chunker": "semantic",
                "base_config": {"summary_budget_tokens": 20}}
    (tmp_path / "development.json").write_text(json.dumps(manifest))
    manifest.update(split="test", development_manifest="development.json")
    path = tmp_path / "test-study.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjoint"):
        run_experiment(path, tmp_path / "out", HashEmbedder(), ExtractiveDemo())
    manifest["transcripts"] = ["test.json"]
    manifest["evidence_chunker"] = "sadc"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Freeze"):
        run_experiment(path, tmp_path / "out", HashEmbedder(), ExtractiveDemo())
    manifest["evidence_chunker"] = "semantic"
    path.write_text(json.dumps(manifest))
    assert len(run_experiment(path, tmp_path / "out", HashEmbedder(), ExtractiveDemo())["runs"]) == 5


@pytest.mark.parametrize("value", [-1, 101, float("nan")])
def test_bad_sadc_percentiles_rejected(value):
    with pytest.raises(ValueError, match="sadc_percentile"):
        run(lecture(), Topics(), ExtractiveDemo(), Config(sadc_percentile=value))
