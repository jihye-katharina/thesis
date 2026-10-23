from copy import deepcopy

import pytest

from sedl.backends import ExtractiveDemo, HashEmbedder
from sedl.data import Transcript
from sedl.judges import ExternalJudge, checked_score, evaluate_judges
from sedl.pipeline import run


class RecordingJudge:
    """No external API calls; verify context routing and aggregation only."""
    model = "test/recording-judge"

    def __init__(self):
        self.calls = []

    def coherence(self, source, summary):
        self.calls.append(("coherence", source, summary))
        return {"status": "scored", "score": 0.5}

    def grounding(self, summary, contexts):
        self.calls.append(("grounding", summary, contexts))
        return checked_score(float("nan") if len(self.calls) == 2 else 0.75)


def test_judge_context_routing_undefined_scores_and_whole_lecture_coherence():
    source = Transcript.from_dict({"segments": [
        {"start": 0, "end": 10, "text": "First example. Second example."}]})
    result = run(source, HashEmbedder(), ExtractiveDemo())
    # Simulate a neural SEDL; this fixture is not written as a research result.
    result = deepcopy(result)
    result["mode"] = "neural"
    judge = RecordingJudge()
    scores = evaluate_judges(result, judge, "both")
    assert judge.calls[0][1] == "First example. Second example."
    assert judge.calls[0][2] == result["entries"][0]["summary"]
    assert judge.calls[1][2] == ["First example.\nSecond example."]
    assert sorted(judge.calls[2][2]) == ["First example.", "Second example."]
    assert scores["aggregates"]["faithfulness_source"] == {"mean": None, "scored": 0, "undefined": 1}
    assert scores["aggregates"]["faithfulness_evidence"]["mean"] == 0.75


def test_external_evaluation_rejects_demo_and_missing_credential(monkeypatch):
    with pytest.raises(ValueError, match="demo"):
        evaluate_judges({"mode": "offline_demo"}, RecordingJudge())
    monkeypatch.delenv("THESIS_TEST_MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="securely"):
        ExternalJudge("gpt-4o", "THESIS_TEST_MISSING_KEY")


def test_out_of_range_scores_are_not_silently_clamped():
    with pytest.raises(ValueError, match="outside"):
        checked_score(1.01)
    assert checked_score(float("nan")) == {"status": "undefined", "score": None}
