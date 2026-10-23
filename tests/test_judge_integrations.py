"""Validate actual evaluation-library APIs using fixed fake model responses.

These are integration fixtures, not scientific evaluations or API requests.
"""

import os

import pytest

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
pytest.importorskip("deepeval")
pytest.importorskip("ragas")

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from sedl.judges import COHERENCE_RUBRIC, COHERENCE_STEPS, ExternalJudge


class FixtureLLM(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        return schema(score=3, reason="Fixed integration-test response.")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "test/fixed-response"


def test_deepeval_rubric_and_ragas_real_apis_without_network():
    adapter = ExternalJudge.__new__(ExternalJudge)
    adapter.case_type = LLMTestCase
    adapter.sample_type = SingleTurnSample
    adapter.geval = GEval(
        name="Test coherence", evaluation_steps=COHERENCE_STEPS,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        rubric=[Rubric(score_range=(i, i), expected_outcome=text)
                for i, text in enumerate(COHERENCE_RUBRIC, 1)],
        model=FixtureLLM(), async_mode=False)
    coherence = adapter.coherence("Source lecture.", "Summary.")
    assert coherence["score"] == 0.5
    assert coherence["score_1_to_5"] == 3
    model = FakeListChatModel(responses=[
        '{"statements": ["A model learns labels.", "The model is perfect."]}',
        '{"statements": [{"statement": "A model learns labels.", "reason": "Supported.", "verdict": 1}, '
        '{"statement": "The model is perfect.", "reason": "Unsupported.", "verdict": 0}]}',
    ])
    adapter.faithfulness = Faithfulness(llm=LangchainLLMWrapper(model))
    score = adapter.grounding("A model learns labels. The model is perfect.", ["A model learns labels."])
    assert score == {"status": "scored", "score": 0.5}
