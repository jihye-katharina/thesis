# Validation record — updated 2026-10-23

Current cloud instance: Python 3.12.14, Linux CPU. Dependencies installed in
`/workspace/.venvs/thesis`, including torch 2.9.0+cpu, transformers 4.57.1,
sentence-transformers 5.1.1, DeepEval 3.7.0, and RAGAs 0.3.9.
The complete installed dependency versions are in `requirements-cpu.lock`.

## Passed

* **73 tests**: transcript validation, source coverage through long-segment
  splitting, original-character provenance, overlapping timestamp handling,
  known-geometry topic changes, raw/MMR ablation parity, MMR diversity and
  exact token budgeting, one-to-one boundary F1, CLI export/evaluation, and
  undefined judge score handling, video ID mapping, and source metadata
  preservation through SEDL export.
* Revised design: percentile SADC adaptation (including strict ties and
  within-segment changes), shared units across all evidence methods, lead/MMR
  controls, lecture output allowances and infeasibility errors, unit-coordinate
  F1/Pk/WindowDiff, blank human audits, five-condition CLI runs, failed-run
  reporting, result overwrite protection, development/test overlap checks and
  frozen manifest configuration checks.
* Nested hierarchy: each segment owns units numbered from `u0`; units have no
  copied parent IDs, indices, or timestamps. Regression checks verify MMR
  embedding-row selection with repeated local IDs, reference resolution in
  audits/judge exports, composite boundary annotations, duplicate rejection,
  JSON CLI round trips, and read compatibility with SEDL 1.x exports.
* Actual SentenceTransformer and BART adapter execution, including rejection
  of overlong model inputs, with tiny **random local test models**. No downloaded
  pretrained checkpoints are used in these tests.
* Actual DeepEval G-Eval rubric/scale and RAGAs Faithfulness library calls with
  **fixed fake model responses**. These verify integration and context routing;
  the fixed scores are test fixtures, not research measurements.
* Offline CLI example: six chunks, eighteen source units, all eighteen units
  retained in chunk coverage, and every selected evidence offset verified
  against its original source segment. Demo uses hashing/extractive echo.
* Boundary F1 report and JSONL judge-input export generated from this example.
* `python -m pip check`: no broken requirements.
* Uploaded Whisper input `-0JE4p3b1YE.json`: all 149 segments pass input
  validation. An offline demo export preserves `video_id`, full lecture text,
  `source`, `language`, and the complete `provenance` object. This checks input
  compatibility and metadata retention, not pretrained summary quality.
  The revised five-condition runner also completes on this input in demo mode,
  with a shared source-unit hash, full ordered source coverage, preserved
  metadata, and the same 1,081-token lecture output cap in each condition.
  These lexical-demo token counts and timings are not neural research results.
  After the hierarchy refactor, all five conditions still preserve exactly the
  same 649 source text units, character offsets, source hashes, metadata,
  selected evidence, and demo summaries as the prior implementation. The new
  output representation uses SEDL 2.0 and segment-local unit IDs.

The pinned RAGAs LangChain wrapper emits deprecation warnings. It remains
functional in the validated version; replacing it belongs with a future
evaluation-dependency upgrade.

## Not yet validated

* **Pretrained MiniLM/BART end-to-end run**: the latest retry can fetch model
  configuration from `huggingface.co` (HTTP 200), but downloading either model's
  weights redirects to `us.aws.cdn.hf.co`, blocked by proxy `403 Forbidden`.
  `huggingface.co` and `cas-bridge.xethub.hf.co` were saved as custom domain
  additions in the environment configuration draft. The additional observed
  artifact host is not in that earlier draft. Saving a draft does not by itself
  apply the policy to the running instance or publish a snapshot.
* **Actual lecture quality**: one uploaded Whisper JSON has been checked for
  input compatibility and metadata retention. Pretrained summary quality and
  comparison against expert-labelled boundaries remain unvalidated.
* **External G-Eval/RAGAs judgements**: no paid API calls made. Neither
  `JUDGE_API_KEY` nor `OPENAI_API_KEY` is present in this instance. Configure a
  judge key securely and allow `api.openai.com` when choosing to run this
  optional step. No credential values were inspected or written into code.
* **Research outcomes and RQ3 learner verifiability study**: not performed.
  Source traceability is implemented; it does not establish claim entailment
  or improved learner trust. No hallucination-elimination claim is made.

Tests ran against the workspace project. Environment publication and fresh-task
restoration have not been validated.
