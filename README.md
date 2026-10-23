# Explainable Semantic Chunking for Lecture Summarisation

Research implementation for **Explainable Semantic Chunking for Abstractive
Summarisation of Long Educational Transcripts** (abstract v4.1, Jihye Kang).

The pipeline takes timestamped transcript segments, detects topic changes using
text embeddings, selects diverse evidence with MMR, generates summaries with
pretrained BART, and writes traceable Semantic Edit Decision Lists (SEDL).
The attached document is a research proposal: model choices and numerical
defaults here are implementation assumptions, not reported experimental results.
The research design keeps the original timestamped, query-independent task and adds
controlled comparisons informed by SAHR and EHDLF. See the
[four-month research plan](docs/research-plan.md) for the scope and paper mapping.

## Install

Python 3.11+ (validated on 3.12). CPU is supported; a GPU is optional.
In the current cloud environment the virtual environment is
`/workspace/.venvs/thesis`; activate it with:

```bash
source /workspace/.venvs/thesis/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_DISABLE_XET=1
cd /workspace/thesis
```

For a fresh CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural,dev]'
```

For offline algorithm development, install only `'.[dev]'` (no torch required).
To reproduce the full validated CPU environment in this cloud checkout, run
`bash scripts/setup-cloud.sh`. This installs the lockfile and runs the checks.
The neural and evaluation integrations are pinned in `pyproject.toml`.
`requirements-cpu.lock` records the validated Python 3.12 CPU environment,
including transitive dependencies and optional evaluation libraries.
Model revision flags accept Hugging Face commit hashes for
repeatable experiments. Resolved revisions and runtime versions are saved in SEDL.

## Input

The anticipated Dragana Naceva transcript format is:

```json
{
  "video_id": "lecture-01",
  "source": "whisper",
  "language": "en",
  "text": "Optional complete lecture text...",
  "segments": [
    {"id": "segment-01", "start": 0.0, "end": 14.8, "text": "Timestamp-aligned transcript text."}
  ],
  "provenance": {"tool": "whisper", "model": "transcription-model-name"}
}
```

Times are numeric **seconds**, ordered by segment start. Segment IDs are optional.
Full lecture text is preserved; timestamped segments are the authoritative input
for processing and provenance. Content present only in `text` is not processed.
The producer's uploaded JSON has been checked: its 149 segments use numeric
`start`/`end` values and `text`, without segment IDs. `video_id` is used as the
lecture ID; legacy `id` is supported as a fallback. Optional `source`, `language`,
and `provenance` are preserved in the SEDL `source_metadata` object. Recording
`language` does not automatically select a model or validate model compatibility.
The current models and sentence/keyword heuristics
target English. Korean transcripts require an appropriate embedding model and
summarisation checkpoint, plus separate validation.

## Transcript, segments, and a flat unit list

Version 0.4 keeps the original transcript separate from its derived units:

```text
Transcript (video_id)
  Segment s00000: timestamps, original text
  Segment s00001: timestamps, original text

Units returned by make_units(transcript, constraints):
  u0 → segment_id s00000
  u1 → segment_id s00000
  u2 → segment_id s00001
```

`make_units(transcript, constraints)` splits each segment's text and returns
one flat list. IDs count continuously through the lecture and restart at `u0`
for another transcript. A unit stores `segment_id`, character offsets, and text;
it has no segment index or copied timestamps. `Segment` has no `units` field,
and `Transcript` has no `with_units` method. The input objects are not changed.

SEDL schema **3.0** stores `source_segments` and `source_units` separately.
Chunks and evidence refer to units using `{ "segment_id": "s00001", "unit_id": "u2" }`;
the enclosing document supplies `transcript_id`. See the
[data model and migration notes](docs/data-model.md).

## Run

```bash
sedl summarize examples/lecture.json -o outputs/semantic-mmr.json
```

Default checkpoints: `sentence-transformers/all-MiniLM-L6-v2` and
`facebook/bart-large-cnn`. The first neural run downloads their pretrained
artifacts (roughly 2 GB, including BART). Model downloads require access to
`huggingface.co` and its artifact hosts, which can include
`cas-bridge.xethub.hf.co` and `us.aws.cdn.hf.co`.
No external API key is needed for local BART inference.

An explicit offline engineering demo runs without model downloads:

```bash
sedl summarize examples/lecture.json -o outputs/demo.json \
  --backend demo --min-chunk-tokens 20 --boundary-window 1
```

**Demo mode uses lexical hashing and echoes extracted evidence. It is neither
semantic modelling nor abstractive BART summarisation.** Outputs identify this
mode and the demo backends; do not include them in thesis performance claims.

Each SEDL entry contains the source interval, summary, evidence references,
MMR scores/ranks, boundary reasons/scores, and keywords. Evidence text and
character offsets are resolved from `source_units` using their segment IDs.
All source units are retained for auditing. Evidence times come from their
parent segments: sentence timestamps
are not fabricated. A chunk-to-evidence link does not prove that each generated
claim is entailed by its evidence.

## Compare methods

Run five conditions with the same encoder, summarizer, source units, and
lecture-level output allowance:

| Condition | Segmentation | Generation input |
| --- | --- | --- |
| `fixed_raw` | Consecutive units under a token limit | Whole chunk |
| `sadc_raw` | SAHR-inspired adjacent-similarity percentile | Whole chunk |
| `semantic_raw` | Local left/right unit windows across the lecture | Whole chunk |
| `<chosen>_lead` | Segmenter chosen on development data | Earliest units under evidence limits |
| `<chosen>_mmr` | Same chosen segmenter | MMR-selected units under the same limits |

The SADC variant is an **adaptation**, using the common encoder and common
token-bounded units. It is not an exact BGE-M3 reproduction of SAHR. Numerical
defaults, including the 10th percentile, are development choices.

Try the complete offline engineering example:

```bash
sedl experiment --manifest examples/study.json \
  --output-dir outputs/demo-study --backend demo
```

It writes one SEDL per condition, a manifest snapshot, `results.json`, and
`results.csv`. Results contain token counts, configuration, model identity,
runtime, and explicit failures. They contain no invented quality scores.
Existing results are never overwritten; choose a new output directory each time.

For real lectures, create a manifest next to your data paths:

```json
{
  "split": "development",
  "transcripts": ["data/transcripts/lecture-01.json", "data/transcripts/lecture-02.json"],
  "evidence_chunker": "semantic",
  "summary_ratio": 0.1,
  "base_config": {"sadc_percentile": 10.0}
}
```

Paths are relative to the **manifest file**, independently of notebook location.
The toy manifest's explicit 128-token budget is only for its small synthetic
lecture; omit that override for real data. A ratio computes one fixed output
allowance per lecture across conditions. Generation may stop early, so report
actual lengths too. Infeasible allocations are reported as failures; adjust
granularity/budgets on development data before freezing the experiment.

For the held-out run, set `split` to `test`, use separate transcript paths,
and add `"development_manifest": "development.json"`. Keep `base_config`,
`evidence_chunker`, and `summary_ratio` equal to the final development manifest.
The runner rejects overlapping paths, IDs, or identical normalized transcript
text. Curate related/near-duplicate lectures manually and use the same model
revisions and generation settings for both splits.

On a Colab **GPU runtime**, after opening the project directory:

```python
!python -m pip install -e ".[neural]"
import torch
print(torch.cuda.is_available())  # Must be True for --device cuda
!python -m sedl.cli experiment --manifest development.json --output-dir outputs/dev-01 --device cuda
```

Use the commands without `!` in a terminal. Python modules remain `.py`; no
conversion to `.ipynb` is needed. A Colab T4 is a suitable starting point for
these inference-only models; actual memory/latency must be measured there.
The CPU lockfile is for the cloud CPU environment, not Colab's GPU environment.

## Evaluate and test

```bash
python -m pytest -q
sedl export-boundaries outputs/demo.json -o outputs/topic-annotation.json
# Human annotator fills boundary_unit_refs and marks annotation_status complete.
sedl evaluate-boundaries outputs/demo.json outputs/topic-annotation.json \
  --tolerance-units 1 -o outputs/boundary-scores.json
sedl export-audit outputs/demo.json -o outputs/evidence-audit.csv
# Human reviewer fills source_support and evidence_support in the CSV.
sedl score-audit outputs/evidence-audit.csv -o outputs/evidence-scores.json
sedl export-judges outputs/demo.json -o outputs/judge-inputs.jsonl
```

Unit boundary annotations use a hash to verify common text units, preserving
distinct cuts inside a single timestamped segment. Evaluation reports one-to-one
boundary F1, Pk, and WindowDiff. Legacy second-based `examples/boundaries.json`
is also supported, but inherited segment times limit its precision.
Annotation templates omit model-predicted boundaries to reduce anchoring.
Audit ratings are `supported`, `unsupported`, or `unclear`, separately for the
full source chunk and selected generation evidence; blank rows remain unrated.
Reviewers may need to split compound claims in a sentence. Reports keep unclear
and unrated counts visible. These reviews do not establish causal attribution.
The example is synthetic; none of its scores are research results.
Neural integration tests use tiny, locally generated random BERT/BART models;
they verify tokenizer/model integration, not pretrained summary quality.

Optional G-Eval and RAGAs integrations run via:

```bash
python -m pip install -e '.[judges]'
# Configure JUDGE_API_KEY securely in your environment, never in source control.
sedl evaluate-judges outputs/semantic-mmr.json \
  --judge-model gpt-4o-2024-08-06 --faithfulness-context both \
  -o outputs/judge-scores.json
```

This command sends source text/summaries to the OpenAI API and incurs API costs.
It requires `JUDGE_API_KEY` and access to `api.openai.com`; neither is needed
for local summarisation. Demo outputs are rejected. G-Eval uses DeepEval's
implementation with a documented 1–5 narrative coherence rubric, applied to the
whole ordered lecture summary. RAGAs Faithfulness is applied per chunk against
the full source chunk, selected evidence, or both. Undefined scores are recorded
and excluded from means with counts. `export-judges` makes no API calls.

The library integrations are tested with fixed fake model responses; **no real
judge API evaluation has been run**. Scientific evaluation still needs the five
annotated lectures and an agreed judge model/rubric. RQ3 also needs a learner
verification study or a defined traceability task; boundary F1 cannot answer it.
Faithfulness scores cannot guarantee elimination of hallucinations.

See [the research implementation notes](docs/methodology.md) for remaining work
and [the current validation record](docs/validation.md) for verified capabilities.
