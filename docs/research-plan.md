# Four-month thesis scope

Recommendation: retain the original task of query-independent summaries of
timestamped lecture transcripts. Reframe the contribution as an empirical study
of segmentation, evidence selection, source support, and inference cost.
Do not claim semantic chunking or evidence-linked educational summaries are new
in themselves. Related work makes those claims difficult to defend.

Possible title: **Evaluating Segmentation and Evidence Selection for Traceable
Lecture Summarisation under Fixed Inference Budgets**.
The supervisor should approve the final research questions and evaluation scope.

## How the two papers inform the implementation

| Source | Relevant method | Use here | Scope difference |
| --- | --- | --- | --- |
| Xiao et al., *A Semantic-Adaptive Hybrid Retrieval Framework for the Education Domain*, DOI [10.1145/3839592.3839618](https://doi.org/10.1145/3839592.3839618) | Section 3.1: adjacent sentence cosine similarity, document percentile threshold, maximum chunk length | `sadc` baseline with a shared encoder and token-bounded units; direct boundary evaluation | Adapted baseline, not an exact BGE-M3 reproduction; no BM25, query retrieval, or reranker |
| Kumar, Rizvi, Hashmi, *An Explainable Hybrid Deep Learning Framework for Outcome-Oriented Long-Document Summarization Using Adaptive Feature Learning and Decision Optimization*, [Research Square v1](https://doi.org/10.21203/rs.3.rs-10936677/v1) | Outcome-conditioned evidence selection, summarisation, and sentence-to-evidence mapping | Separate human review of summary support in the source and selected generation evidence | Query-independent inference-only BART; no LoRA, outcome supervision, attention-based attribution, or composite-loss reproduction |

The second source is the supplied preprint, not a verified peer-reviewed result.
Its abstract names LED-LoRA while the detailed model description names BART-LoRA;
do not treat its architecture specification as unambiguous. Its evidence mapping
is finer than our chunk-level evidence links. This revision exports sentence
review rows, but does not implement its learned attribution mechanism.

## Research questions and comparisons

1. How do fixed-budget packing, adjacent-similarity percentile segmentation,
   and local unit-window segmentation compare on expert topic boundaries and
   lecture-summary coherence under the same output allowance?
2. For one segmenter chosen on development lectures, how does MMR evidence
   selection compare with raw input and chronological lead evidence in summary
   support, input compression, and inference time?
3. What proportion of reviewed summary sentences is supported by the source
   chunk and by the selected evidence, and which failures limit traceability?

RQ3 is an evidence-support audit. It does not demonstrate better learner trust
or faster verification. Retaining a learner-benefit claim would require a
separate approved study, recruitment, task design, and analysis.

Use two development lectures and approximately five held-out evaluation
lectures. Group related lectures to reduce leakage; near-duplicates need manual
review. With five test lectures, show each lecture's results and uncertainty;
individual chunks are not independent lecture-level samples.

First compare `fixed_raw`, `sadc_raw`, and `semantic_raw`. Select the evidence
segmenter on development results only. Run its `lead` and `mmr` variants as the
two additional conditions. The runner always produces these five conditions;
development runs can be repeated with changed choices in new output directories.
Freeze the final development manifest before creating the test manifest.
The runner checks disjoint paths, lecture IDs, and exact normalized text, and
rejects changes to segmentation/evidence configuration in the held-out manifest.
Use identical checkpoint revisions and generation arguments across both splits;
the manifest does not enforce CLI checkpoint arguments against earlier runs.

## Controls and measurements

- Same tokenizers, encoder, summarizer, unitization parameters, and maximum
  chunk budget. SADC operates on adjacent units, while the main semantic method
  compares left/right windows of units. Original time segments retain source
  timing without restricting topic-boundary locations.
- Same **lecture-level maximum output allowance** across segmenters. The default
  real-data manifest ratio is 10% of source BART tokens, configurable on development
  data. Allowances are distributed across every chunk without dropping source
  chunks. Exact equal generated lengths are not guaranteed.
- Lead and MMR use identical token and unit-count caps. Their actual selected
  lengths may differ. Raw versus MMR includes both compression and selection;
  lead versus MMR better isolates the choice of evidence under shared limits.
- Boundary F1 with a preregistered unit tolerance, plus Pk and WindowDiff.
  Unit annotations are shared across methods by their unitization hash.
  Prefer two annotators and adjudication if feasible; do not show predicted
  boundaries during initial annotation.
- Blind human summary-support review, reporting supported/unsupported/unclear
  and unrated counts separately for full-source and selected-evidence support.
  Automated G-Eval/RAGAs are optional secondary measures. Reviewers may split
  multi-claim sentences; document the final review unit consistently.
  Audit CSVs omit method labels. Randomize review order, rename files with
  opaque IDs, and retain a separate method mapping before distributing them.
- Actual summary/input token counts, synchronized total GPU runtime and peak
  allocated/reserved CUDA memory when on CUDA. CPU runs omit CUDA metrics.
  Stage timers are wall-clock estimates. Resident models are included in GPU
  memory; model loading is excluded from run times. Warm-up, order, and repeats
  must be controlled separately for publication-quality latency claims.

Check feasibility on development data: the output budget must allow at least
four tokens per chunk and at most the configured per-chunk generation cap.
This minimum is an engineering guard, not a sufficient summary-quality length.
Very fine segmentation may require a lower percentile, a larger maximum chunk
size, or a different output cap. Never silently omit
chunks or adjust a single method's budget after seeing test results.

## Sixteen-week schedule

| Weeks | Deliverable |
| --- | --- |
| 1–2 | Confirm RQs and corpus permissions; review related work; select development/test lectures |
| 3–4 | Validate JSON, pretrained model execution on Colab T4, all five conditions, annotation guide |
| 5–6 | Annotate development data; choose parameters/segmenter; freeze protocol and manifests |
| 7–9 | Annotate held-out boundaries; run frozen experiments; collect inference measurements |
| 10–11 | Blind summary-support review, adjudication, optional automated judging |
| 12–13 | Per-lecture analysis, error taxonomy, sensitivity checks restricted to development data |
| 14–16 | Write thesis, document reproducibility and limits, prepare a small demonstration |

The architecture, configurable model adapters, experiment records, evaluation,
and deployment notes provide a portfolio relevant to applied engineering and
technical consulting. A UI or hosted API is an optional final demonstration,
after the experiments; it should not displace the research evaluation.
