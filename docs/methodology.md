# Mapping the proposal to this implementation

Source: abstract v4.1, January 19, 2026, especially sections 5.1–5.7.
Document content is research context, not executable setup instructions.

## Pipeline

1. Validate ordered timestamped segments. Preserve original segment text,
   optional full lecture text, source, language, and transcription provenance.
   Use `video_id` as lecture identity, falling back to legacy `id` when needed.
   Split into sentence-like units and split any
   oversized unit so it fits both the embedding and generation tokenizers.
   Nest units under their original segment, storing character offsets on each
   unit and retaining timing only on the parent segment. Unit IDs restart at
   u0 within each segment; references use both segment_id and unit_id.
2. Encode units with all-MiniLM-L6-v2 by default. Compute a character-weighted
   mean of unit embeddings for each original segment and normalize it.
3. For each inter-segment boundary, average normalized segment embeddings on
   its left and right (default: two segments each). The boundary score is
   `1 - cosine(left_mean, right_mean)`. A score at least 0.35 starts a new chunk
   once the preceding chunk reaches 80 BART tokens. A 900-token maximum forces
   a boundary even if cohesion is high. Minimum length is a preference: final
   chunks and capacity-forced chunks can be shorter. Forced splits can occur
   inside a source segment, and their intervals may therefore overlap.
4. Rank units in each chunk against its character-weighted embedding centroid.
   Greedy MMR chooses the highest fitting score:

   ```text
   score(i) = lambda * cosine(unit_i, chunk_centroid)
              - (1-lambda) * max(cosine(unit_i, selected_j))
   ```

   Redundancy is zero before the first selection. Default lambda: 0.7. Select at
   most eight units with a combined generation input of at most 512 BART tokens.
   Exact combined token counts are checked after chronological ordering; no
   assumption that token counts are additive. Output records both selection
   ranks and chronological order. Units are split to fit the evidence budget
   in both ablations, so raw/MMR see identical chunk candidates.
5. BART-large-CNN generates a summary of the evidence in chronological order,
   with deterministic beam search, four beams, and at most 128 output tokens.
   No source input is silently truncated. The raw ablation uses all chunk text.
   Unlike the input, the generated summary is intentionally length bounded.
   An optional lecture-level allowance distributes output tokens across all
   chunks proportionally to their source lengths, subject to a four-token
   minimum and the per-chunk cap. Allocation totals are exact; actual generated
   lengths can be shorter. The experiment runner always supplies this allowance.
6. Export source and generation budgets, actual source intervals and offsets,
   boundary decisions, selected evidence, and frequency-based English keywords
   into SEDL. Original text and source hashes make decisions auditable.
   Schema 2.0 stores the source hierarchy as `source_segments[].units` and
   records `source_unit_refs` on chunks. Evidence contains references and scores;
   text and timing are resolved from the parent hierarchy. See
   [data-model.md](data-model.md) for Python usage and migration.

These numerical settings are configurable development defaults, not values
specified or validated by the proposal. Tune on a development split, freeze
settings and model revisions, then evaluate on held-out lectures. BART-large-CNN
is news-trained; suitability for educational text is an experimental question.
The encoder is an English sentence model, with its own shorter input capacity.

## Experimental design

Version 0.2 revises the design in light of SAHR and EHDLF; see
[research-plan.md](research-plan.md) for citations, five conditions, data splits,
and the four-month scope. Keep the original summarisation task and evaluate
engineering tradeoffs rather than claim the component algorithms are new.

The additional `sadc` segmenter computes cosine similarity between each pair
of adjacent common units, sets a document-specific percentile threshold, and
cuts where similarity is strictly below it or the hard token maximum is hit.
It follows SAHR's percentile rule without the local-window method's minimum
chunk length. Default percentile 10 is an implementation assumption. Shared
MiniLM embeddings and token-bounded units make it an adaptation, not an exact
reproduction of SAHR's BGE-M3 sentence method.

The `lead` evidence control takes a chronological prefix, stopping at the first
unit that does not fit. It has MMR's same token and unit-count caps. MMR may
instead skip a non-fitting candidate and select a shorter later unit. Both
serialize chosen evidence chronologically. Neither guarantees equal actual
input length; token counts are recorded for interpreting this difference.

| Research question | Comparison | Implemented here | Still required |
| --- | --- | --- | --- |
| RQ1: narrative/segmentation | semantic vs SADC adaptation vs fixed chunks | paths, boundary candidates, unit F1/Pk/WindowDiff | expert topic boundaries, pretrained summaries, coherence judgement |
| RQ2: evidence stage | chosen segmenter: raw vs lead vs MMR | shared segmentation and evidence caps, lecture output cap | held-out quality and cost measurements |
| RQ3: evidence support | source support vs selected-evidence support | blank sentence review sheets and explicit rating aggregation | blind human review and error analysis |

The fixed baseline packs units under a token limit and covers the entire
transcript. It is a sentence-preserving fixed-budget baseline, not an experiment
that truncates a whole lecture to the first model input window. Explicitly name
that difference in any thesis experiment. No model is fine-tuned here.

## Evaluation limits

* Preferred reference boundaries identify the first common unit after a topic
  transition, excluding lecture start. Annotation status must be complete and
  the unitization hash must match. One-to-one F1 uses a configurable unit
  tolerance (default one). Pk and WindowDiff use a window equal to half the
  reference mean segment length, rounded to the nearest integer, constrained
  to 1..N-1 units. A single-unit lecture has undefined Pk/WindowDiff. These
  metrics are independent of coarse source timing.
* Legacy reference boundaries are internal topic transition times, in seconds. One
  prediction matches at most one reference and vice versa; an earliest-feasible
  sorted matching maximizes cardinality within the tolerance window. Empty/empty
  F1 is 1; one empty side yields F1 0. Use a fixed tolerance across methods.
* When a boundary lies inside a segment, only segment timing is known.
  The report flags such cuts; duplicates at the same timestamp are collapsed.
  Do not treat this legacy coarse temporal measure as exact unit accuracy.
* Exported judge inputs include both shortlisted evidence and the full source
  chunk. `evaluate-judges --faithfulness-context both` runs separate RAGAs
  Faithfulness evaluations for grounding in the generation input and in the
  larger source. Export itself does not run a judge.
* G-Eval uses DeepEval 3.7.0 with explicit evaluation steps and a five-level
  narrative coherence rubric in `src/sedl/judges.py`. The rubric targets logical
  ordering and transitions, not factuality; RAGAs evaluates grounding separately.
  Evaluation is lecture-level across all ordered summary chunks. DeepEval uses
  log-probability weighting when the configured judge supports it and falls back
  to a direct rating otherwise. This adapter is not a claimed exact reproduction
  of the original G-Eval paper. Its native normalized score is saved along with
  the reverse-transformed 1–5 rubric score and the judge's rationale.
* RAGAs 0.3.9 Faithfulness decomposes responses into statements and judges
  support from supplied contexts. A high score is not proof of zero
  hallucinations. Library integration is tested with fixed fake responses;
  external model evaluation has not been executed. Nonfinite scores are marked
  undefined and excluded from reported means with explicit counts.
* SEDL provides post-hoc provenance, not attribution of BART's internal causal
  decisions. Evidence may be insufficient for a generated claim. Per-claim
  entailment is not implemented. The manual audit export presents each summary
  sentence, full source chunk, and selected evidence. It leaves both support
  ratings blank. Aggregation reports supported, unsupported, unclear, and
  unrated counts; supported fractions exclude unclear/unrated rows and must
  be reported with determinate coverage. No ratings means an undefined rate.

## Next inputs needed

1. One actual Whisper JSON has now been supplied and checked: `video_id`,
   `source`, `language`, `text`, `segments`, and `provenance` are supported.
   Its 149 segments pass validation. Further data-quality checks, including
   consistency between full lecture text and segments, remain outstanding.
2. The five evaluation lectures and manually annotated internal boundaries,
   with development/test split specified before tuning.
3. Completed source/evidence-support audit; optionally G-Eval/RAGAs judge access.
   A separate learner verification study is needed only if retaining the
   original stronger learner-benefit RQ3.

The initial implementation intentionally keeps these unknowns visible rather
than presenting a synthetic-data run as thesis validation.
