# Transcript, segments, and a flat list of units

| Object | Fields | Scope of its ID |
| --- | --- | --- |
| `Transcript` | `id`, `segments`, full text and optional metadata | One lecture; input `video_id` is preferred |
| `Segment` | `id`, `start`, `end`, original `text` | Unique within its transcript |
| `Unit` | `id`, `segment_id`, `char_start`, `char_end`, `text` | Unique across its transcript |

Default segment IDs remain `s00000`, `s00001`, etc., unless supplied by the input.
Units are numbered `u0`, `u1`, ... continuously across all segments. The next
lecture starts at `u0` again. Each sentence-like span receives the next ID.

`Segment` has no `units` property. `Transcript` has no `with_units` method.
Units are returned as a separate flat list. The `segment_id` on a unit still
links it to its original text and timestamp interval, without storing units
inside the segment or duplicating timestamps on each unit.

## Python usage

After parsing the input JSON as a `Transcript`:

```python
units = make_units(transcript)
unit = units[0]

segments_by_id = {segment.id: segment for segment in transcript.segments}
segment = segments_by_id[unit.segment_id]

print(transcript.id, unit.id, unit.segment_id)
print(segment.start, segment.end)
assert segment.text[unit.char_start:unit.char_end] == unit.text
```

`make_units` processes the segments in order, then sentence-like spans within
each segment. It never mutates the transcript. `units = []` is created once
before the segment loop, so `len(units)` gives continuous IDs. Character
offsets remain relative to the original segment, with an inclusive start and
exclusive end. Units have no `segment_index`, `start`, or `end` fields.

There is no `constraints` argument or token-based subdivision in `make_units`.
Whole sentences are assumed to fit the selected models. Chunk and evidence
budgets still govern how sentences are combined. An oversized individual
sentence causes a capacity error at the relevant pipeline/model stage, rather
than automatic splitting or silent truncation. Capacities depend on the selected
model and tokenizer; a sentence below 512 tokens may exceed a smaller encoder's
configured window.

This splitter assumes sentence boundaries are preserved by the transcript
producer's time windows. It does not reconstruct a sentence cut between two
source segments. Abbreviation handling is a simple English-oriented heuristic.

## Semantic comparisons

The entire ordered unit list is embedded together, in batches as needed.
The main `semantic` method compares equal-weight left/right windows of unit
embeddings at every inter-unit boundary. `boundary_window=2` means up to two
units on each side. A topic boundary can occur inside a source time segment.
Original timestamps affect source tracing, not semantic boundary candidates.

The `sadc` adaptation compares adjacent units using a document percentile.
Fixed chunking packs consecutive units under the same token cap. Token limits
and minimum chunk preferences still apply as documented in methodology.md.
Two chunks can share a time interval when only coarse source timing is known.

## SEDL output and references

New outputs use schema `3.0`. A shortened example is:

```json
{
  "schema_version": "3.0",
  "transcript_id": "lecture-01",
  "source_segments": [
    {"id": "s00000", "start": 0.0, "end": 10.0, "text": "Hello. World."},
    {"id": "s00001", "start": 10.0, "end": 20.0, "text": "Next."}
  ],
  "source_units": [
    {"id": "u0", "segment_id": "s00000", "char_start": 0, "char_end": 6, "text": "Hello."},
    {"id": "u1", "segment_id": "s00000", "char_start": 7, "char_end": 13, "text": "World."},
    {"id": "u2", "segment_id": "s00001", "char_start": 0, "char_end": 5, "text": "Next."}
  ]
}
```

Chunks retain `source_unit_refs`, and evidence records retain source reference
fields plus selection scores. The reference `{ "segment_id": "s00001", "unit_id": "u2" }`
resolves to the original segment and the specific unit. The document supplies
the transcript ID. When combining lectures, keep `transcript_id` too.
Audit and judge exporters resolve text/timing from these source lists.

## Migration

- Replace calls with token constraints or `transcript.with_units(...)`
  with `make_units(transcript)`.
- Read units from the returned list or exported `source_units`, rather than
  from `segment.units` / `source_segments[].units`.
- For new outputs, create fresh boundary annotation templates with
  `export-boundaries`. Unit numbering and the unitization hash change.
- Readers continue to accept old flat SEDL 1.x and nested SEDL 2.0 exports
  together with the corresponding annotations. Nested 2.0 local IDs still need
  both segment and unit IDs; current flat IDs are unique within the lecture.
- Version 0.4 changes `semantic` from windows of original time segments to
  windows of units. Record implementation versions and rerun development
  selection before held-out experiments. Earlier semantic results are not the
  same experimental condition. `fixed` and `sadc` retain their segmentation rules.

Original input transcript JSON files require no changes.
